"""MACE unfolded-cell heat-flux backend.

Thin adapter around
:class:`mace_unfolded.unfolded_heat.unfolder_calculator.UnfoldedHeatFluxCalculator`
exposed through the :class:`vasp_mace.heat.heat_flux.HeatFluxCalculator`
interface. Install the underlying package with ``pip install vasp-mace[heat]``.

This implements the *potential* term of the heat flux only. For non-diffusive
solids that omission is usually a small effect; ``flux_type`` is recorded in the
``ML_HEAT.json`` sidecar so downstream tools (e.g. ``sportran``) know what was
written. Convective and gauge-fixed flavours are deliberately deferred (see
``not_for_release/ml_heat_implementation_instructions.md``).

References
----------
* M. F. Langer *et al.*, Phys. Rev. B 108, L100302.
* S. Wieser, Y. Cen, G. K. H. Madsen, J. Carrete, *J. Chem. Theory Comput.*
  22, 513 (2026); arXiv:2509.08573.
* Upstream package: https://github.com/pulgon-project/mace-unfolded.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any, Sequence, Tuple

import numpy as np

from .heat_flux import HeatFluxCalculator, INSTALL_HINT


# mace-unfolded works internally with velocities in Å/ps and divides by the
# cell volume, returning J in eV / (Å² · ps). vasp-mace stores the *extensive*
# heat flux in the VASP-style ``ML_HEAT`` convention of eV · Å · fs⁻¹. The
# conversion is therefore J · V · (1 ps / 1000 fs).
_PS_TO_FS = 1000.0


@contextlib.contextmanager
def _suppress_unfolder_artefact_files():
    """Run a block in a scratch directory.

    ``UnfoldedHeatFluxCalculator.calculate`` unconditionally writes a
    ``POSCAR_unfolding`` file to ``cwd`` (and ``POSCAR_unfolded`` when its
    ``debug`` flag is set). For per-MD-step heat-flux evaluation we do not want
    to pollute the user's run directory, so we briefly chdir to a tempdir and
    discard whatever the upstream call leaves behind.
    """
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="vasp_mace_heat_") as scratch:
        os.chdir(scratch)
        try:
            yield
        finally:
            os.chdir(cwd)


class MACEUnfoldedHeatFluxCalculator(HeatFluxCalculator):
    """Potential heat-flux calculator backed by ``mace-unfolded``.

    Loads a MACE checkpoint via :func:`vasp_mace.mace_loader.load_calc` so
    device/dtype resolution and CUDA/MPS→CPU fallback match the rest of the
    package, then drives the upstream unfolded calculator on each call to
    :meth:`compute`.

    Parameters
    ----------
    model_path
        Path to a MACE ``.model`` checkpoint.
    flux_type
        Only ``"potential"`` is supported in this release. Convective and
        gauge-fixed variants are deferred.
    device
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``. ``"auto"`` follows the
        same precedence as :func:`vasp_mace.mace_loader.load_calc`.
    dtype
        Floating-point dtype. Defaults to ``"float64"`` for heat-flux work;
        ``"float32"`` is allowed but discouraged because of numerical noise in
        the autograd derivatives.
    pbc
        Periodic-boundary flags forwarded to the unfolder. Stage 2 supports
        only fully periodic systems; non-periodic configurations are rejected.
    forward
        Whether to use forward-mode autodiff inside ``mace-unfolded``.
        Defaults to ``False`` (reverse-mode, three ``grad`` passes per call)
        because the forward-mode (``functorch.jvp``) path in mace-unfolded
        currently fails with ``mace-torch`` ≥ 0.3.10: ``mace.modules.utils
        .prepare_graph`` calls ``data["positions"].requires_grad_(True)``
        inside ``model.forward``, which functorch transforms forbid.
        ``forward=True`` is the upstream production-script default and is
        several times faster when it works, so keep it on the radar — once
        the upstream incompatibility is patched (vendor a compatible
        ``mace-torch`` pin, or a fix lands in mace-unfolded), flip the
        default. Reverse mode is workable on a GPU; on CPU it can take
        many minutes per call for the cell sizes the unfolder requires.

    Raises
    ------
    ImportError
        If ``mace-unfolded`` is not installed.
    ValueError
        If ``flux_type`` is not ``"potential"`` or ``pbc`` is not fully
        periodic.
    """

    def __init__(
        self,
        model_path: str,
        flux_type: str = "potential",
        device: str = "auto",
        dtype: str = "float64",
        pbc: Sequence[bool] = (True, True, True),
        forward: bool = False,
    ) -> None:
        if flux_type != "potential":
            raise ValueError(
                f"flux_type={flux_type!r} is not supported; only 'potential' "
                "is implemented in this release"
            )
        pbc_tuple: Tuple[bool, bool, bool] = tuple(bool(p) for p in pbc)  # type: ignore[assignment]
        if pbc_tuple != (True, True, True):
            raise ValueError(
                f"pbc={pbc_tuple} is not supported; stage 2 of vasp-mace's "
                "heat-flux backend only handles fully periodic systems"
            )

        try:
            from mace_unfolded.unfolded_heat.unfolder_calculator import (
                UnfoldedHeatFluxCalculator,
            )
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc

        from ..mace_loader import load_calc

        calc, resolved_device, resolved_dtype = load_calc(
            model_path, device=device, dtype=dtype
        )
        # MACECalculator stores the underlying torch model in ``models[0]``;
        # mace-unfolded operates on that raw model rather than the calculator
        # so it can read per-atom energies from the unfolded graph.
        torch_model = calc.models[0]

        self._unf = UnfoldedHeatFluxCalculator(
            torch_model,
            device=resolved_device,
            dtype=resolved_dtype,
            forward=forward,
            pbc=list(pbc_tuple),
        )
        self._device = resolved_device
        self._dtype = resolved_dtype
        self._pbc = pbc_tuple
        self._flux_type = flux_type
        self._model_path = model_path
        self._forward = forward

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def pbc(self) -> Tuple[bool, bool, bool]:
        return self._pbc

    @property
    def flux_type(self) -> str:
        return self._flux_type

    @property
    def forward(self) -> bool:
        return self._forward

    @property
    def model_path(self) -> str:
        return self._model_path

    def compute(self, atoms: Any, velocities: np.ndarray) -> np.ndarray:
        """Return the potential heat flux for one MD frame.

        Parameters
        ----------
        atoms
            ASE ``Atoms`` object for the cell.
        velocities
            Velocity array in ASE units (Å × √(eV/u)), shape
            ``(len(atoms), 3)``. The supplied array is what the upstream
            calculator will see — the caller's ``atoms`` is not mutated.

        Returns
        -------
        numpy.ndarray
            Heat-flux vector ``[qx, qy, qz]`` in ``eV·Å·fs⁻¹``, shape ``(3,)``.
        """
        v = np.asarray(velocities, dtype=float)
        if v.ndim != 2 or v.shape != (len(atoms), 3):
            raise ValueError(
                f"velocities must have shape ({len(atoms)}, 3); got {v.shape}"
            )

        # Don't mutate the caller's atoms — mace-unfolded reads velocities via
        # ``atoms.get_velocities()`` internally.
        atoms_for_calc = atoms.copy()
        atoms_for_calc.set_velocities(v)

        with _suppress_unfolder_artefact_files():
            results = self._unf.calculate(atoms_for_calc)

        flux = np.asarray(results["heat_flux"], dtype=float).reshape(-1)
        if flux.size != 3:
            raise RuntimeError(
                f"mace-unfolded returned heat flux of length {flux.size}; "
                "expected 3 components for fully periodic systems"
            )

        volume = float(atoms.get_volume())
        return flux * volume / _PS_TO_FS
