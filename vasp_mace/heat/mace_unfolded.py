"""MACE unfolded-cell heat-flux backend.

Thin adapter around
:class:`mace_unfolded.unfolded_heat.unfolder_calculator.UnfoldedHeatFluxCalculator`
exposed through the :class:`vasp_mace.heat.heat_flux.HeatFluxCalculator`
interface. Install the underlying packages from ``requirements-heat.txt`` in a
source checkout, or directly from their GitHub repositories.

This implements the *potential* term of the heat flux only. For non-diffusive
solids that omission is usually a small effect; ``flux_type`` is recorded in the
``ML_HEAT.json`` sidecar so downstream tools (e.g. ``sportran``) know what was
written. Convective and gauge-fixed flavours are deliberately deferred (see
``not_for_release/ml_heat_implementation_instructions.md``).

**Scope of the first release.** Heat-flux output is restricted to fixed-cell
NVE production MD with the bare MACE potential (``IVDW = 0``). Only fully
periodic 3D bulk solids in sufficiently large supercells are supported.
Concretely, every perpendicular cell height must exceed
``2 × num_message_passing_layers × r_cutoff + cell_size_margin`` (default
margin 2 Å). Slabs, wires, molecules, and small primitive cells are rejected
with a clear ``ValueError`` rather than silently returning a wrong flux. The
restriction matches the user's actual workflow — equilibrium MD for
Green-Kubo thermal conductivity already needs supercells of this size — and
keeps the unfolded-cell algorithm in the regime where it is well-defined.

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
from typing import Any, Optional

import numpy as np

from .heat_flux import HeatFluxCalculator, INSTALL_HINT


# mace-unfolded works internally with velocities in Å/ps and divides by the
# cell volume, returning J in eV / (Å² · ps). vasp-mace stores the *extensive*
# heat flux in the VASP-style ``ML_HEAT`` convention of eV · Å · fs⁻¹. The
# conversion is therefore J · V · (1 ps / 1000 fs).
_PS_TO_FS = 1000.0


def validate_3d_bulk_cell(
    atoms: Any,
    r_cutoff: float,
    num_message_passing: int,
    margin: float = 2.0,
) -> None:
    """Reject inputs that violate the documented 3D-bulk-solid scope.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` object to validate.
    r_cutoff
        MACE radial cutoff in Å (the model's ``r_max`` buffer).
    num_message_passing
        Number of MACE message-passing layers (``num_interactions``).
    margin
        Safety margin in Å. Each perpendicular cell height must strictly
        exceed ``2 × num_message_passing × r_cutoff + margin``.

    Raises
    ------
    ValueError
        If ``atoms.pbc`` is not fully periodic, or any perpendicular cell
        height fails the precondition.
    """
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if not pbc.all():
        raise ValueError(
            "ML_LHEAT only supports fully periodic 3D systems "
            f"(atoms.pbc must be [True, True, True]); got {pbc.tolist()}"
        )

    cell = np.asarray(atoms.cell.array, dtype=float)
    volume = float(abs(np.linalg.det(cell)))
    heights = np.array(
        [
            volume / float(np.linalg.norm(np.cross(cell[1], cell[2]))),
            volume / float(np.linalg.norm(np.cross(cell[2], cell[0]))),
            volume / float(np.linalg.norm(np.cross(cell[0], cell[1]))),
        ]
    )
    bound = 2.0 * num_message_passing * r_cutoff + margin
    if heights.min() <= bound:
        raise ValueError(
            "Heat-flux calculation requires every perpendicular cell "
            "height to exceed "
            f"2 × n_layers × r_cutoff + margin = "
            f"2 × {num_message_passing} × {r_cutoff:.3f} Å + "
            f"{margin:.3f} Å = {bound:.3f} Å; got "
            f"heights = {heights.tolist()} Å. Use a larger supercell "
            "(this release deliberately rejects slabs, wires, "
            "molecules, and small primitive cells)."
        )


@contextlib.contextmanager
def _suppress_unfolder_artefact_files(scratch_dir: Optional[str] = None):
    """Run a block in a scratch directory.

    ``UnfoldedHeatFluxCalculator.calculate`` unconditionally writes a
    ``POSCAR_unfolding`` file to ``cwd`` (and ``POSCAR_unfolded`` when its
    ``debug`` flag is set). For per-MD-step heat-flux evaluation we do not want
    to pollute the user's run directory, so we briefly chdir to a tempdir and
    discard whatever the upstream call leaves behind. The known artefact writes
    are patched to no-op inside this context; the chdir remains as a guard
    against future upstream side effects. ``scratch_dir`` lets long MD runs
    reuse one hidden scratch directory instead of creating and removing a
    temporary directory at every heat-flux step.
    """
    import ase.io

    cwd = os.getcwd()
    original_write = ase.io.write

    def _write_without_unfolder_artefacts(filename, *args, **kwargs):
        try:
            name = os.fspath(filename)
        except TypeError:
            name = str(filename)
        if os.path.basename(name) in {"POSCAR_unfolding", "POSCAR_unfolded"}:
            return None
        return original_write(filename, *args, **kwargs)

    if scratch_dir is None:
        with tempfile.TemporaryDirectory(prefix="vasp_mace_heat_") as scratch:
            os.chdir(scratch)
            ase.io.write = _write_without_unfolder_artefacts
            try:
                yield
            finally:
                ase.io.write = original_write
                os.chdir(cwd)
        return

    os.makedirs(scratch_dir, exist_ok=True)
    try:
        os.chdir(scratch_dir)
        ase.io.write = _write_without_unfolder_artefacts
        yield
    finally:
        ase.io.write = original_write
        os.chdir(cwd)


def _tensor_to_numpy_1d(value: Any) -> np.ndarray:
    """Convert a torch/numpy/list vector to a 1-D float64 NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def _model_device_dtype(torch_model: Any, device: str, dtype: str) -> tuple[str, str]:
    """Resolve device/dtype strings for a caller-supplied torch model."""
    if device == "auto" or dtype == "auto":
        try:
            first_param = next(torch_model.parameters())
        except StopIteration:
            first_param = None
        if device == "auto":
            device = first_param.device.type if first_param is not None else "cpu"
        if dtype == "auto":
            if first_param is not None and str(first_param.dtype).endswith("float32"):
                dtype = "float32"
            else:
                dtype = "float64"

    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}")

    try:
        if dtype == "float64":
            torch_model.double()
        else:
            torch_model.float()
        torch_model.to(device)
    except Exception as exc:
        if device not in ("cuda", "mps"):
            raise
        print(
            f"[warning] {device.upper()} heat-flux model transfer failed ({exc}); "
            "falling back to CPU/float64."
        )
        device = "cpu"
        dtype = "float64"
        torch_model.double()
        torch_model.to(device)

    return device, dtype


class MACEUnfoldedHeatFluxCalculator(HeatFluxCalculator):
    """Potential heat-flux calculator backed by ``mace-unfolded``.

    Loads a MACE checkpoint via :func:`vasp_mace.mace_loader.load_calc` unless
    the caller supplies an already-loaded raw MACE torch model. The latter path
    is used by ``run_md`` so ``ML_LHEAT`` does not keep a second copy of the
    model in memory.

    Parameters
    ----------
    model_path
        Path to a MACE ``.model`` checkpoint. Used for loading when
        ``torch_model`` is not supplied and recorded in metadata by callers.
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
    cell_size_margin
        Safety margin (in Å) added to the perpendicular-cell-height
        precondition checked at every :meth:`compute` call. Each height must
        strictly exceed ``2 × num_message_passing_layers × r_cutoff +
        cell_size_margin`` or :meth:`compute` raises ``ValueError``. The
        default of ``2.0`` Å is generous and matches the documented scope
        (3D bulk solids in supercells big enough for Green-Kubo). Negative
        values relax the bound and are intended only for unit tests.
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
    torch_model
        Optional raw MACE torch model, typically ``main_calc.models[0]`` from
        the ASE calculator already attached to the MD atoms. Supplying this
        avoids loading the same checkpoint twice.

    Raises
    ------
    ImportError
        If ``mace-unfolded`` is not installed.
    ValueError
        If ``flux_type`` is not ``"potential"``.
    """

    def __init__(
        self,
        model_path: str,
        flux_type: str = "potential",
        device: str = "auto",
        dtype: str = "float64",
        cell_size_margin: float = 2.0,
        forward: bool = False,
        torch_model: Optional[Any] = None,
    ) -> None:
        if flux_type != "potential":
            raise ValueError(
                f"flux_type={flux_type!r} is not supported; only 'potential' "
                "is implemented in this release"
            )

        try:
            from mace_unfolded.unfolded_heat.unfolder_calculator import (
                UnfoldedHeatFluxCalculator,
            )
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc

        if torch_model is None:
            from ..mace_loader import load_calc

            calc, resolved_device, resolved_dtype = load_calc(
                model_path, device=device, dtype=dtype
            )
            # MACECalculator stores the underlying torch model in ``models[0]``;
            # mace-unfolded operates on that raw model rather than the calculator
            # so it can read per-atom energies from the unfolded graph.
            torch_model = calc.models[0]
            uses_shared_model = False
        else:
            resolved_device, resolved_dtype = _model_device_dtype(
                torch_model, device=device, dtype=dtype
            )
            uses_shared_model = True

        # Read the model's interaction radius and message-passing depth so
        # ``compute()`` can validate the cell size on every call.
        r_cutoff = float(torch_model.get_buffer("r_max").cpu())
        num_message_passing = int(torch_model.get_buffer("num_interactions").cpu())

        self._unf = UnfoldedHeatFluxCalculator(
            torch_model,
            device=resolved_device,
            dtype=resolved_dtype,
            forward=forward,
            pbc=[True, True, True],
        )
        self._device = resolved_device
        self._dtype = resolved_dtype
        self._flux_type = flux_type
        self._model_path = model_path
        self._forward = forward
        self._r_cutoff = r_cutoff
        self._num_message_passing = num_message_passing
        self._cell_size_margin = float(cell_size_margin)
        self._uses_shared_model = uses_shared_model
        self._scratch = tempfile.TemporaryDirectory(prefix="vasp_mace_heat_")

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def flux_type(self) -> str:
        return self._flux_type

    @property
    def forward(self) -> bool:
        return self._forward

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def uses_shared_model(self) -> bool:
        """Whether the backend reuses the main MD calculator's torch model."""
        return self._uses_shared_model

    @property
    def r_cutoff(self) -> float:
        """MACE radial cutoff (Å) read from the model's ``r_max`` buffer."""
        return self._r_cutoff

    @property
    def num_message_passing(self) -> int:
        """Number of MACE message-passing layers (``num_interactions``)."""
        return self._num_message_passing

    @property
    def cell_size_margin(self) -> float:
        """Safety margin (Å) on top of ``2 × n_layers × r_cutoff``."""
        return self._cell_size_margin

    def compute(self, atoms: Any, velocities: np.ndarray) -> np.ndarray:
        """Return the potential heat flux for one MD frame.

        Parameters
        ----------
        atoms
            ASE ``Atoms`` object for the cell. Must be fully periodic and
            satisfy the perpendicular-cell-height precondition (see
            ``cell_size_margin``).
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

        validate_3d_bulk_cell(
            atoms,
            self._r_cutoff,
            self._num_message_passing,
            self._cell_size_margin,
        )

        # Don't mutate the caller's atoms — mace-unfolded reads velocities via
        # ``atoms.get_velocities()`` internally.
        atoms_for_calc = atoms.copy()
        atoms_for_calc.set_velocities(v)

        with _suppress_unfolder_artefact_files(self._scratch.name):
            results = self._unf.calculate(atoms_for_calc)

        # Prefer the extensive numerator directly. Upstream also returns
        # ``heat_flux = numerator / volume``; using the numerator avoids
        # depending on upstream's cached volume and is identical for fixed-cell
        # NVE, the only ensemble accepted for ML_LHEAT.
        if (
            "heat_flux_potential_term" in results
            and "heat_flux_force_term" in results
        ):
            flux = _tensor_to_numpy_1d(
                results["heat_flux_potential_term"]
                - results["heat_flux_force_term"]
            )
        else:
            flux = _tensor_to_numpy_1d(results["heat_flux"]) * float(atoms.get_volume())
        if flux.size != 3:
            raise RuntimeError(
                f"mace-unfolded returned heat flux of length {flux.size}; "
                "expected 3 components for fully periodic systems"
            )

        return flux / _PS_TO_FS

    def close(self) -> None:
        """Release the persistent scratch directory used for upstream artefacts."""
        scratch = getattr(self, "_scratch", None)
        if scratch is not None:
            scratch.cleanup()
            self._scratch = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
