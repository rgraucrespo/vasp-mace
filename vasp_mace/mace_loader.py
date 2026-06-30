import io
import warnings
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Optional, Tuple

# vasp_mace/mace_loader.py


# IVDW -> torch-dftd D3 damping name. None means "no dispersion".
_IVDW_DAMPING = {0: None, 11: "zero", 12: "bj"}

# IVDW value selecting DFT-D4 (VASP >= 6.2), provided by the dftd4 backend.
_IVDW_D4 = 13

# xc functional passed to torch-dftd. Hardcoded to PBE: vasp-mace MACE models
# are trained against PBE references, and exposing this as an INCAR tag would
# invite mismatches with the underlying potential.
_D3_XC = "pbe"

# DFT-D4 functional, fixed to PBE for the same reason as D3.
_D4_METHOD = "pbe"


def _silenced_import_mace() -> Any:
    # Import MACECalculator with stdout/stderr and warnings silenced
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf), redirect_stderr(buf):
            from mace.calculators.mace import MACECalculator
    return MACECalculator


def load_calc(
    model_path: str,
    device: str = "auto",
    dtype: str = "auto",
    ivdw: int = 0,
    lsol: bool = False,
    tau: float = 0.525,
) -> Tuple[Any, str, str]:
    """Load a MACE calculator with device and dtype resolution.

    When ``ivdw`` selects a DFT-D3 flavor, the returned object is an
    ``ase.calculators.mixing.SumCalculator`` wrapping the MACE calculator and
    a ``torch_dftd.TorchDFTD3Calculator`` (xc=PBE, cutoff=40 Bohr). Passing
    ``dispersion=True`` directly into ``MACECalculator`` is a silent no-op in
    mace-torch 0.3.x — the kwarg is swallowed by ``Calculator.__init__`` — so
    D3 must be added on top.

    Parameters
    ----------
    model_path
        Path to a MACE ``.model`` checkpoint.
    device
        Requested execution device: ``"auto"``, ``"cpu"``, ``"cuda"``, or
        ``"mps"``. ``"auto"`` prefers CUDA, then MPS, then CPU.
    dtype
        Requested floating-point dtype: ``"auto"``, ``"float32"``, or
        ``"float64"``. ``"auto"`` uses ``float64`` on CPU and ``float32`` on
        accelerator devices.
    ivdw
        IVDW selector: ``0`` disables dispersion, ``11`` adds D3 with
        zero-damping, ``12`` adds D3 with Becke-Johnson damping, and ``13``
        adds DFT-D4 via the ``dftd4`` backend (periodic, xc=PBE). Other values
        are rejected by ``parse_incar`` upstream.
    lsol
        When true, add the implicit-solvation cavitation term
        (``SASASolvationCalculator``) on top of MACE (and any dispersion).
        Slab/cluster only; dense 3D bulk is rejected at evaluation time.
    tau
        Solvation surface tension in meV/Angstrom^2 (used only when ``lsol``).

    Returns
    -------
    tuple
        ``(calculator, resolved_device, resolved_dtype)``. The calculator is an
        ASE-compatible MACE calculator instance.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    """
    import os

    if ivdw not in _IVDW_DAMPING and ivdw != _IVDW_D4:
        raise ValueError(
            f"load_calc: IVDW={ivdw} is not supported; expected one of "
            f"{sorted(_IVDW_DAMPING) + [_IVDW_D4]}."
        )
    damping = _IVDW_DAMPING.get(ivdw)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MACE model file not found: {model_path}")

    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype == "auto":
        dtype = "float64" if device == "cpu" else "float32"

    # hush third-party loggers as an extra guard
    for name in ("cuequivariance", "cuequivariance_torch", "e3nn", "mace"):
        logging.getLogger(name).setLevel(logging.ERROR)

    MACECalculator = _silenced_import_mace()

    def _build_mace(dev, dt):
        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                return MACECalculator(
                    model_paths=[model_path],
                    device=dev,
                    default_dtype=dt,
                )

    def _build_calc(dev, dt):
        mace_calc = _build_mace(dev, dt)
        extras = []
        if ivdw == _IVDW_D4:
            extras.append(_make_d4_calc())
        elif damping is not None:
            extras.append(_make_d3_calc(dev, dt, damping))
        if lsol:
            extras.append(_make_solvation_calc(tau))
        if not extras:
            return mace_calc
        from ase.calculators.mixing import SumCalculator

        return SumCalculator([mace_calc, *extras])

    if device in ("cuda", "mps"):
        try:
            calc = _build_calc(device, dtype)
        except Exception as e:
            print(
                f"[warning] {device.upper()} device failed ({e}); falling back to CPU/float64."
            )
            device = "cpu"
            dtype = "float64"
            calc = _build_calc(device, dtype)
    else:
        calc = _build_calc(device, dtype)

    return calc, device, dtype


def _make_d3_calc(device: str, dtype: str, damping: str) -> Any:
    """Return a ``TorchDFTD3Calculator`` (xc=PBE, cutoff=40 Bohr)."""
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:
        raise RuntimeError(
            "IVDW>0 requires the optional torch-dftd package. Install it with "
            "`pip install torch-dftd` (see https://github.com/pfnet-research/torch-dftd)."
        ) from exc
    from ase.units import Bohr
    import torch

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    # 40 Bohr cutoff matches mace_mp()'s default; xc is fixed to PBE.
    return TorchDFTD3Calculator(
        device=device,
        damping=damping,
        dtype=torch_dtype,
        xc=_D3_XC,
        cutoff=40.0 * Bohr,
    )


def _make_d4_calc() -> Any:
    """Return a ``dftd4.ase.DFTD4`` calculator (xc=PBE, periodic-capable).

    DFT-D4 (``IVDW=13``) is provided by the Fortran-backed ``dftd4`` package,
    which supports periodic (3D bulk) systems and returns energy, forces, and
    stress. Unlike the torch D3 backend it runs on CPU and needs no device or
    dtype.
    """
    try:
        from dftd4.ase import DFTD4
    except ImportError as exc:
        raise RuntimeError(
            "IVDW=13 (DFT-D4) requires the optional dftd4 package. Install it "
            "with `pip install dftd4` (see requirements/dftd4.txt)."
        ) from exc

    return DFTD4(method=_D4_METHOD)


def _make_solvation_calc(tau: float) -> Any:
    """Return a ``SASASolvationCalculator`` for the nonpolar solvation term."""
    from .solvation import SASASolvationCalculator

    return SASASolvationCalculator(tau=tau)


def _wrap_with_d3(mace_calc: Any, device: str, dtype: str, damping: str) -> Any:
    """Return ``SumCalculator([mace_calc, TorchDFTD3Calculator(...)])``."""
    from ase.calculators.mixing import SumCalculator

    return SumCalculator([mace_calc, _make_d3_calc(device, dtype, damping)])


def _wrap_with_d4(mace_calc: Any) -> Any:
    """Return ``SumCalculator([mace_calc, DFTD4(method="pbe")])``."""
    from ase.calculators.mixing import SumCalculator

    return SumCalculator([mace_calc, _make_d4_calc()])
