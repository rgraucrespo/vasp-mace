import io
import warnings
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Optional, Tuple

# vasp_mace/mace_loader.py


# IVDW -> torch-dftd damping name. None means "no dispersion".
_IVDW_DAMPING = {0: None, 11: "zero", 12: "bj"}

# xc functional passed to torch-dftd. Hardcoded to PBE: vasp-mace MACE models
# are trained against PBE references, and exposing this as an INCAR tag would
# invite mismatches with the underlying potential.
_D3_XC = "pbe"


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
        zero-damping, ``12`` adds D3 with Becke-Johnson damping. Other values
        are rejected by ``parse_incar`` upstream.

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

    if ivdw not in _IVDW_DAMPING:
        raise ValueError(
            f"load_calc: IVDW={ivdw} is not supported; expected one of "
            f"{sorted(_IVDW_DAMPING)}."
        )
    damping = _IVDW_DAMPING[ivdw]

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
        if damping is None:
            return mace_calc
        return _wrap_with_d3(mace_calc, dev, dt, damping)

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


def _wrap_with_d3(mace_calc: Any, device: str, dtype: str, damping: str) -> Any:
    """Return SumCalculator([mace_calc, TorchDFTD3Calculator(...)])."""
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:
        raise RuntimeError(
            "IVDW>0 requires the optional torch-dftd package. Install it with "
            "`pip install torch-dftd` (see https://github.com/pfnet-research/torch-dftd)."
        ) from exc
    from ase.calculators.mixing import SumCalculator
    from ase.units import Bohr
    import torch

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    # 40 Bohr cutoff matches mace_mp()'s default; xc is fixed to PBE.
    d3_calc = TorchDFTD3Calculator(
        device=device,
        damping=damping,
        dtype=torch_dtype,
        xc=_D3_XC,
        cutoff=40.0 * Bohr,
    )
    return SumCalculator([mace_calc, d3_calc])
