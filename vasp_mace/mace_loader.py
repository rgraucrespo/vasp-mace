import io
import warnings
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Tuple

# vasp_mace/mace_loader.py


def _silenced_import_mace() -> Any:
    # Import MACECalculator with stdout/stderr and warnings silenced
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf), redirect_stderr(buf):
            from mace.calculators.mace import MACECalculator
    return MACECalculator


def load_calc(
    model_path: str, device: str = "auto", dtype: str = "auto", dispersion: bool = False
) -> Tuple[Any, str, str]:
    """Load a MACE calculator with device and dtype resolution.

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
    dispersion
        Whether to enable the MACE calculator's DFT-D3 dispersion correction.

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

    def _build_calc(dev, dt):
        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                return MACECalculator(
                    model_paths=[model_path],
                    device=dev,
                    default_dtype=dt,
                    dispersion=dispersion,
                )

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
