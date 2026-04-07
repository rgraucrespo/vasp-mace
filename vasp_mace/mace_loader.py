import io
import warnings
import logging
from contextlib import redirect_stdout, redirect_stderr

# vasp_mace/mace_loader.py

def _silenced_import_mace():
    # Import MACECalculator with stdout/stderr and warnings silenced
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf), redirect_stderr(buf):
            from mace.calculators.mace import MACECalculator
    return MACECalculator

def load_calc(model_path: str, device: str = "auto", dtype: str = "auto", dispersion: bool = False):
    import os
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MACE model file not found: {model_path}")

    # Default to CPU/float64 for robustness on macOS (MPS lacks float64)
    if device == "auto":
        device = "cpu"
    if dtype == "auto":
        dtype = "float64" if device == "cpu" else "float32"

    # hush third-party loggers as an extra guard
    for name in ("cuequivariance", "cuequivariance_torch", "e3nn", "mace"):
        logging.getLogger(name).setLevel(logging.ERROR)

    MACECalculator = _silenced_import_mace()

    # Silence during calculator construction as well (some libs print here)
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf), redirect_stderr(buf):
            calc = MACECalculator(
                model_paths=[model_path],
                device=device,
                default_dtype=dtype,
                dispersion=dispersion,
            )

    return calc, device, dtype
