"""Parse the VASP-style INCAR subset supported by vasp-mace."""

import os
import re
from typing import List

import numpy as np

from .types_ import IncarConfig


def _to_int(v: object, default: int) -> int:
    """Parse the first whitespace-delimited token as an integer."""
    try:
        return int(re.split(r"\s+", str(v).strip())[0])
    except Exception:
        return default


def _to_float(v: object, default: float) -> float:
    """Parse the first whitespace-delimited token as a float."""
    try:
        return float(re.split(r"\s+", str(v).strip())[0])
    except Exception:
        return default


def _to_bool(v, default: bool) -> bool:
    """Parse VASP-style logical values such as ``.TRUE.`` and ``F``."""
    s = str(v).strip().upper().strip(".")
    if s in ("TRUE", "T"):
        return True
    if s in ("FALSE", "F"):
        return False
    return default


def _to_float_list(v: object, default: List[float]) -> List[float]:
    """Parse a whitespace-delimited list of floats."""
    try:
        # Split by any whitespace and filter out empty strings
        parts = [p for p in re.split(r"\s+", str(v).strip()) if p]
        if not parts:
            return default
        return [float(p) for p in parts]
    except Exception:
        return default


def parse_incar(path: str = "INCAR") -> IncarConfig:
    """Read a VASP-style INCAR file into an :class:`IncarConfig`.

    Only the INCAR tags implemented by vasp-mace are interpreted. Unknown tags
    are preserved in ``IncarConfig.raw`` so they can be echoed into output files,
    but they do not affect the calculation. Inline comments beginning with
    ``#`` or ``!`` are stripped before parsing.

    Parameters
    ----------
    path
        Path to the INCAR file to parse.

    Returns
    -------
    IncarConfig
        Parsed configuration with VASP-like defaults and normalized values. For
        example, ``ISIF=0`` and ``ISIF=1`` are coerced to ``ISIF=2`` because
        vasp-mace always computes the full stress tensor.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a supported tag has an invalid value, such as ``POTIM <= 0`` or an
        unsupported ``IVDW`` value.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"INCAR not found: {os.path.abspath(path)}")

    raw: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            # strip comments (# or !) and whitespace
            s = line.split("#", 1)[0].split("!", 1)[0].strip()
            if not s or "=" not in s:
                continue
            k, v = s.split("=", 1)
            raw[k.strip().upper()] = v.strip()

    # Parse with defaults (VASP-like)
    ediffg = _to_float(raw.get("EDIFFG", -0.05), -0.05)
    nsw = _to_int(raw.get("NSW", 0), 0)
    isif = _to_int(raw.get("ISIF", 2), 2)
    pstress = _to_float(raw.get("PSTRESS", 0.0), 0.0)
    ibrion = _to_int(raw.get("IBRION", -1), -1)
    ivdw = _to_int(raw.get("IVDW", 0), 0)
    tebeg = _to_float(raw.get("TEBEG", 0.0), 0.0)
    teend = _to_float(raw.get("TEEND", -1.0), -1.0)
    potim = _to_float(raw.get("POTIM", 0.5), 0.5)
    if potim <= 0:
        raise ValueError(f"POTIM must be positive, got {potim}")
    nblock = _to_int(raw.get("NBLOCK", 1), 1)
    if nblock < 1:
        print(f"[warn] NBLOCK={nblock} is invalid. Setting to 1.")
        nblock = 1
    mdalgo = _to_int(raw.get("MDALGO", 3), 3)
    andersen_prob = _to_float(raw.get("ANDERSEN_PROB", 0.0), 0.0)
    smass = _to_float(raw.get("SMASS", -3.0), -3.0)

    # LANGEVIN_GAMMA: if not provided, use SMASS if SMASS > 0, else 10 ps^-1 (VASP-like)
    # VASP default for LANGEVIN_GAMMA is actually 10.0 ps^-1 if MDALGO=3.
    # If SMASS is provided, we use it for Langevin as well if LANGEVIN_GAMMA is missing.
    lg_default = [smass] if smass > 0 else [10.0]
    langevin_gamma = np.array(_to_float_list(raw.get("LANGEVIN_GAMMA", ""), lg_default))

    # LANGEVIN_GAMMA_L: lattice friction for MDALGO=3, ISIF=3
    langevin_gamma_l = _to_float(raw.get("LANGEVIN_GAMMA_L", 10.0), 10.0)

    # PMASS: piston mass for Langevin NPT (amu); 0 = auto
    pmass = _to_float(raw.get("PMASS", 0.0), 0.0)
    if pmass < 0.0:
        print(
            f"[warn] PMASS={pmass} is negative. Using automatic piston mass (N × 10000 amu)."
        )
        pmass = 0.0

    # Coerce ISIF=0/1 → ISIF=2 (positions only, no cell DOF)
    # In VASP, 0/1/2 all fix the cell; they differ only in how much stress VASP computes
    # internally, which is irrelevant here (we always compute the full stress tensor).
    if isif in (0, 1):
        print(
            f"[warn] ISIF={isif} requested. Treating as ISIF=2 (positions only, no cell relaxation)."
        )
        isif = 2

    # Validate IVDW
    if ivdw not in (0, 11, 12, 13, 14):
        raise ValueError(
            f"IVDW={ivdw} is not supported. "
            f"Supported values: 0 (none), 11 (D3-zero), 12 (D3-BJ), "
            f"13 (D3-zero+ATM), 14 (D3-BJ+ATM)."
        )

    nfree = _to_int(raw.get("NFREE", 2), 2)
    if nfree not in (1, 2):
        print(f"[warn] NFREE={nfree} is not supported. Using NFREE=2.")
        nfree = 2

    images = _to_int(raw.get("IMAGES", 0), 0)
    spring = _to_float(raw.get("SPRING", -5.0), -5.0)
    lclimb = _to_bool(raw.get("LCLIMB", False), False)

    ml_lheat = _to_bool(raw.get("ML_LHEAT", False), False)
    ml_heat_interval = _to_int(raw.get("ML_HEAT_INTERVAL", 1), 1)
    if ml_heat_interval < 1:
        print(f"[warn] ML_HEAT_INTERVAL={ml_heat_interval} is invalid. Setting to 1.")
        ml_heat_interval = 1

    return IncarConfig(
        EDIFFG=ediffg,
        NSW=nsw,
        ISIF=isif,
        PSTRESS=pstress,
        IBRION=ibrion,
        IVDW=ivdw,
        TEBEG=tebeg,
        TEEND=teend,
        POTIM=potim,
        NBLOCK=nblock,
        MDALGO=mdalgo,
        ANDERSEN_PROB=andersen_prob,
        LANGEVIN_GAMMA=langevin_gamma,
        LANGEVIN_GAMMA_L=langevin_gamma_l,
        SMASS=smass,
        PMASS=pmass,
        NFREE=nfree,
        IMAGES=images,
        SPRING=spring,
        LCLIMB=lclimb,
        ML_LHEAT=ml_lheat,
        ML_HEAT_INTERVAL=ml_heat_interval,
        raw=raw,
    )
