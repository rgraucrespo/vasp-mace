"""Parse the VASP-style INCAR subset supported by vasp-mace."""

import os
import re
from typing import List, Optional

import numpy as np

from .types_ import IncarConfig


def _first_token(v: object) -> str:
    """Return the first whitespace-delimited token from an INCAR value."""
    return re.split(r"\s+", str(v).strip())[0]


def _parse_float_token(v: object, key: str) -> float:
    """Parse the first token as a VASP-style float, including D exponents."""
    token = _first_token(v).replace("D", "E").replace("d", "e")
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {v!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite, got {v!r}")
    return value


def _to_int(v: object, default: int, key: Optional[str] = None) -> int:
    """Parse the first whitespace-delimited token as an integer."""
    if key is None:
        try:
            return int(_first_token(v))
        except (IndexError, ValueError):
            return default
    value = _parse_float_token(v, key)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer, got {v!r}")
    return int(value)


def _to_float(v: object, default: float, key: Optional[str] = None) -> float:
    """Parse the first whitespace-delimited token as a float."""
    if key is None:
        try:
            return _parse_float_token(v, "value")
        except (IndexError, ValueError):
            return default
    return _parse_float_token(v, key)


def _to_bool(v: object, default: bool, key: Optional[str] = None) -> bool:
    """Parse VASP-style logical values such as ``.TRUE.`` and ``F``."""
    try:
        s = _first_token(v).upper().strip(".")
    except IndexError:
        if key is None:
            return default
        raise ValueError(f"{key} must be a logical value, got {v!r}")
    if s in ("TRUE", "T"):
        return True
    if s in ("FALSE", "F"):
        return False
    if key is None:
        return default
    raise ValueError(f"{key} must be a logical value, got {v!r}")


def _to_float_list(
    v: object, default: List[float], key: Optional[str] = None
) -> List[float]:
    """Parse a whitespace-delimited list of floats."""
    parts = [p for p in re.split(r"\s+", str(v).strip()) if p]
    if not parts:
        return default
    values = []
    try:
        for part in parts:
            token = part.replace("D", "E").replace("d", "e")
            value = float(token)
            if not np.isfinite(value):
                raise ValueError
            values.append(value)
    except ValueError as exc:
        if key is None:
            return default
        raise ValueError(f"{key} must be a list of finite numbers, got {v!r}") from exc
    return values


def _get_int(raw: dict[str, str], key: str, default: int) -> int:
    """Read an integer INCAR tag, raising if a present value is malformed."""
    if key not in raw:
        return default
    return _to_int(raw[key], default, key)


def _get_float(raw: dict[str, str], key: str, default: float) -> float:
    """Read a float INCAR tag, raising if a present value is malformed."""
    if key not in raw:
        return default
    return _to_float(raw[key], default, key)


def _get_bool(raw: dict[str, str], key: str, default: bool) -> bool:
    """Read a logical INCAR tag, raising if a present value is malformed."""
    if key not in raw:
        return default
    return _to_bool(raw[key], default, key)


def _get_float_list(raw: dict[str, str], key: str, default: List[float]) -> List[float]:
    """Read a float-list INCAR tag, raising if a present value is malformed."""
    if key not in raw:
        return default
    return _to_float_list(raw[key], default, key)


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
    ediffg = _get_float(raw, "EDIFFG", -0.05)
    nsw = _get_int(raw, "NSW", 0)
    isif = _get_int(raw, "ISIF", 2)
    pstress = _get_float(raw, "PSTRESS", 0.0)
    ibrion = _get_int(raw, "IBRION", -1)
    ivdw = _get_int(raw, "IVDW", 0)
    tebeg = _get_float(raw, "TEBEG", 0.0)
    teend = _get_float(raw, "TEEND", -1.0)
    potim = _get_float(raw, "POTIM", 0.5)
    if potim <= 0:
        raise ValueError(f"POTIM must be positive, got {potim}")
    nblock = _get_int(raw, "NBLOCK", 1)
    if nblock < 1:
        print(f"[warn] NBLOCK={nblock} is invalid. Setting to 1.")
        nblock = 1
    mdalgo = _get_int(raw, "MDALGO", 3)
    andersen_prob = _get_float(raw, "ANDERSEN_PROB", 0.0)
    smass = _get_float(raw, "SMASS", -3.0)
    random_seed = _get_int(raw, "RANDOM_SEED", 0) if "RANDOM_SEED" in raw else None
    if random_seed is not None and random_seed < 0:
        raise ValueError(f"RANDOM_SEED must be non-negative, got {random_seed}")

    # LANGEVIN_GAMMA: if not provided, use SMASS if SMASS > 0, else 10 ps^-1 (VASP-like)
    # VASP default for LANGEVIN_GAMMA is actually 10.0 ps^-1 if MDALGO=3.
    # If SMASS is provided, we use it for Langevin as well if LANGEVIN_GAMMA is missing.
    lg_default = [smass] if smass > 0 else [10.0]
    langevin_gamma = np.array(_get_float_list(raw, "LANGEVIN_GAMMA", lg_default))

    # LANGEVIN_GAMMA_L: lattice friction for MDALGO=3, ISIF=3
    langevin_gamma_l = _get_float(raw, "LANGEVIN_GAMMA_L", 10.0)

    # PMASS: piston mass for Langevin NPT (amu); 0 = auto
    pmass = _get_float(raw, "PMASS", 0.0)
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

    # Validate IVDW. IVDW=13 selects DFT-D4 (VASP >= 6.2, external package).
    # D4 support is being added to vasp-mace on top of the `dftd4` backend but
    # is not wired yet, so reject it explicitly with an accurate message rather
    # than the previous (incorrect) "D3 + ATM three-body" label.
    if ivdw == 13:
        raise ValueError(
            "IVDW=13 (DFT-D4) is not yet implemented in vasp-mace. "
            "Use IVDW=11 (D3-zero) or IVDW=12 (D3-BJ) for dispersion."
        )
    if ivdw not in (0, 11, 12):
        raise ValueError(
            f"IVDW={ivdw} is not supported. "
            f"Supported values: 0 (none), 11 (D3-zero), 12 (D3-BJ)."
        )

    nfree = _get_int(raw, "NFREE", 2)
    if nfree not in (1, 2):
        print(f"[warn] NFREE={nfree} is not supported. Using NFREE=2.")
        nfree = 2

    images = _get_int(raw, "IMAGES", 0)
    spring = _get_float(raw, "SPRING", -5.0)
    if images > 0 and nsw < 1:
        raise ValueError(
            f"NSW={nsw} is not supported for NEB in vasp-mace. "
            "Use NSW >= 1 so at least one NEB optimization step is logged."
        )
    if images > 0 and spring >= 0.0:
        raise ValueError(
            f"SPRING={spring} is not supported for NEB in vasp-mace. "
            "Use a negative SPRING value (VASP NEB convention), e.g. SPRING=-5."
        )
    lclimb = _get_bool(raw, "LCLIMB", False)

    ml_lheat = _get_bool(raw, "ML_LHEAT", False)
    ml_heat_interval = _get_int(raw, "ML_HEAT_INTERVAL", 1)
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
        RANDOM_SEED=random_seed,
        NFREE=nfree,
        IMAGES=images,
        SPRING=spring,
        LCLIMB=lclimb,
        ML_LHEAT=ml_lheat,
        ML_HEAT_INTERVAL=ml_heat_interval,
        raw=raw,
    )
