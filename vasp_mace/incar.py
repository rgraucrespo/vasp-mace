import os, re
from .types_ import IncarConfig

def _to_int(v, default):
    try:
        return int(re.split(r"\s+", str(v).strip())[0])
    except Exception:
        return default

def _to_float(v, default):
    try:
        return float(re.split(r"\s+", str(v).strip())[0])
    except Exception:
        return default

def parse_incar(path: str = "INCAR") -> IncarConfig:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"INCAR not found: {os.path.abspath(path)}")

    raw = {}
    with open(path) as fh:
        for line in fh:
            # strip comments (# or !) and whitespace
            s = line.split("#", 1)[0].split("!", 1)[0].strip()
            if not s or "=" not in s:
                continue
            k, v = s.split("=", 1)
            raw[k.strip().upper()] = v.strip()

    # Parse with defaults (VASP-like)
    ediffg  = _to_float(raw.get("EDIFFG",  -0.05), -0.05)
    nsw     = _to_int(raw.get("NSW",    0),     0)
    isif    = _to_int(raw.get("ISIF",   2),     2)
    pstress = _to_float(raw.get("PSTRESS", 0.0),  0.0)
    ibrion  = _to_int(raw.get("IBRION", -1),   -1)
    ivdw    = _to_int(raw.get("IVDW",    0),    0)
    tebeg   = _to_float(raw.get("TEBEG",   0.0),  0.0)
    teend   = _to_float(raw.get("TEEND",  -1.0), -1.0)
    potim   = _to_float(raw.get("POTIM",   0.5),  0.5)
    if potim <= 0:
        raise ValueError(f"POTIM must be positive, got {potim}")
    nblock  = _to_int(raw.get("NBLOCK",  1),    1)
    if nblock < 1:
        print(f"[warn] NBLOCK={nblock} is invalid. Setting to 1.")
        nblock = 1
    mdalgo  = _to_int(raw.get("MDALGO",  2),    2)
    smass   = _to_float(raw.get("SMASS",  -3.0), -3.0)

    # Coerce ISIF=0 → ISIF=2 (no cell relaxation, stress still computed)
    if isif == 0:
        print("[warn] ISIF=0 requested. Treating as ISIF=2 (no cell relaxation).")
        isif = 2

    # Validate IVDW
    if ivdw not in (0, 11, 12, 13, 14):
        raise ValueError(
            f"IVDW={ivdw} is not supported. "
            f"Supported values: 0 (none), 11 (D3-zero), 12 (D3-BJ), "
            f"13 (D3-zero+ATM), 14 (D3-BJ+ATM)."
        )

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
        SMASS=smass,
        raw=raw,
    )
