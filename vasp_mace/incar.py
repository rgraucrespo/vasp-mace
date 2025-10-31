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
    ediffg = _to_float(raw.get("EDIFFG", -0.05), -0.05)
    nsw    = _to_int(raw.get("NSW", 0), 0)
    isif   = _to_int(raw.get("ISIF", 2), 2)
    pstress = _to_float(raw.get("PSTRESS", 0.0), 0.0)

    # Coerce ISIF=0 → ISIF=2 (compute stress even without cell relaxation)
    if isif == 0:
        print("[warn] ISIF=0 requested. Treating as ISIF=2 (no cell relaxation, stress computed).")
        isif = 2

    return IncarConfig(
        EDIFFG=ediffg,
        NSW=nsw,
        ISIF=isif,
        PSTRESS=pstress,
        raw=raw,
    )
