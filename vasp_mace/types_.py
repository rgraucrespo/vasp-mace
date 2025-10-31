from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class IncarConfig:
    EDIFFG: float      # <0 force tol (eV/Å); >0 energy tol (eV)
    NSW: int           # max ionic steps
    ISIF: int          # 2 positions-only, 3 variable-cell
    PSTRESS: float     # pressure in kB
    raw: Dict[str, str]

@dataclass
class StepRecord:
    n: int
    energy: float
    dE: float
    fmax: float
