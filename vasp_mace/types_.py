from dataclasses import dataclass
from typing import Dict

@dataclass
class IncarConfig:
    EDIFFG: float      # <0 force tol (eV/Å); >0 energy tol (eV)
    NSW: int           # max ionic steps
    ISIF: int          # 2 positions-only, 3 variable-cell, 4 shape+atoms, 7 volume-only, 8 positions+volume
    PSTRESS: float     # pressure in kB
    IBRION: int        # -1 none, 0 MD, 1 LBFGS, 2 BFGS, 3 FIRE
    IVDW: int          # 0 none, 11 D3(zero), 12 D3(BJ), 13 D3(zero)+ATM, 14 D3(BJ)+ATM
    TEBEG: float       # MD: starting temperature (K)
    TEEND: float       # MD: ending temperature (K); -1 = same as TEBEG
    POTIM: float       # MD: timestep (fs)
    NBLOCK: int        # MD: write XDATCAR / trajectory every NBLOCK steps
    MDALGO: int        # MD: 1 NVE, 2 NVT Langevin
    SMASS: float       # MD: Langevin friction (ps^-1); <=0 uses default 0.01 fs^-1
    raw: Dict[str, str]


@dataclass
class MDRecord:
    n: int              # step index (1-based)
    energy_pot: float   # potential energy (eV)
    energy_kin: float   # kinetic energy (eV)
    temperature: float  # instantaneous temperature (K)
