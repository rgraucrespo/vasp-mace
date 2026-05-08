from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class IncarConfig:
    EDIFFG: float  # <0 force tol (eV/Å); >0 energy tol (eV)
    NSW: int  # max ionic steps
    ISIF: int  # 2 positions-only, 3 variable-cell, 4 shape+atoms, 7 volume-only, 8 positions+volume
    PSTRESS: float  # pressure in kB
    IBRION: int  # -1 none, 0 MD; relaxation: 1 LBFGS, 2 BFGS, 3 FIRE; NEB: 1/2 MDMin, 3 FIRE
    IVDW: int  # 0 none, 11 D3(zero), 12 D3(BJ), 13 D3(zero)+ATM, 14 D3(BJ)+ATM
    TEBEG: float  # MD: starting temperature (K)
    TEEND: float  # MD: ending temperature (K); -1 = same as TEBEG
    POTIM: float  # MD: timestep (fs)
    NBLOCK: int  # MD: write XDATCAR / trajectory every NBLOCK steps
    MDALGO: int  # MD: 1 NVE/Andersen, 2 Nose-Hoover, 3 Langevin
    ANDERSEN_PROB: float  # MD: collision probability for Andersen thermostat
    LANGEVIN_GAMMA: (
        np.ndarray
    )  # MD: friction coefficients (ps^-1) for Langevin; one value per species or a scalar
    LANGEVIN_GAMMA_L: float  # MD: lattice friction coefficient (ps^-1) for Langevin NPT
    SMASS: float  # MD: Nose-Hoover mass or Langevin friction (ps^-1); <=0 uses default 0.01 fs^-1
    PMASS: float = (
        0.0  # MD: piston mass (amu) for Langevin NPT; 0 = auto (N × 10000 amu)
    )
    NFREE: int = 2  # phonons: displacements per DOF (1 = +only, 2 = ±central)
    IMAGES: int = 0  # NEB: number of intermediate images
    SPRING: float = (
        -5.0
    )  # NEB: spring constant; SPRING < 0 (NEB convention); k = |SPRING| eV/Å^2
    LCLIMB: bool = False  # NEB: enable climbing-image (CI-NEB)
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class MDRecord:
    n: int  # step index (1-based)
    energy_pot: float  # potential energy (eV)
    energy_kin: float  # kinetic energy (eV)
    temperature: float  # instantaneous temperature (K)
