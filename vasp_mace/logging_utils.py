from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class StepRecord:
    n: int          # ionic step index (1-based)
    energy: float   # total E (eV)
    dE: float       # delta E since previous ionic step (eV)
    fmax: float     # max |F| (eV/Å)
    positions: Optional[np.ndarray] = field(default=None)  # (N, 3) Cartesian Å
    forces: Optional[np.ndarray] = field(default=None)     # (N, 3) eV/Å
    stress: Optional[np.ndarray] = field(default=None)     # (6,) eV/Å³ Voigt [xx,yy,zz,yz,xz,xy]
    cell: Optional[np.ndarray] = field(default=None)       # (3, 3) Å


class StepLogger:
    def __init__(self):
        self.steps: List[StepRecord] = []
        self._last_E = None

    def log(self, n: int, energy: float, forces, atoms=None):
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        dE = 0.0 if self._last_E is None else energy - self._last_E
        self._last_E = energy

        positions = None
        cell = None
        stress = None
        forces_arr = np.asarray(forces, dtype=float).copy()

        if atoms is not None:
            positions = atoms.get_positions().copy()
            cell = np.array(atoms.get_cell()).copy()
            try:
                stress = atoms.get_stress(voigt=True).copy()
            except Exception:
                pass

        rec = StepRecord(
            n=n, energy=energy, dE=dE, fmax=fmax,
            positions=positions, forces=forces_arr,
            stress=stress, cell=cell,
        )
        self.steps.append(rec)
        return rec
