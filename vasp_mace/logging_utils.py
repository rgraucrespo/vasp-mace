import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List
import numpy as np

@contextmanager
def timer(label=""):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[time] {label}: {dt:.3f}s")


@dataclass
class StepRecord:
    n: int        # ionic step index (1-based)
    energy: float # total E (eV)
    dE: float     # delta E since previous ionic step (eV)
    fmax: float   # max |F| (eV/Å)

class StepLogger:
    def __init__(self):
        self.steps: List[StepRecord] = []
        self._last_E = None

    def log(self, n: int, energy: float, forces):
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        if self._last_E is None:
            dE = 0.0
        else:
            dE = energy - self._last_E
        self._last_E = energy
        rec = StepRecord(n=n, energy=energy, dE=dE, fmax=fmax)
        self.steps.append(rec)
        return rec
