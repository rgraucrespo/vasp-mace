"""Shared deterministic fixture for the heat-flux opt-in tests.

The mace-unfolded unfolder asserts that each cell length strictly exceeds the
effective MACE cutoff (``r_max × num_interactions``) so the cell only needs to
be replicated once per direction. For the typical MACE-MP-0 setting
(``r_max=6, num_interactions=2 → 12 Å``), the smallest fully-periodic cell
that satisfies this is a 3×3×3 supercell of cubic MgO (3 × 4.211 Å ≈
12.633 Å > 12 Å). Smaller models with shorter cutoffs would also fit smaller
cells, but using the worst-case supercell keeps the tests robust across
checkpoints.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def build_mgo_fixture() -> Tuple[Any, np.ndarray]:
    """Return ``(atoms, velocities)`` for the heat-flux regression fixture.

    The atoms object is a 3×3×3 supercell of cubic rocksalt MgO at
    ``a = 4.211 Å`` (216 atoms). Velocities are sampled from a fixed-seed
    NumPy generator so successive calls return bitwise-identical arrays.
    """
    from ase.build import bulk

    primitive = bulk("MgO", "rocksalt", a=4.211, cubic=True)
    atoms = primitive.repeat((3, 3, 3))
    rng = np.random.default_rng(42)
    velocities = rng.normal(0.0, 0.005, size=(len(atoms), 3))
    return atoms, velocities
