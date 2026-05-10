"""Shared deterministic fixture for the heat-flux opt-in tests.

The mace-unfolded unfolder asserts that each cell length strictly exceeds the
effective MACE cutoff (``r_max × num_interactions``) so the cell only needs
to be replicated once per direction. For the typical MACE-MP-0 setting
(``r_max=6, num_interactions=2 → 12 Å``), a 2×2×2 supercell of cubic
rocksalt PbTe at ``a = 6.46 Å`` gives perpendicular heights of 12.92 Å —
just above the bound — at only 64 atoms (1728 atoms in the unfolded graph).
PbTe is the user's actual target system for Green-Kubo thermal-conductivity
work, so the fixture doubles as a smaller-than-production sanity-check
geometry that fits comfortably on a 16 GB GPU once the small MACE-MP-0
checkpoint is used.

The 12.92 Å height *fails* the stricter production bound
``L > 2·n·r + 2 = 26 Å`` enforced by
:func:`vasp_mace.heat.validate_3d_bulk_cell`. The opt-in tests therefore
pass ``cell_size_margin=-100.0`` to bypass that production guard for
testing only.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def build_pbte_fixture() -> Tuple[Any, np.ndarray]:
    """Return ``(atoms, velocities)`` for the heat-flux regression fixture.

    The atoms object is a 2×2×2 supercell of cubic rocksalt PbTe at
    ``a = 6.46 Å`` (64 atoms). Velocities are sampled from a fixed-seed
    NumPy generator so successive calls return bitwise-identical arrays.
    """
    from ase.build import bulk

    primitive = bulk("PbTe", "rocksalt", a=6.46, cubic=True)
    atoms = primitive.repeat((2, 2, 2))
    rng = np.random.default_rng(42)
    velocities = rng.normal(0.0, 0.005, size=(len(atoms), 3))
    return atoms, velocities
