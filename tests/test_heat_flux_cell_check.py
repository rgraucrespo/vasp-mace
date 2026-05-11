"""Unit tests for the heat-flux cell-size precondition.

These run in the default suite (no MACE model, no GPU, no
``mace_unfolded`` install) because they exercise only the validation
logic.
"""

from __future__ import annotations

import unittest

import numpy as np
from ase import Atoms

from vasp_mace.heat import validate_3d_bulk_cell


def _cubic(a: float, n_atoms: int = 1) -> Atoms:
    """Tiny periodic ``Atoms`` with a cubic cell and ``n_atoms`` Mg sites."""
    positions = np.zeros((n_atoms, 3))
    return Atoms(
        symbols=["Mg"] * n_atoms,
        positions=positions,
        cell=[a, a, a],
        pbc=True,
    )


class CellPreconditionTests(unittest.TestCase):
    # MACE-MP-0 numbers; bound = 2 × 2 × 6 + margin.
    R_CUTOFF = 6.0
    NUM_LAYERS = 2

    def test_passes_when_all_heights_above_bound(self) -> None:
        # 7×4.211 ≈ 29.477 Å > 2·2·6 + 2 = 26 Å.
        atoms = _cubic(29.5)
        validate_3d_bulk_cell(atoms, self.R_CUTOFF, self.NUM_LAYERS, margin=2.0)

    def test_rejects_cell_at_or_below_bound(self) -> None:
        # 3×4.211 ≈ 12.633 Å. Fails 26 Å bound.
        atoms = _cubic(12.633)
        with self.assertRaisesRegex(ValueError, "perpendicular cell height"):
            validate_3d_bulk_cell(
                atoms, self.R_CUTOFF, self.NUM_LAYERS, margin=2.0
            )

    def test_negative_margin_relaxes_bound(self) -> None:
        # Same too-small cell as above passes once the margin drops far
        # enough; this is the unit-test escape hatch documented on
        # MACEUnfoldedHeatFluxCalculator.cell_size_margin.
        atoms = _cubic(12.633)
        validate_3d_bulk_cell(
            atoms, self.R_CUTOFF, self.NUM_LAYERS, margin=-100.0
        )

    def test_rejects_partially_periodic(self) -> None:
        atoms = _cubic(30.0)
        atoms.pbc = [True, True, False]
        with self.assertRaisesRegex(ValueError, "fully periodic 3D"):
            validate_3d_bulk_cell(
                atoms, self.R_CUTOFF, self.NUM_LAYERS, margin=2.0
            )

    def test_rejects_thin_direction_in_otherwise_large_cell(self) -> None:
        # 30×30×10: two heights pass, third fails.
        atoms = Atoms(
            symbols=["Mg"], positions=[[0, 0, 0]],
            cell=[30.0, 30.0, 10.0], pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "10\\.0"):
            validate_3d_bulk_cell(
                atoms, self.R_CUTOFF, self.NUM_LAYERS, margin=2.0
            )

    def test_uses_perpendicular_height_not_lattice_vector_length(self) -> None:
        # Triclinic cell where |a|, |b|, |c| are all > 26 Å but the
        # perpendicular height to one face is much smaller. The validator
        # must use the height, not the vector length.
        a, b = 30.0, 30.0
        # Skew c heavily into the xy plane so its perpendicular component
        # along z is small.
        cell = np.array(
            [[a, 0, 0], [0, b, 0], [25.0, 25.0, 5.0]],
            dtype=float,
        )
        # |c| = sqrt(25² + 25² + 5²) ≈ 35.7 Å (long), but height ≈ 5 Å.
        atoms = Atoms(symbols=["Mg"], positions=[[0, 0, 0]], cell=cell, pbc=True)
        with self.assertRaisesRegex(ValueError, "perpendicular cell height"):
            validate_3d_bulk_cell(
                atoms, self.R_CUTOFF, self.NUM_LAYERS, margin=2.0
            )


if __name__ == "__main__":
    unittest.main()
