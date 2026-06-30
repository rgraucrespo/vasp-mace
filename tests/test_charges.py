"""Tests for the shared EEQ partial-charge engine (vasp_mace.charges)."""

from __future__ import annotations

import unittest


def _dftd4_available() -> bool:
    try:
        import dftd4  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_dftd4_available(), "dftd4 not installed")
class EEQChargeTests(unittest.TestCase):
    def test_water_is_neutral_and_polar(self) -> None:
        from ase import Atoms
        from vasp_mace.charges import eeq_charges

        # O at origin, two H above: O electronegative -> negative, H positive.
        atoms = Atoms(
            "OH2",
            positions=[(0, 0, 0), (0, 0.757, 0.586), (0, -0.757, 0.586)],
        )
        q = eeq_charges(atoms)
        self.assertEqual(q.shape, (3,))
        self.assertAlmostEqual(float(q.sum()), 0.0, places=6)
        self.assertLess(q[0], 0.0)  # O
        self.assertGreater(q[1], 0.0)  # H
        self.assertGreater(q[2], 0.0)  # H

    def test_periodic_neutral(self) -> None:
        import numpy as np
        from ase import Atoms
        from vasp_mace.charges import eeq_charges

        # Rock-salt-like NaCl: Na electropositive -> positive, Cl negative.
        atoms = Atoms(
            "NaCl",
            positions=[(0, 0, 0), (2.8, 2.8, 2.8)],
            cell=np.eye(3) * 5.6,
            pbc=True,
        )
        q = eeq_charges(atoms)
        self.assertEqual(q.shape, (2,))
        self.assertAlmostEqual(float(q.sum()), 0.0, places=6)
        self.assertGreater(q[0], 0.0)  # Na
        self.assertLess(q[1], 0.0)  # Cl


if __name__ == "__main__":
    unittest.main()
