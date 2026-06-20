"""Tests for POSCAR/CONTCAR I/O helpers."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from vasp_mace.io_poscar import read_poscar


SELECTIVE_POSCAR = textwrap.dedent(
    """
    selective dynamics test
    1.0
    5.0 0.0 0.0
    0.0 5.0 0.0
    0.0 0.0 5.0
    H
    2
    Selective dynamics
    Direct
    0.0 0.0 0.0 F F F
    0.5 0.5 0.5 T T T
    """
).strip()


class PoscarIOTests(unittest.TestCase):
    def _write_poscar(self, directory: str) -> str:
        path = Path(directory) / "POSCAR"
        path.write_text(SELECTIVE_POSCAR + "\n")
        return str(path)

    def test_read_poscar_applies_selective_dynamics_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            atoms = read_poscar(self._write_poscar(td))

        self.assertTrue(atoms.constraints)
        self.assertLess(atoms.get_number_of_degrees_of_freedom(), 3 * len(atoms))

    def test_read_poscar_can_ignore_selective_dynamics_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            atoms = read_poscar(
                self._write_poscar(td),
                apply_selective_dynamics=False,
            )

        self.assertEqual(atoms.constraints, [])
        self.assertEqual(atoms.get_number_of_degrees_of_freedom(), 3 * len(atoms))


if __name__ == "__main__":
    unittest.main()
