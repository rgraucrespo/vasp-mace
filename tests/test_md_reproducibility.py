"""Tests for deterministic MD setup controls."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from vasp_mace.incar import parse_incar
from vasp_mace.io_outcar import write_md_oszicar
from vasp_mace.md import run_md
from vasp_mace.types_ import MDRecord


class ZeroForceCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        n_atoms = len(self.atoms)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((n_atoms, 3)),
            "stress": np.zeros(6),
        }


class MDReproducibilityTests(unittest.TestCase):
    def _run_one_step(self, seed: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "INCAR").write_text(
                textwrap.dedent(
                    f"""
                    IBRION = 0
                    MDALGO = 1
                    ANDERSEN_PROB = 0.0
                    NSW = 1
                    TEBEG = 300
                    POTIM = 1.0
                    NBLOCK = 1
                    RANDOM_SEED = {seed}
                    """
                ).strip()
                + "\n"
            )
            cfg = parse_incar(str(path / "INCAR"))
            atoms = Atoms(
                "Ar2",
                positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                masses=[39.948, 39.948],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
            cwd = os.getcwd()
            try:
                os.chdir(path)
                run_md(atoms, ZeroForceCalculator(), cfg)
            finally:
                os.chdir(cwd)
            return atoms.get_velocities().copy()

    def _run_one_step_with_velocities(self, velocities: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "INCAR").write_text(
                textwrap.dedent(
                    """
                    IBRION = 0
                    MDALGO = 1
                    ANDERSEN_PROB = 0.0
                    NSW = 1
                    TEBEG = 300
                    POTIM = 1.0
                    NBLOCK = 1
                    RANDOM_SEED = 123
                    """
                ).strip()
                + "\n"
            )
            cfg = parse_incar(str(path / "INCAR"))
            atoms = Atoms(
                "Ar2",
                positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                masses=[39.948, 39.948],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
            atoms.set_velocities(velocities)
            cwd = os.getcwd()
            try:
                os.chdir(path)
                run_md(atoms, ZeroForceCalculator(), cfg)
            finally:
                os.chdir(cwd)
            return atoms.get_velocities().copy()

    def test_random_seed_reproduces_initial_md_velocities(self) -> None:
        np.testing.assert_allclose(self._run_one_step(123), self._run_one_step(123))

    def test_different_random_seeds_change_initial_md_velocities(self) -> None:
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(self._run_one_step(123), self._run_one_step(456))

    def test_run_md_preserves_existing_nonzero_velocities(self) -> None:
        velocities = np.array([[0.01, 0.02, 0.03], [-0.02, 0.0, 0.01]])

        np.testing.assert_allclose(
            self._run_one_step_with_velocities(velocities),
            velocities,
        )

    def test_write_md_oszicar_records_temperature_and_energies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "OSZICAR"
            write_md_oszicar(
                str(path),
                [
                    MDRecord(
                        n=1,
                        energy_pot=-2.0,
                        energy_kin=0.5,
                        temperature=300.0,
                    )
                ],
            )

            text = path.read_text()

        self.assertIn("T=   300.00", text)
        self.assertIn("E=", text)
        self.assertIn("F=", text)
        self.assertIn("EK=", text)


if __name__ == "__main__":
    unittest.main()
