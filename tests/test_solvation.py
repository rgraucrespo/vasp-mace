"""Tests for the nonpolar (SASA) implicit-solvation term (Stage 3a)."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111

from vasp_mace.incar import parse_incar
from vasp_mace.solvation import (
    _PROBE,
    _atom_radii,
    SASASolvationCalculator,
    solvent_accessible_surface_area,
)


def _write_incar(tmp: Path, body: str) -> str:
    path = tmp / "INCAR"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return str(path)


class SASAGeometryTests(unittest.TestCase):
    def test_isolated_atom_is_full_sphere(self) -> None:
        # A lone atom has no occluding neighbours: SASA == 4*pi*(r_vdw+probe)^2.
        at = Atoms("O", positions=[(0, 0, 0)])
        r = _atom_radii(at.get_atomic_numbers())[0] + _PROBE
        expected = 4.0 * np.pi * r**2
        sasa = solvent_accessible_surface_area(at)
        self.assertAlmostEqual(sasa, expected, places=6)

    def test_overlap_reduces_area(self) -> None:
        lone = solvent_accessible_surface_area(Atoms("O", positions=[(0, 0, 0)]))
        dimer = solvent_accessible_surface_area(
            Atoms("O2", positions=[(0, 0, 0), (1.0, 0, 0)])
        )
        # Two heavily overlapping atoms expose less than twice a lone atom.
        self.assertLess(dimer, 2.0 * lone)
        self.assertGreater(dimer, lone)


class SASACalculatorTests(unittest.TestCase):
    def _calc(self, tau=0.525):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SASASolvationCalculator(tau=tau)

    def test_energy_matches_tau_times_area(self) -> None:
        at = Atoms("O", positions=[(0, 0, 0)])
        at.calc = self._calc(tau=0.525)
        sasa = solvent_accessible_surface_area(at)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = at.get_potential_energy()
        self.assertAlmostEqual(e, 0.525e-3 * sasa, places=8)

    def test_energy_scales_with_tau(self) -> None:
        at = Atoms("O2", positions=[(0, 0, 0), (2.5, 0, 0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at.calc = self._calc(tau=1.0)
            e1 = at.get_potential_energy()
            at.calc = self._calc(tau=2.0)
            e2 = at.get_potential_energy()
        self.assertAlmostEqual(e2, 2.0 * e1, places=8)

    def test_isolated_atom_has_zero_force(self) -> None:
        at = Atoms("O", positions=[(0, 0, 0)])
        at.calc = self._calc()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = at.get_forces()
        self.assertLess(np.abs(f).max(), 1.0e-6)

    def test_dimer_forces_antisymmetric_and_nonzero(self) -> None:
        at = Atoms("O2", positions=[(0, 0, 0), (2.5, 0, 0)])
        at.calc = self._calc()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = at.get_forces()
        self.assertGreater(np.abs(f).max(), 1.0e-6)
        np.testing.assert_allclose(f[0], -f[1], atol=1.0e-8)


class SASAScopeTests(unittest.TestCase):
    def test_dense_bulk_rejected(self) -> None:
        at = bulk("Cu", "fcc", a=3.6)  # filled primitive cell, no vacuum
        at.calc = SASASolvationCalculator(tau=0.525)
        with self.assertRaisesRegex(ValueError, "solvent-accessible surface"):
            at.get_potential_energy()

    def test_slab_with_vacuum_allowed(self) -> None:
        slab = fcc111("Cu", size=(1, 1, 3), vacuum=8.0)
        slab.calc = SASASolvationCalculator(tau=0.525)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = slab.get_potential_energy()
        self.assertGreater(e, 0.0)


class SolvationIncarTests(unittest.TestCase):
    def test_defaults_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = parse_incar(_write_incar(Path(td), "NSW = 0\n"))
            self.assertFalse(cfg.LSOL)
            self.assertEqual(cfg.TAU, 0.525)

    def test_parses_lsol_and_tau(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = parse_incar(_write_incar(Path(td), "LSOL = .TRUE.\nTAU = 0.8\n"))
            self.assertTrue(cfg.LSOL)
            self.assertAlmostEqual(cfg.TAU, 0.8)

    def test_rejects_negative_tau(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "TAU"):
                parse_incar(_write_incar(Path(td), "LSOL = .TRUE.\nTAU = -1.0\n"))


if __name__ == "__main__":
    unittest.main()
