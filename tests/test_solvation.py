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
    _EPS_IN,
    _KE,
    _OFFSET,
    _PROBE,
    _atom_radii,
    born_radii,
    born_polar_energy,
    SolvationCalculator,
    solvent_accessible_surface_area,
)


def _dftd4_available() -> bool:
    try:
        import dftd4  # noqa: F401
    except ImportError:
        return False
    return True


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
            return SolvationCalculator(tau=tau)

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
        at.calc = SolvationCalculator(tau=0.525)
        with self.assertRaisesRegex(ValueError, "solvent-accessible surface"):
            at.get_potential_energy()

    def test_slab_with_vacuum_allowed(self) -> None:
        slab = fcc111("Cu", size=(1, 1, 3), vacuum=8.0)
        slab.calc = SolvationCalculator(tau=0.525)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = slab.get_potential_energy()
        self.assertGreater(e, 0.0)


class GeneralizedBornTests(unittest.TestCase):
    """Polar GB term — unit tests with explicit charges (no dftd4 needed)."""

    def test_single_atom_born_radius_is_offset_radius(self) -> None:
        # With no descreening neighbours, the OBC tanh sum is zero and the Born
        # radius reduces to (r_vdw - offset).
        at = Atoms("O", positions=[(0, 0, 0)])
        r = born_radii(at)
        expected = _atom_radii(at.get_atomic_numbers())[0] - _OFFSET
        self.assertAlmostEqual(r[0], expected, places=8)

    def test_single_ion_matches_born_formula(self) -> None:
        at = Atoms("O", positions=[(0, 0, 0)])
        q = np.array([1.0])
        eb_k = 78.4
        e = born_polar_energy(at, q, eb_k)
        radius = _atom_radii(at.get_atomic_numbers())[0] - _OFFSET
        expected = -0.5 * _KE * (1.0 / _EPS_IN - 1.0 / eb_k) * 1.0 / radius
        self.assertAlmostEqual(e, expected, places=8)

    def test_polar_energy_is_favorable(self) -> None:
        # Opposite charges in solvent -> negative (favorable) polar energy.
        at = Atoms("Na Cl".split(), positions=[(0, 0, 0), (2.8, 0, 0)])
        e = born_polar_energy(at, np.array([1.0, -1.0]), 78.4)
        self.assertLess(e, 0.0)

    def test_cross_term_reduces_dipole_solvation(self) -> None:
        # The GB cross term (opposite charges nearby) must make a neutral
        # dipole less solvated than the sum of the two isolated Born self
        # energies. This cancellation is what keeps neutral-system solvation
        # physically small (and is computed with minimum-image distances, not a
        # periodic lattice sum).
        at = Atoms("Na Cl".split(), positions=[(0, 0, 0), (2.8, 0, 0)])
        q = np.array([1.0, -1.0])
        e_full = born_polar_energy(at, q, 78.4)
        r = born_radii(at)
        pre = -0.5 * _KE * (1.0 / _EPS_IN - 1.0 / 78.4)
        e_self_only = pre * float((q * q / r).sum())
        self.assertGreater(e_full, e_self_only)  # cross cancels -> less negative
        self.assertLess(e_full, 0.0)

    def test_higher_dielectric_more_negative(self) -> None:
        at = Atoms("O", positions=[(0, 0, 0)])
        q = np.array([1.0])
        e_low = born_polar_energy(at, q, 2.0)
        e_high = born_polar_energy(at, q, 78.4)
        self.assertLess(e_high, e_low)


@unittest.skipUnless(_dftd4_available(), "dftd4 not installed")
class PolarSolvationCalculatorTests(unittest.TestCase):
    def test_polar_term_lowers_energy(self) -> None:
        # NaCl dimer: EEQ gives Na+/Cl-, so the polar term must lower the
        # energy relative to the nonpolar-only calculator.
        at = Atoms("Na Cl".split(), positions=[(0, 0, 0), (2.8, 0, 0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at.calc = SolvationCalculator(tau=0.525, eb_k=None)
            e_nonpolar = at.get_potential_energy()
            at.calc = SolvationCalculator(tau=0.525, eb_k=78.4)
            e_full = at.get_potential_energy()
        self.assertLess(e_full, e_nonpolar)


class SolvationIncarTests(unittest.TestCase):
    def test_defaults_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = parse_incar(_write_incar(Path(td), "NSW = 0\n"))
            self.assertFalse(cfg.LSOL)
            self.assertEqual(cfg.TAU, 0.525)
            self.assertEqual(cfg.EB_K, 78.4)

    def test_parses_lsol_tau_eb_k(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = parse_incar(
                _write_incar(Path(td), "LSOL = .TRUE.\nTAU = 0.8\nEB_K = 80\n")
            )
            self.assertTrue(cfg.LSOL)
            self.assertAlmostEqual(cfg.TAU, 0.8)
            self.assertAlmostEqual(cfg.EB_K, 80.0)

    def test_rejects_negative_tau(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "TAU"):
                parse_incar(_write_incar(Path(td), "LSOL = .TRUE.\nTAU = -1.0\n"))

    def test_rejects_eb_k_below_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "EB_K"):
                parse_incar(_write_incar(Path(td), "LSOL = .TRUE.\nEB_K = 0.5\n"))


if __name__ == "__main__":
    unittest.main()
