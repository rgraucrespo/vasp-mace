"""Tests for MD temperature-ramp helpers."""

from __future__ import annotations

import unittest

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.units import fs as ASE_FS, kB

from vasp_mace.md import (
    _instantaneous_temperature,
    _set_dynamics_temperature,
    _temperature_target,
)


class _SetTemperatureDynamics:
    def __init__(self) -> None:
        self.temperature_K = None

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = temperature_K


class _TempKOnlyDynamics:
    def __init__(self) -> None:
        self.temp_K = 100.0


class MDTemperatureRampTests(unittest.TestCase):
    def test_temperature_target_linearly_interpolates(self) -> None:
        self.assertEqual(_temperature_target(100.0, 300.0, 1, 5), 100.0)
        self.assertEqual(_temperature_target(100.0, 300.0, 3, 5), 200.0)
        self.assertEqual(_temperature_target(100.0, 300.0, 5, 5), 300.0)

    def test_temperature_target_single_step_uses_start(self) -> None:
        self.assertEqual(_temperature_target(100.0, 300.0, 1, 1), 100.0)

    def test_instantaneous_temperature_uses_ase_dof_count(self) -> None:
        atoms = Atoms(
            "Ar2",
            positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            masses=[39.948, 39.948],
        )
        atoms.set_velocities([[0.0, 0.0, 0.0], [0.02, 0.03, 0.04]])
        atoms.set_constraint(FixAtoms(indices=[0]))
        kinetic = atoms.get_kinetic_energy()

        self.assertEqual(atoms.get_number_of_degrees_of_freedom(), 3)
        self.assertAlmostEqual(
            _instantaneous_temperature(atoms, kinetic),
            2.0 * kinetic / (3.0 * kB),
        )

    def test_instantaneous_temperature_returns_zero_for_no_dof(self) -> None:
        atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], masses=[39.948])
        atoms.set_constraint(FixAtoms(indices=[0]))

        self.assertEqual(_instantaneous_temperature(atoms, 1.0), 0.0)

    def test_set_temperature_uses_public_setter(self) -> None:
        dyn = _SetTemperatureDynamics()
        self.assertTrue(_set_dynamics_temperature(dyn, 250.0))
        self.assertEqual(dyn.temperature_K, 250.0)

    def test_set_temperature_updates_temp_k_dynamics(self) -> None:
        dyn = _TempKOnlyDynamics()
        self.assertTrue(_set_dynamics_temperature(dyn, 250.0))
        self.assertEqual(dyn.temp_K, 250.0)

    @staticmethod
    def _make_nhc(temperature_K: float, tdamp: float) -> NoseHooverChainNVT:
        # Three atoms so the chain mass depends on a non-trivial dof count;
        # a single atom would hide a wrong degrees-of-freedom factor.
        atoms = Atoms(
            "Ar3",
            positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            masses=[39.948, 39.948, 39.948],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        atoms.set_velocities(np.zeros((3, 3)))
        return NoseHooverChainNVT(
            atoms,
            timestep=1.0 * ASE_FS,
            temperature_K=temperature_K,
            tdamp=tdamp,
        )

    def test_set_temperature_updates_nose_hoover_chain_target(self) -> None:
        tdamp = 10.0 * ASE_FS
        # Build the thermostat at the start temperature, ramp it, and compare
        # against a thermostat ASE constructs directly at the target. This
        # catches drift in ASE's mass-matrix formula without reproducing it.
        dyn = self._make_nhc(100.0, tdamp)
        reference = self._make_nhc(300.0, tdamp)

        self.assertTrue(_set_dynamics_temperature(dyn, 300.0))

        thermostat = dyn._thermostat
        self.assertAlmostEqual(thermostat._kT, kB * 300.0)
        self.assertAlmostEqual(thermostat._kT, reference._thermostat._kT)
        np.testing.assert_allclose(thermostat._Q, reference._thermostat._Q)

    def test_set_temperature_returns_false_for_unsupported_dynamics(self) -> None:
        self.assertFalse(_set_dynamics_temperature(object(), 250.0))


if __name__ == "__main__":
    unittest.main()
