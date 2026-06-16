"""Tests for MD temperature-ramp helpers."""

from __future__ import annotations

import unittest

import numpy as np
from ase import Atoms
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.units import fs as ASE_FS, kB

from vasp_mace.md import _set_dynamics_temperature, _temperature_target


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

    def test_set_temperature_uses_public_setter(self) -> None:
        dyn = _SetTemperatureDynamics()
        self.assertTrue(_set_dynamics_temperature(dyn, 250.0))
        self.assertEqual(dyn.temperature_K, 250.0)

    def test_set_temperature_updates_temp_k_dynamics(self) -> None:
        dyn = _TempKOnlyDynamics()
        self.assertTrue(_set_dynamics_temperature(dyn, 250.0))
        self.assertEqual(dyn.temp_K, 250.0)

    def test_set_temperature_updates_nose_hoover_chain_target(self) -> None:
        atoms = Atoms(
            "Ar",
            positions=[[0.0, 0.0, 0.0]],
            masses=[39.948],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        atoms.set_velocities(np.zeros((1, 3)))
        tdamp = 10.0 * ASE_FS
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=1.0 * ASE_FS,
            temperature_K=100.0,
            tdamp=tdamp,
        )

        self.assertTrue(_set_dynamics_temperature(dyn, 300.0))

        thermostat = dyn._thermostat
        expected_kT = kB * 300.0
        self.assertAlmostEqual(thermostat._kT, expected_kT)
        self.assertAlmostEqual(thermostat._Q[0], 3 * expected_kT * tdamp**2)
        np.testing.assert_allclose(thermostat._Q[1:], expected_kT * tdamp**2)

    def test_set_temperature_returns_false_for_unsupported_dynamics(self) -> None:
        self.assertFalse(_set_dynamics_temperature(object(), 250.0))


if __name__ == "__main__":
    unittest.main()
