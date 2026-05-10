"""Opt-in smoke test for the MACE unfolded-cell heat-flux backend.

Skipped unless all of the following hold:

* ``RUN_VASP_MACE_EXAMPLES=1`` (matches the convention used by
  :class:`tests.test_examples.ExampleSmokeTests`).
* ``MACE_MODEL_PATH`` points at a readable MACE checkpoint.
* The optional ``mace_unfolded`` package is importable
  (``pip install vasp-mace[heat]``).

The test asserts only that the backend returns a finite, length-3 heat-flux
vector for a fixed 8-atom MgO configuration with deterministic velocities. The
numerical regression check lives in
:mod:`tests.test_mace_unfolded_regression`.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    os.environ.get("RUN_VASP_MACE_EXAMPLES") == "1",
    "Set RUN_VASP_MACE_EXAMPLES=1 to run MACE-backed smoke tests",
)
class MACEUnfoldedHeatFluxSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        model = os.environ.get("MACE_MODEL_PATH")
        if not model:
            self.skipTest("MACE_MODEL_PATH is required for heat-flux smoke tests")
        if not Path(model).is_file():
            self.skipTest(f"MACE model checkpoint not found: {model}")
        if importlib.util.find_spec("mace_unfolded") is None:
            self.skipTest(
                "mace_unfolded not installed; install with "
                "`pip install vasp-mace[heat]`"
            )
        self.model_path = model

    def test_compute_returns_finite_three_vector(self) -> None:
        from ase.build import bulk

        from vasp_mace.heat import make_heat_flux_calculator

        atoms = bulk("MgO", "rocksalt", a=4.211, cubic=True)
        rng = np.random.default_rng(42)
        velocities = rng.normal(0.0, 0.005, size=(len(atoms), 3))

        calc = make_heat_flux_calculator(
            self.model_path,
            settings={"device": "auto", "dtype": "float64"},
        )
        qxyz = calc.compute(atoms, velocities)

        self.assertEqual(qxyz.shape, (3,))
        self.assertTrue(np.all(np.isfinite(qxyz)), f"flux not finite: {qxyz}")
        self.assertEqual(qxyz.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
