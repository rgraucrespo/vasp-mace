"""Opt-in smoke test for the MACE unfolded-cell heat-flux backend.

Skipped unless all of the following hold:

* ``RUN_VASP_MACE_EXAMPLES=1`` (matches the convention used by
  :class:`tests.test_examples.ExampleSmokeTests`).
* ``MACE_MODEL_PATH`` points at a readable MACE checkpoint.
* The optional ``mace_unfolded`` package is importable
  (``pip install -r requirements-heat.txt`` from a source checkout).

The test asserts only that the backend returns a finite, length-3 heat-flux
vector for the shared PbTe fixture (see :mod:`tests._heat_flux_fixtures`).
The numerical regression check lives in
:mod:`tests.test_mace_unfolded_regression`.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

from tests._heat_flux_fixtures import build_pbte_fixture


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
                "`pip install -r requirements-heat.txt` from a source checkout"
            )
        self.model_path = model

    def test_compute_returns_finite_three_vector(self) -> None:
        from vasp_mace.heat import make_heat_flux_calculator
        from vasp_mace.mace_loader import load_calc

        atoms, velocities = build_pbte_fixture()

        # Stay on the calculator's float64 default end-to-end: mace-unfolded
        # has an internal dtype-mismatch bug under float32 (positions stay
        # float64 even after `set_default_dtype("float32")`), and the small
        # MACE-MP-0 + 64-atom PbTe configuration fits comfortably in 16 GB
        # of GPU memory at float64 anyway. cell_size_margin=-100 disables
        # the production cell-size check (the 2×2×2 PbTe fixture clears
        # mace-unfolded's L > R requirement but not the stricter
        # L > 2R + 2 Å vasp-mace bound). Production callers never touch
        # this; only unit tests do.
        main_calc, device, dtype = load_calc(
            self.model_path,
            device="auto",
            dtype="float64",
        )
        calc = make_heat_flux_calculator(
            self.model_path,
            settings={
                "device": device,
                "dtype": dtype,
                "cell_size_margin": -100.0,
                "torch_model": main_calc.models[0],
            },
        )
        qxyz = calc.compute(atoms, velocities)

        self.assertEqual(qxyz.shape, (3,))
        self.assertTrue(np.all(np.isfinite(qxyz)), f"flux not finite: {qxyz}")
        self.assertEqual(qxyz.dtype, np.float64)
        self.assertTrue(calc.uses_shared_model)
        calc.close()


if __name__ == "__main__":
    unittest.main()
