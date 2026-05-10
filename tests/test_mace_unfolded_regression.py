"""Opt-in regression test for the MACE unfolded-cell heat-flux backend.

Compares the heat flux computed by ``MACEUnfoldedHeatFluxCalculator`` for the
shared PbTe fixture (see :mod:`tests._heat_flux_fixtures`) to a saved
reference. The reference is *not* checked into the repository because it is
model-checkpoint-specific. Generate it once locally with::

    RUN_VASP_MACE_EXAMPLES=1 \\
    MACE_MODEL_PATH=/path/to/your.model \\
    python -m tests.test_mace_unfolded_regression --create-reference

That writes ``tests/data/mace_unfolded_reference.npz`` (the path the test
loads). Commit it alongside the model identifier you used so subsequent runs
catch any unintended numerical drift in the backend.

Gated identically to :mod:`tests.test_mace_heat_flux_smoke`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from tests._heat_flux_fixtures import build_pbte_fixture


REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "mace_unfolded_reference.npz"


def _compute_reference_flux(model_path: str) -> np.ndarray:
    from vasp_mace.heat import make_heat_flux_calculator

    atoms, velocities = build_pbte_fixture()
    # dtype="auto" → float32 on CUDA/MPS, float64 on CPU. Required to fit on
    # the smaller GPUs (≤16 GB) we routinely use for heat-flux work. The
    # saved reference is therefore both model-checkpoint *and*
    # hardware-specific; regenerate it on the same device + dtype where you
    # plan to run the regression.
    # cell_size_margin=-100 disables the production cell-size check; see
    # the comment in tests/test_mace_heat_flux_smoke.py for the rationale.
    calc = make_heat_flux_calculator(
        model_path,
        settings={
            "device": "auto",
            "dtype": "auto",
            "cell_size_margin": -100.0,
        },
    )
    return calc.compute(atoms, velocities)


@unittest.skipUnless(
    os.environ.get("RUN_VASP_MACE_EXAMPLES") == "1",
    "Set RUN_VASP_MACE_EXAMPLES=1 to run MACE-backed regression tests",
)
class MACEUnfoldedRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        model = os.environ.get("MACE_MODEL_PATH")
        if not model:
            self.skipTest("MACE_MODEL_PATH is required for the regression test")
        if not Path(model).is_file():
            self.skipTest(f"MACE model checkpoint not found: {model}")
        if importlib.util.find_spec("mace_unfolded") is None:
            self.skipTest(
                "mace_unfolded not installed; install with "
                "`pip install vasp-mace[heat]`"
            )
        if not REFERENCE_PATH.is_file():
            self.skipTest(
                f"Reference flux not found at {REFERENCE_PATH}. Generate it "
                "once with `python -m tests.test_mace_unfolded_regression "
                "--create-reference` and commit the resulting file."
            )
        self.model_path = model

    def test_flux_matches_saved_reference(self) -> None:
        ref = np.load(REFERENCE_PATH)
        expected = ref["qxyz"]

        flux = _compute_reference_flux(self.model_path)

        self.assertEqual(flux.shape, expected.shape)
        # rtol=1e-5 is loose enough to absorb the float32 noise of the
        # internal autograd path on CUDA (heat flux is a difference of two
        # large cancelling terms, the canonical hard-on-float32 case) while
        # still tight enough to catch any systematic drift in the backend.
        # On the CPU/float64 path this passes with margin to spare.
        np.testing.assert_allclose(flux, expected, rtol=1e-5, atol=1e-8)


def _create_reference(model_path: str) -> None:
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    flux = _compute_reference_flux(model_path)
    np.savez(REFERENCE_PATH, qxyz=flux)
    print(f"wrote reference flux {flux.tolist()} to {REFERENCE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create-reference",
        action="store_true",
        help="recompute and overwrite the saved reference flux",
    )
    args, remaining = parser.parse_known_args()

    if args.create_reference:
        model = os.environ.get("MACE_MODEL_PATH")
        if not model or not Path(model).is_file():
            sys.exit(
                "MACE_MODEL_PATH must point at a readable MACE checkpoint to "
                "generate the reference."
            )
        if importlib.util.find_spec("mace_unfolded") is None:
            sys.exit(
                "mace_unfolded is not installed; "
                "`pip install vasp-mace[heat]` first."
            )
        _create_reference(model)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
