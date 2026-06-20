"""Regression tests for INCAR value parsing."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from vasp_mace.incar import parse_incar


class IncarParsingTests(unittest.TestCase):
    def _parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "INCAR"
            path.write_text(textwrap.dedent(text).strip() + "\n")
            return parse_incar(str(path))

    def test_accepts_fortran_d_exponents(self) -> None:
        cfg = self._parse_text(
            """
            EDIFFG = -1D-2
            POTIM = 2d0
            TEBEG = 3D2
            TEEND = 4d2
            LANGEVIN_GAMMA = 1D1 2D1
            NSW = 1D1
            RANDOM_SEED = 123
            """
        )

        self.assertEqual(cfg.EDIFFG, -0.01)
        self.assertEqual(cfg.POTIM, 2.0)
        self.assertEqual(cfg.TEBEG, 300.0)
        self.assertEqual(cfg.TEEND, 400.0)
        self.assertEqual(cfg.NSW, 10)
        self.assertEqual(cfg.RANDOM_SEED, 123)
        np.testing.assert_allclose(cfg.LANGEVIN_GAMMA, [10.0, 20.0])

    def test_rejects_malformed_numeric_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "POTIM must be a number"):
            self._parse_text("POTIM = not-a-number")

    def test_rejects_malformed_integer_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "NSW must be an integer"):
            self._parse_text("NSW = 1.5")

    def test_rejects_malformed_logical_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "ML_LHEAT must be a logical"):
            self._parse_text("ML_LHEAT = maybe")

    def test_rejects_malformed_float_list_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "LANGEVIN_GAMMA must be a list"):
            self._parse_text("LANGEVIN_GAMMA = 10 bad")

    def test_rejects_negative_random_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "RANDOM_SEED must be non-negative"):
            self._parse_text("RANDOM_SEED = -1")


if __name__ == "__main__":
    unittest.main()
