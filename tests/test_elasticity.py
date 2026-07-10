"""Tests for elastic-modulus post-processing."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vasp_mace.elasticity import (
    _append_elastic_outcar,
    _hashin_shtrikman_shear_bounds,
    _print_elastic_summary,
)


class HashinShtrikmanShearTests(unittest.TestCase):
    def test_isotropic_tensor_has_coincident_hs_shear_values(self) -> None:
        bulk_modulus = 120.0
        shear_modulus = 50.0
        lam = bulk_modulus - 2.0 * shear_modulus / 3.0
        tensor = np.array(
            [
                [lam + 2 * shear_modulus, lam, lam, 0, 0, 0],
                [lam, lam + 2 * shear_modulus, lam, 0, 0, 0],
                [lam, lam, lam + 2 * shear_modulus, 0, 0, 0],
                [0, 0, 0, shear_modulus, 0, 0],
                [0, 0, 0, 0, shear_modulus, 0],
                [0, 0, 0, 0, 0, shear_modulus],
            ]
        )

        lower, upper, midpoint = _hashin_shtrikman_shear_bounds(tensor)

        self.assertAlmostEqual(lower, shear_modulus, places=4)
        self.assertAlmostEqual(upper, shear_modulus, places=4)
        self.assertAlmostEqual(midpoint, shear_modulus, places=4)

    def test_anisotropic_tensor_returns_ordered_bounds_and_midpoint(self) -> None:
        tensor = np.array(
            [
                [200, 120, 120, 0, 0, 0],
                [120, 200, 120, 0, 0, 0],
                [120, 120, 200, 0, 0, 0],
                [0, 0, 0, 100, 0, 0],
                [0, 0, 0, 0, 100, 0],
                [0, 0, 0, 0, 0, 100],
            ],
            dtype=float,
        )

        lower, upper, midpoint = _hashin_shtrikman_shear_bounds(tensor)

        self.assertLess(lower, upper)
        self.assertAlmostEqual(midpoint, (lower + upper) / 2.0)

    def test_hs_shear_values_are_written_to_stdout_and_outcar(self) -> None:
        tensor = np.eye(6) * 100.0
        output = StringIO()
        with redirect_stdout(output):
            _print_elastic_summary(
                tensor, 80.0, 60.0, 70.0, 50.0, 75.0, 55.0, 52.0, 54.0, 53.0, 140.0, 0.2
            )

        self.assertIn("Hashin-Shtrikman G: lower =   52.00 GPa", output.getvalue())

        with TemporaryDirectory() as directory:
            outcar_path = Path(directory) / "OUTCAR"
            _append_elastic_outcar(
                str(outcar_path),
                tensor,
                80.0,
                60.0,
                70.0,
                50.0,
                75.0,
                55.0,
                52.0,
                54.0,
                53.0,
                140.0,
                0.2,
            )

            self.assertIn(
                "Hashin-Shtrikman shear modulus (GPa): lower = 52.000  upper = 54.000  "
                "midpoint = 53.000",
                outcar_path.read_text(),
            )


if __name__ == "__main__":
    unittest.main()
