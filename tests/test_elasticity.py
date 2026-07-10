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
    _hashin_shtrikman_moduli,
    _modulus_row,
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

        lower, upper, midpoint = _hashin_shtrikman_moduli(tensor)

        self.assertAlmostEqual(lower[2], shear_modulus, places=4)
        self.assertAlmostEqual(upper[2], shear_modulus, places=4)
        self.assertAlmostEqual(midpoint[2], shear_modulus, places=4)
        expected_nu = (3 * bulk_modulus - 2 * shear_modulus) / (
            6 * bulk_modulus + 2 * shear_modulus
        )
        self.assertAlmostEqual(lower[4], expected_nu, places=4)
        self.assertAlmostEqual(upper[4], expected_nu, places=4)
        self.assertAlmostEqual(midpoint[4], expected_nu, places=4)

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

        lower, upper, midpoint = _hashin_shtrikman_moduli(tensor)

        self.assertLess(lower[2], upper[2])
        self.assertAlmostEqual(midpoint[2], (lower[2] + upper[2]) / 2.0)
        self.assertGreater(lower[4], -1.0)
        self.assertLess(upper[4], 0.5)
        self.assertGreater(midpoint[4], -1.0)
        self.assertLess(midpoint[4], 0.5)

    def test_hs_shear_values_are_written_to_stdout_and_outcar(self) -> None:
        tensor = np.eye(6) * 100.0
        modulus_rows = [
            _modulus_row("Voigt", 80.0, 60.0),
            _modulus_row("Reuss", 70.0, 50.0),
            _modulus_row("Hill", 75.0, 55.0),
            _modulus_row("Hashin-Shtrikman lower", 74.0, 52.0),
            _modulus_row("Hashin-Shtrikman upper", 76.0, 54.0),
            _modulus_row("Hashin-Shtrikman midpoint", 75.0, 53.0),
        ]
        output = StringIO()
        with redirect_stdout(output):
            _print_elastic_summary(tensor, modulus_rows)

        self.assertIn(
            "Approximation                    K (GPa)",
            output.getvalue(),
        )
        self.assertIn("Hashin-Shtrikman lower", output.getvalue())

        with TemporaryDirectory() as directory:
            outcar_path = Path(directory) / "OUTCAR"
            _append_elastic_outcar(str(outcar_path), tensor, modulus_rows)

            self.assertIn(
                "POLYCRYSTALLINE ELASTIC MODULI (GPa)",
                outcar_path.read_text(),
            )
            self.assertIn(
                "Hashin-Shtrikman midpoint",
                outcar_path.read_text(),
            )


if __name__ == "__main__":
    unittest.main()
