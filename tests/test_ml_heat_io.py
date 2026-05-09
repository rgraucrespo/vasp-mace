"""Stage 1 tests for the VASP-compatible ML_HEAT writer/reader.

These tests cover:

* Round-tripping a synthetic ``(steps, qxyz)`` array through the streaming
  writer and the parser.
* Parsing the verbatim VASP example block from the implementation
  instructions, including tolerance for Fortran-style ``D`` exponents.
* Parsing the ML_LHEAT and ML_HEAT_INTERVAL keywords from an INCAR snippet.

No MACE model or external dependency is required — this is pure file I/O.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from vasp_mace.heat import MLHeatWriter, read_ml_heat, write_ml_heat
from vasp_mace.incar import parse_incar


# Verbatim from not_for_release/ml_heat_implementation_instructions.md, lines 60-63.
VASP_EXAMPLE_LINES = (
    "NSTEP=         1 QXYZ=  0.36329995E-03 -0.18158424E-03 -0.89885493E-03\n"
    "NSTEP=         2 QXYZ=  0.12017813E+00 -0.24353637E+00 -0.24858697E-02\n"
)


class MLHeatRoundTripTests(unittest.TestCase):
    def test_streaming_writer_roundtrip(self) -> None:
        rng = np.random.default_rng(seed=0)
        steps = np.arange(1, 51)
        qxyz = rng.standard_normal((50, 3)) * 1.0e-3

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            with MLHeatWriter(path) as writer:
                for s, q in zip(steps, qxyz):
                    writer.write(int(s), q)

            steps_back, qxyz_back = read_ml_heat(path)

        self.assertEqual(steps_back.dtype, np.int64)
        self.assertEqual(qxyz_back.shape, (50, 3))
        np.testing.assert_array_equal(steps_back, steps)
        # 8-significant-figure scientific notation gives ≲1e-12 relative loss
        # for the magnitudes we picked.
        np.testing.assert_allclose(qxyz_back, qxyz, atol=1e-10, rtol=1e-7)

    def test_bulk_write_roundtrip(self) -> None:
        steps = [10, 20, 30]
        qxyz = np.array(
            [
                [1.0e-3, -2.0e-3, 3.0e-3],
                [0.0, 0.0, 0.0],
                [-4.5e-2, 1.25e-1, -7.5e-1],
            ],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            write_ml_heat(path, steps, qxyz)
            steps_back, qxyz_back = read_ml_heat(path)

        np.testing.assert_array_equal(steps_back, np.asarray(steps, dtype=np.int64))
        np.testing.assert_allclose(qxyz_back, qxyz, atol=1e-12)

    def test_writer_rejects_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            with MLHeatWriter(path) as writer:
                with self.assertRaises(ValueError):
                    writer.write(1, np.array([1.0, 2.0]))  # only 2 components

    def test_bulk_writer_rejects_mismatched_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            with self.assertRaises(ValueError):
                write_ml_heat(path, [1, 2], np.zeros((3, 3)))


class MLHeatVASPCompatibilityTests(unittest.TestCase):
    def test_parses_verbatim_vasp_example(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            Path(path).write_text(VASP_EXAMPLE_LINES)

            steps, qxyz = read_ml_heat(path)

        np.testing.assert_array_equal(steps, np.array([1, 2], dtype=np.int64))
        expected = np.array(
            [
                [0.36329995e-03, -0.18158424e-03, -0.89885493e-03],
                [0.12017813e00, -0.24353637e00, -0.24858697e-02],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(qxyz, expected, atol=1e-12)

    def test_parses_fortran_d_exponents(self) -> None:
        # Real VASP output occasionally uses 'D' as the exponent marker.
        text = (
            "NSTEP=         1 QXYZ=  0.36329995D-03 -0.18158424D-03 -0.89885493D-03\n"
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            Path(path).write_text(text)
            steps, qxyz = read_ml_heat(path)

        self.assertEqual(steps.tolist(), [1])
        np.testing.assert_allclose(
            qxyz[0],
            [0.36329995e-03, -0.18158424e-03, -0.89885493e-03],
            atol=1e-12,
        )

    def test_writer_format_matches_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            with MLHeatWriter(path) as writer:
                writer.write(
                    1,
                    np.array([0.36329995e-03, -0.18158424e-03, -0.89885493e-03]),
                )
            line = Path(path).read_text().splitlines()[0]

        # Must start with 'NSTEP=' followed by right-aligned step number, then
        # ' QXYZ=' and three 16-wide scientific fields.
        self.assertTrue(line.startswith("NSTEP="))
        self.assertIn(" QXYZ= ", line)
        # Round-trip the line we just wrote through the parser.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            Path(path).write_text(line + "\n")
            steps, qxyz = read_ml_heat(path)
        self.assertEqual(steps.tolist(), [1])
        np.testing.assert_allclose(
            qxyz[0],
            [0.36329995e-03, -0.18158424e-03, -0.89885493e-03],
            atol=1e-10,
        )

    def test_reader_rejects_garbage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ML_HEAT")
            Path(path).write_text("this is not an ML_HEAT line\n")
            with self.assertRaises(ValueError):
                read_ml_heat(path)


class MLLheatIncarTests(unittest.TestCase):
    def test_default_is_false_with_interval_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "INCAR")
            Path(path).write_text("IBRION = 0\nNSW = 10\nPOTIM = 1.0\n")
            cfg = parse_incar(path)
        self.assertFalse(cfg.ML_LHEAT)
        self.assertEqual(cfg.ML_HEAT_INTERVAL, 1)

    def test_lheat_true_parses(self) -> None:
        text = textwrap.dedent(
            """
            IBRION = 0
            NSW    = 100
            POTIM  = 1.0
            ML_LHEAT = .TRUE.
            ML_HEAT_INTERVAL = 5
            """
        ).strip()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "INCAR")
            Path(path).write_text(text + "\n")
            cfg = parse_incar(path)
        self.assertTrue(cfg.ML_LHEAT)
        self.assertEqual(cfg.ML_HEAT_INTERVAL, 5)

    def test_invalid_interval_falls_back_to_one(self) -> None:
        text = "IBRION = 0\nNSW = 1\nPOTIM = 1.0\nML_LHEAT = T\nML_HEAT_INTERVAL = 0\n"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "INCAR")
            Path(path).write_text(text)
            cfg = parse_incar(path)
        self.assertTrue(cfg.ML_LHEAT)
        self.assertEqual(cfg.ML_HEAT_INTERVAL, 1)


if __name__ == "__main__":
    unittest.main()
