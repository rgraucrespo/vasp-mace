"""Tests for NEB SPRING sign validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vasp_mace.incar import parse_incar
from vasp_mace.neb import run_neb


class NEBSpringValidationTests(unittest.TestCase):
    def _parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "INCAR"
            path.write_text(text.strip() + "\n")
            return parse_incar(str(path))

    def test_neb_accepts_negative_spring(self) -> None:
        cfg = self._parse_text("IMAGES = 1\nNSW = 1\nSPRING = -5\n")
        self.assertEqual(cfg.SPRING, -5.0)

    def test_neb_rejects_positive_spring_at_parse_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative SPRING"):
            self._parse_text("IMAGES = 1\nNSW = 1\nSPRING = 5\n")

    def test_neb_rejects_zero_spring_at_parse_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative SPRING"):
            self._parse_text("IMAGES = 1\nNSW = 1\nSPRING = 0\n")

    def test_non_neb_positive_spring_is_ignored(self) -> None:
        cfg = self._parse_text("IMAGES = 0\nSPRING = 5\n")
        self.assertEqual(cfg.SPRING, 5.0)

    def test_neb_rejects_zero_nsw_at_parse_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "NSW=0 is not supported for NEB"):
            self._parse_text("IMAGES = 1\nSPRING = -5\n")

    def test_run_neb_rejects_constructed_config_with_positive_spring(self) -> None:
        cfg = SimpleNamespace(IMAGES=1, SPRING=5.0)
        with self.assertRaisesRegex(ValueError, "negative SPRING"):
            run_neb(cfg, model_path="unused.model")

    def test_run_neb_rejects_constructed_config_with_zero_nsw(self) -> None:
        cfg = SimpleNamespace(IMAGES=1, SPRING=-5.0, NSW=0)
        with self.assertRaisesRegex(ValueError, "NSW=0 is not supported for NEB"):
            run_neb(cfg, model_path="unused.model")


if __name__ == "__main__":
    unittest.main()
