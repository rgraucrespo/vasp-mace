"""Release packaging checks."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_has_no_direct_url_dependencies(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text()
        forbidden = (" @ git+", " @ http://", " @ https://")
        found = [token for token in forbidden if token in text]
        self.assertEqual(found, [], "PyPI metadata must not contain direct URLs")

    def test_heat_requirements_are_in_sdist_manifest(self) -> None:
        manifest = (REPO_ROOT / "MANIFEST.in").read_text()
        self.assertIn("include requirements-heat.txt", manifest)


if __name__ == "__main__":
    unittest.main()
