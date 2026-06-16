"""Release packaging checks."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import vasp_mace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(part) for part in version.split(".")]
    return tuple((parts + [0, 0, 0])[:3])


class PackagingMetadataTests(unittest.TestCase):
    def test_package_version_matches_pyproject(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version = "([^"]+)"', text, flags=re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml must declare a version")
        assert match is not None
        self.assertEqual(vasp_mace.__version__, match.group(1))

    def test_changelog_links_include_current_release(self) -> None:
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        version = vasp_mace.__version__
        releases = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)
        self.assertIn(version, releases)
        release_index = releases.index(version)
        self.assertLess(
            release_index + 1,
            len(releases),
            "current release must have a previous release for compare links",
        )
        previous = releases[release_index + 1]

        self.assertIn(f"## [{version}]", changelog)
        self.assertIn(
            f"[Unreleased]: https://github.com/rgraucrespo/vasp-mace/compare/v{version}...HEAD",
            changelog,
        )
        self.assertIn(
            f"[{version}]: https://github.com/rgraucrespo/vasp-mace/compare/v{previous}...v{version}",
            changelog,
        )

    def test_pyproject_has_no_direct_url_dependencies(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text()
        forbidden = (" @ git+", " @ http://", " @ https://")
        found = [token for token in forbidden if token in text]
        self.assertEqual(found, [], "PyPI metadata must not contain direct URLs")

    def test_ase_dependency_floor_includes_nose_hoover_chain_api(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text()
        match = re.search(r'"ase>=([^"]+)"', text)
        self.assertIsNotNone(match, "pyproject.toml must declare an ASE floor")
        assert match is not None
        floor = _version_tuple(match.group(1))
        self.assertGreaterEqual(
            floor,
            (3, 24, 0),
            "ase.md.nose_hoover_chain is not present before ASE 3.24.0",
        )

    def test_heat_requirements_are_in_sdist_manifest(self) -> None:
        manifest = (REPO_ROOT / "MANIFEST.in").read_text()
        self.assertIn("include requirements-heat.txt", manifest)


if __name__ == "__main__":
    unittest.main()
