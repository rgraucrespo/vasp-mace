"""Tests built from the repository examples.

The default suite is intentionally lightweight: it verifies that every example
INCAR parses and every POSCAR can be read. MACE-backed smoke runs are opt-in via
RUN_VASP_MACE_EXAMPLES=1 because they require a model checkpoint and can take
noticeably longer than ordinary unit tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vasp_mace.incar import parse_incar
from vasp_mace.io_poscar import read_poscar


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    example_dir: str
    incar: str
    outputs: tuple[str, ...]
    tags: tuple[str, ...] = ()
    optional_import: Optional[str] = None


SMOKE_CASES = (
    SmokeCase(
        name="example01_relax_mgo",
        example_dir="example01_MgO",
        incar="""
            NSW    = 1
            ISIF   = 3
            EDIFFG = -10.0
        """,
        outputs=("CONTCAR", "OUTCAR", "OSZICAR", "XDATCAR", "vasprun.xml"),
        tags=("quick",),
    ),
    SmokeCase(
        name="example02_dispersion_hbn",
        example_dir="example02_hBN_D3-dispersion",
        incar="""
            NSW    = 1
            ISIF   = 3
            EDIFFG = -10.0
            IVDW   = 12
        """,
        outputs=("CONTCAR", "OUTCAR", "OSZICAR", "XDATCAR", "vasprun.xml"),
        tags=("all",),
        optional_import="dftd4",
    ),
    SmokeCase(
        name="example03_md_cspbi3",
        example_dir="example03_CsPbI3_MA_MD",
        incar="""
            IBRION = 0
            MDALGO = 2
            NSW    = 1
            TEBEG  = 100
            POTIM  = 0.5
            NBLOCK = 1
            SMASS  = 1.0
        """,
        outputs=("CONTCAR", "OUTCAR", "XDATCAR", "ase_files/mace.traj"),
        tags=("all",),
    ),
    SmokeCase(
        name="example04_pressure_pbte",
        example_dir="example04_PbTe_pressure",
        incar="""
            NSW     = 1
            ISIF    = 3
            EDIFFG  = -10.0
            PSTRESS = 15
        """,
        outputs=("CONTCAR", "OUTCAR", "OSZICAR", "XDATCAR", "vasprun.xml"),
        tags=("quick",),
    ),
    SmokeCase(
        name="example05_neb_si",
        example_dir="example05_Si_NEB",
        incar="""
            NSW    = 1
            EDIFFG = -10.0
            IBRION = 1
            ISIF   = 2
            IMAGES = 4
            SPRING = -5
            LCLIMB = .TRUE.
        """,
        outputs=(
            "00/CONTCAR",
            "01/CONTCAR",
            "05/CONTCAR",
            "00/OUTCAR",
            "01/OSZICAR",
            "05/vasprun.xml",
            "ase_files/mace.traj",
        ),
        tags=("all",),
    ),
    SmokeCase(
        name="example06_neb_pt",
        example_dir="example06_Pt_NEB",
        incar="""
            NSW    = 1
            EDIFFG = -10.0
            IBRION = 1
            ISIF   = 2
            IMAGES = 3
            SPRING = -5
            LCLIMB = .TRUE.
        """,
        outputs=(
            "00/CONTCAR",
            "01/CONTCAR",
            "04/CONTCAR",
            "00/OUTCAR",
            "01/OSZICAR",
            "04/vasprun.xml",
            "ase_files/mace.traj",
        ),
        tags=("all",),
    ),
    SmokeCase(
        name="example07_md_pbte_nvt",
        example_dir="example07_PbTe_MD",
        incar="""
            IBRION         = 0
            MDALGO         = 3
            NSW            = 1
            TEBEG          = 100
            POTIM          = 1.0
            NBLOCK         = 1
            ISIF           = 2
            LANGEVIN_GAMMA = 10.0 20.0
        """,
        outputs=("CONTCAR", "OUTCAR", "XDATCAR", "ase_files/mace.traj"),
        tags=("all",),
    ),
    SmokeCase(
        name="example08_phonons_pbte",
        example_dir="example08_PbTe_phonons",
        incar="""
            ISIF   = 2
            IBRION = 6
            NFREE  = 2
            POTIM  = 0.02
            NSW    = 1
        """,
        outputs=(
            "CONTCAR",
            "DYNMAT",
            "OUTCAR",
            "OSZICAR",
            "XDATCAR",
            "ase_files/force_constants.npy",
            "ase_files/phonopy_params.yaml",
        ),
        tags=("quick",),
        optional_import="phonopy",
    ),
    SmokeCase(
        name="example09_phonons_elastic_mgo",
        example_dir="example09_MgO_elastic",
        incar="""
            IBRION = 6
            NFREE  = 2
            POTIM  = 0.015
            ISIF   = 3
        """,
        outputs=(
            "CONTCAR",
            "DYNMAT",
            "OUTCAR",
            "OSZICAR",
            "XDATCAR",
            "ase_files/force_constants.npy",
            "ase_files/phonopy_params.yaml",
        ),
        tags=("all",),
        optional_import="phonopy",
    ),
    SmokeCase(
        name="example10_heat_flux_pbte",
        example_dir="example10_heat_flux",
        incar="""
            IBRION           = 0
            MDALGO           = 1
            ANDERSEN_PROB    = 0.0
            ISIF             = 2
            NSW              = 1
            TEBEG            = 300
            POTIM            = 1.0
            NBLOCK           = 1
            ML_LHEAT         = .TRUE.
            ML_HEAT_INTERVAL = 1
        """,
        outputs=(
            "CONTCAR",
            "OUTCAR",
            "XDATCAR",
            "ML_HEAT",
            "ase_files/ML_HEAT.json",
            "ase_files/mace.traj",
        ),
        tags=("all",),
        optional_import="mace_unfolded",
    ),
)


def _example_incar_paths() -> list[Path]:
    return sorted(
        p
        for p in EXAMPLES_DIR.rglob("*")
        if p.is_file() and (p.name == "INCAR" or p.name.startswith("INCAR_"))
    )


def _example_poscar_paths() -> list[Path]:
    return sorted(p for p in EXAMPLES_DIR.rglob("POSCAR") if p.is_file())


def _optional_module_available(module_name: str) -> bool:
    code = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
    )
    return subprocess.run([sys.executable, "-c", code]).returncode == 0


def _clean_incar(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


class ExampleInputTests(unittest.TestCase):
    def test_all_example_incars_parse(self) -> None:
        incars = _example_incar_paths()
        self.assertGreater(len(incars), 0, "No example INCAR files found")

        for path in incars:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                cfg = parse_incar(str(path))
                self.assertGreaterEqual(cfg.NSW, 0)
                self.assertIn(cfg.IVDW, (0, 11, 12, 13, 14))
                self.assertIn(cfg.NFREE, (1, 2))
                if cfg.IMAGES > 0:
                    self.assertLess(cfg.SPRING, 0)

    def test_all_example_poscars_read(self) -> None:
        poscars = _example_poscar_paths()
        self.assertGreater(len(poscars), 0, "No example POSCAR files found")

        for path in poscars:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                atoms = read_poscar(str(path))
                self.assertGreater(len(atoms), 0)
                self.assertEqual(len(atoms.get_positions()), len(atoms))

    def test_neb_examples_have_expected_image_layout(self) -> None:
        for incar_path in _example_incar_paths():
            cfg = parse_incar(str(incar_path))
            if cfg.IMAGES <= 0:
                continue

            with self.subTest(path=incar_path.relative_to(REPO_ROOT)):
                example_root = incar_path.parent
                for idx in range(cfg.IMAGES + 2):
                    poscar = example_root / f"{idx:02d}" / "POSCAR"
                    self.assertTrue(poscar.exists(), f"Missing {poscar}")


@unittest.skipUnless(
    os.environ.get("RUN_VASP_MACE_EXAMPLES") == "1",
    "Set RUN_VASP_MACE_EXAMPLES=1 to run MACE-backed example smoke tests",
)
class ExampleSmokeTests(unittest.TestCase):
    maxDiff = 4000

    def test_examples_run_as_smoke_cases(self) -> None:
        model = os.environ.get("MACE_MODEL_PATH")
        if not model:
            self.skipTest("MACE_MODEL_PATH is required for example smoke tests")
        if not Path(model).is_file():
            self.skipTest(f"MACE model checkpoint not found: {model}")

        selected = os.environ.get("VASP_MACE_EXAMPLE_SET", "quick").lower()
        if selected not in {"quick", "all"}:
            self.fail("VASP_MACE_EXAMPLE_SET must be 'quick' or 'all'")

        device = os.environ.get("VASP_MACE_TEST_DEVICE", "auto")
        dtype = os.environ.get("VASP_MACE_TEST_DTYPE", "auto")
        timeout = int(os.environ.get("VASP_MACE_TEST_TIMEOUT", "600"))

        cases = [
            case
            for case in SMOKE_CASES
            if selected == "all" or "quick" in case.tags
        ]
        self.assertGreater(len(cases), 0)

        for case in cases:
            with self.subTest(case=case.name):
                if case.optional_import and not _optional_module_available(
                    case.optional_import
                ):
                    self.skipTest(
                        f"{case.name} requires optional dependency "
                        f"{case.optional_import}"
                    )
                self._run_smoke_case(case, model, device, dtype, timeout)

    def _run_smoke_case(
        self,
        case: SmokeCase,
        model: str,
        device: str,
        dtype: str,
        timeout: int,
    ) -> None:
        src = EXAMPLES_DIR / case.example_dir
        self.assertTrue(src.exists(), f"Missing example directory: {src}")

        with tempfile.TemporaryDirectory(prefix=f"vasp_mace_{case.name}_") as td:
            run_dir = Path(td) / case.example_dir
            shutil.copytree(src, run_dir)
            (run_dir / "INCAR").write_text(_clean_incar(case.incar))

            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(REPO_ROOT)
                if not env.get("PYTHONPATH")
                else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
            )
            cmd = [
                sys.executable,
                "-c",
                "from vasp_mace.cli import main; main()",
                "--model",
                model,
                "--device",
                device,
                "--dtype",
                dtype,
            ]
            proc = subprocess.run(
                cmd,
                cwd=run_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )

            if proc.returncode != 0:
                tail = "\n".join(proc.stdout.splitlines()[-80:])
                self.fail(
                    f"{case.name} failed with exit code {proc.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output tail:\n{tail}"
                )

            for rel in case.outputs:
                out = run_dir / rel
                self.assertTrue(out.exists(), f"Expected output missing: {rel}")
                self.assertGreater(out.stat().st_size, 0, f"Empty output: {rel}")
