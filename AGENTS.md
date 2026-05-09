# AGENTS.md — Development guide for AI coding agents

This file contains project-specific guidance for AI agents (Claude Code, Codex, etc.)
working on vasp-mace. Keep entries concise; expand as conventions solidify.

## Project overview

vasp-mace is a VASP-like CLI wrapper that runs ML potential calculations (primarily
MACE) using VASP-style input files (INCAR, POSCAR, KPOINTS) and producing VASP-style
outputs (OUTCAR, OSZICAR, CONTCAR, XDATCAR, vasprun.xml). It is a pure-Python package;
there is no Fortran, no VASP binary dependency, and no CUDA requirement at import time.

## Python compatibility

Target: **Python 3.9 – 3.12**. Never use syntax or stdlib features that require 3.10+
without a `from __future__ import annotations` guard or an explicit version check.
Specifically:
- Use `Optional[X]` / `Union[X, Y]` from `typing`, not `X | Y` union syntax.
- Use `from __future__ import annotations` in any new file that uses PEP 585 built-in
  generics (`list[str]`, `tuple[int, ...]`) in class bodies or function signatures.

## Code style

- Formatter: **black** (`line-length = 88`). Run `black vasp_mace/` before committing.
- Type hints: PEP 484 annotations on all public functions and dataclasses.
- Docstrings: NumPy style on all public classes and functions.
- Comments: only when the *why* is non-obvious; never paraphrase what the code does.

## Module layout

| Module | Responsibility |
|---|---|
| `cli.py` | Entry point; argument parsing, dispatch |
| `incar.py` | INCAR parsing → `IncarConfig` |
| `types_.py` | Shared dataclasses (`IncarConfig`, `MDRecord`) |
| `logging_utils.py` | `StepRecord`, `StepLogger` |
| `io_poscar.py` | POSCAR/CONTCAR/XDATCAR read/write |
| `io_outcar.py` | OUTCAR/OSZICAR write |
| `io_xml.py` | vasprun.xml write |
| `io_vasp.py` | Re-export shim (do not add logic here) |
| `mace_loader.py` | MACE calculator loading with device/dtype fallback |
| `relax.py` | Relaxation and single-point run mode |
| `md.py` | Molecular dynamics run mode |
| `phonons.py` | Phonon run mode (IBRION=5/6) |
| `elasticity.py` | Elastic tensor run mode |
| `neb.py` | NEB run mode |

## Testing

- Tests live in `tests/test_examples.py` (lightweight, no MACE model needed by default).
- Run with: `python scripts/run_tests.py`
- MACE-backed smoke runs: `python scripts/run_tests.py --with-examples --model <path>`
- Do not require a MACE model checkpoint for the default test suite.

## Changelog

`CHANGELOG.md` is the single source of truth for what has changed and what is planned.
- Record every meaningful change under `[Unreleased]` using the standard headings:
  Added / Changed / Fixed / Removed.
- Future plans and upcoming features are tracked there too; check it before starting
  work to avoid duplicating effort or conflicting with planned directions.
