# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`vasp-mace` is a lightweight Python CLI that emulates VASP geometry optimizations using MACE machine-learning interatomic potentials via ASE. It reads VASP-style inputs (`POSCAR`, `INCAR`) and produces VASP-compatible outputs (`CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`).

## Installation

```bash
conda create -n vasp_mace_env python=3.11 -y
conda activate vasp_mace_env
pip install ase torch mace-torch
conda install -c conda-forge dftd4
pip install -e .
```

Download the MACE model checkpoint:
```bash
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model
```

## Running

The CLI entry point is `vasp-mace`. It must be run in a directory containing `POSCAR` and `INCAR`:

```bash
vasp-mace                          # uses $MACE_MODEL_PATH or ~/software/mace/...model
vasp-mace --model /path/to/model   # explicit model path
vasp-mace --device cpu             # cpu, mps (default: auto→cpu)
vasp-mace --dtype float64          # float32, float64 (default: auto)
vasp-mace --optimizer FIRE         # BFGS or FIRE (default: BFGS)
```

## Development

```bash
pip install -e ".[dev]"   # installs pytest, black, ruff, mypy
pytest                    # run all tests (test/ directory)
ruff check vasp_mace/     # lint
black vasp_mace/          # format
```

## Architecture

The package (`vasp_mace/`) has a clear pipeline:

1. **`cli.py`** — entry point (`vasp-mace` command). Parses CLI args, reads `INCAR`/`POSCAR`, dispatches to single-point or relaxation, writes outputs.

2. **`incar.py`** — parses `INCAR` into an `IncarConfig` dataclass. Supported tags: `NSW`, `ISIF`, `EDIFFG`, `PSTRESS`. `NSW=0` → single-point; `NSW>0` → relaxation.

3. **`mace_loader.py`** — loads the `MACECalculator` with suppressed third-party output. Auto-selects CPU/float64 for robustness on macOS MPS.

4. **`relax.py`** — relaxation driver. Wraps atoms in `UnitCellFilter` for `ISIF=3` (variable cell). Iterates one ASE optimizer step at a time to emulate VASP ionic steps. Convergence logic:
   - `EDIFFG < 0`: force-based (`|F| ≤ |EDIFFG|`); ISIF=3 also checks per-component stress error against target pressure.
   - `EDIFFG > 0`: energy-per-ion based.
   - Writes `mace.traj` (ASE trajectory) and `opt.log` (optimizer log).

5. **`io_vasp.py`** — VASP-format I/O via ASE. Handles Selective Dynamics in POSCAR/CONTCAR. Writes OSZICAR, OUTCAR, CONTCAR, and two XML variants: `write_relax_vasprun_xml` (relaxation) vs `write_single_vasprun_xml` (single-point, compatible with ShengBTE/Phonopy).

6. **`types_.py`** — dataclasses: `IncarConfig` (INCAR parameters), `StepRecord` (per-ionic-step energy/force data).

7. **`logging_utils.py`** — `StepLogger` accumulates `StepRecord`s; `timer` context manager.

## Key INCAR Parameters

| Tag | Default | Meaning |
|-----|---------|---------|
| `NSW` | 0 | Max ionic steps; 0 = single-point |
| `ISIF` | 2 | 2 = positions only, 3 = variable cell |
| `EDIFFG` | -0.05 | Convergence: <0 = force (eV/Å), >0 = energy/ion (eV) |
| `PSTRESS` | 0.0 | Target pressure in kBar (ISIF=3 only) |
