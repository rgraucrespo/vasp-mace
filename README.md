# vasp-mace

**VASP-like interface for structure relaxation and energy calculations using MACE machine-learning potentials**

`vasp-mace` emulates VASP for fast, low-cost structure optimisations using pretrained MACE interatomic potentials, with optional empirical dispersion corrections (DFT-D3).  
It reads VASP-style inputs (`POSCAR`, `INCAR`) and produces VASP-compatible outputs (`CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`), enabling seamless integration with existing VASP workflows and post-processing tools.

---

## Features

- Geometry relaxation (atomic positions and/or unit cell) driven by MACE potentials
- Single-point energy and force evaluation (`NSW = 0`)
- Optional DFT-D3 dispersion correction via `IVDW = 12` in INCAR
- Supports `ISIF = 2` (positions only) and `ISIF = 3` (variable cell with target pressure via `PSTRESS`)
- Force-based (`EDIFFG < 0`) and energy-based (`EDIFFG > 0`) convergence criteria
- Selective dynamics (per-atom coordinate fixing) from POSCAR
- VASP-compatible output: `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`
- ASE trajectory and optimizer log written to `ase_files/` subdirectory, keeping the run directory clean

---

## Installation

```bash
git clone https://github.com/rgraucrespo/vasp-mace.git
cd vasp-mace
conda create -n vasp_mace_env python=3.11 -y
conda activate vasp_mace_env
pip install ase torch mace-torch
conda install -c conda-forge dftd4
pip install -e .
```

### Model checkpoint

Download the pretrained model (e.g MACE-MP-0):

```bash
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model
```

Set the path via the `MACE_MODEL_PATH` environment variable (e.g. in your `.bashrc`):

```bash
export MACE_MODEL_PATH=/path/to/2024-01-07-mace-128-L2_epoch-199.model
```

---

## Usage

Prepare `POSCAR` and `INCAR` as in a standard VASP calculation, then run:

```bash
vasp-mace
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | `$MACE_MODEL_PATH` | Path to MACE `.model` checkpoint |
| `--device` | `auto` | `auto` (→ cpu), `cpu`, or `mps` |
| `--dtype` | `auto` | `auto` (→ float64 on cpu), `float32`, or `float64` |
| `--optimizer` | `BFGS` | `BFGS` or `FIRE` |

---

## INCAR parameters

Only the tags relevant to `vasp-mace` are parsed; all others are silently ignored.

| Tag | Default | Description |
|-----|---------|-------------|
| `NSW` | `0` | Max ionic steps. `0` = single-point calculation |
| `ISIF` | `2` | `2` = relax positions only; `3` = relax cell + positions |
| `EDIFFG` | `-0.05` | Convergence criterion. `< 0`: max force in eV/Å; `> 0`: energy change per ion in eV |
| `PSTRESS` | `0.0` | Target pressure in kBar (applied when `ISIF = 3`) |
| `IVDW` | `0` | `0` = none; `11` = D3(zero); `12` = D3(BJ); `13` = D3(zero)+ATM; `14` = D3(BJ)+ATM. Other values exit with an error |

---

## Example

**INCAR** — variable-cell relaxation with dispersion:
```
NSW = 200
ISIF = 3
EDIFFG = -0.01
PSTRESS = 0
IVDW = 12
```

**Run:**
```bash
vasp-mace --device cpu --dtype float64
```

**Outputs:**
- VASP-compatible (written to the run directory): `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`
- ASE/MACE-specific (written to `ase_files/`): `mace.traj`, `opt.log`

---

## Examples

Ready-to-run examples are provided in the `examples/` directory. Copy the example folder to your working directory and run `vasp-mace` inside it.

| Example | System | Description |
|---------|--------|-------------|
| `example01_MgO/` | MgO (rock salt) | Variable-cell relaxation, no dispersion |
| `example02_hBN/` | h-BN (hexagonal) | Variable-cell relaxation with DFT-D3 (`IVDW = 12`) |

---

## License and citation

MIT License © 2025 Ricardo Grau-Crespo.

If you use `vasp-mace` in your work, please cite:

- Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS 2022
- Batatia et al., *A foundation model for atomistic materials chemistry*, arXiv:2401.00096 (2024)