# vasp-mace

**VASP-like interface for structure relaxation, molecular dynamics, and energy calculations using MACE machine-learning potentials**

`vasp-mace` emulates VASP for fast, low-cost atomistic simulations using pretrained MACE interatomic potentials, with optional empirical dispersion corrections (DFT-D3).
It reads VASP-style inputs (`POSCAR`, `INCAR`) and produces VASP-compatible outputs (`CONTCAR`, `OUTCAR`, `OSZICAR`, `XDATCAR`, `vasprun.xml`), enabling seamless integration with existing VASP workflows and post-processing tools.

---

## Features

- **Single-point** energy, force, and stress evaluation (`NSW = 0`)
- **Geometry relaxation** of atomic positions and/or unit cell, driven by MACE potentials
- **Molecular dynamics** (NVE and NVT Langevin) with XDATCAR output
- **Selective dynamics**: per-atom coordinate fixing from POSCAR, preserved in CONTCAR
- **DFT-D3 dispersion correction** via `IVDW` in INCAR (zero-damping and Becke-Johnson variants, with optional three-body ATM term)
- **Multiple ISIF modes**: positions-only, full cell relaxation, constant-volume shape relaxation, volume-only
- **Force-based** (`EDIFFG < 0`) and **energy-based** (`EDIFFG > 0`) convergence criteria
- **Target pressure** support via `PSTRESS` (ISIF = 3)
- VASP-compatible output written to the run directory: `CONTCAR`, `OUTCAR`, `OSZICAR`, `XDATCAR`, `vasprun.xml`
- ASE trajectory and log written to `ase_files/` subdirectory, keeping the run directory clean

## When would you use this?

For most workflows, a pure Python/ASE script is the better way to run MACE (more flexible, easier to customise, no file overhead). `vasp-mace` is not trying to compete with that.

It exists for the cases where VASP-style files are what you already have or what you need:

- You have a set of `POSCAR`/`INCAR` files from a previous VASP project and want a quick MACE relaxation without rewriting any input.
- You are using an external code that reads VASP output (e.g. `vasprun.xml` for ShengBTE, `CONTCAR` for a downstream workflow) and you want MACE to slot in transparently.
- You are comparing MACE results against VASP calculations and prefer to keep the input/output format identical to reduce variables.
- You are simply too fond of VASP files to let go, and that is a perfectly valid reason.

Whatever brings you here, enjoy `vasp-mace`.

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

Download a pretrained MACE-MP-0 model checkpoint:

```bash
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model
```

Point `vasp-mace` to it by setting the environment variable (e.g. in your `.bashrc` or `.zshrc`):

```bash
export MACE_MODEL_PATH=/path/to/2024-01-07-mace-128-L2_epoch-199.model
```

---

## Usage

Place `POSCAR` and `INCAR` in your working directory, then run:

```bash
vasp-mace
```

The mode (single-point, relaxation, or MD) is determined automatically from the INCAR tags.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | `$MACE_MODEL_PATH` | Path to MACE `.model` checkpoint |
| `--device` | `auto` | `auto` (→ `cpu`), `cpu`, or `mps` |
| `--dtype` | `auto` | `auto` (→ `float64` on CPU), `float32`, or `float64` |
| `--optimizer` | `BFGS` | Fallback optimizer: `BFGS`, `FIRE`, or `LBFGS`. Overridden by `IBRION` if set in INCAR |

---

## INCAR parameters

Only the tags relevant to `vasp-mace` are parsed; all others are silently ignored.

### General

| Tag | Default | Description |
|-----|---------|-------------|
| `NSW` | `0` | Max ionic steps. `0` = single-point calculation |
| `IBRION` | `-1` | `-1` = use `--optimizer` CLI flag; `0` = MD; `1` = LBFGS; `2` = BFGS; `3` = FIRE |
| `EDIFFG` | `-0.05` | Convergence criterion. `< 0`: max force (eV/Å); `> 0`: energy change per ion (eV) |
| `ISIF` | `2` | Degrees of freedom to relax (see table below) |
| `PSTRESS` | `0.0` | Target hydrostatic pressure in kBar, applied when `ISIF = 3` |
| `IVDW` | `0` | Empirical dispersion correction (see table below) |

### ISIF modes

| `ISIF` | Positions | Cell shape | Cell volume |
|--------|-----------|------------|-------------|
| `2` | relaxed | fixed | fixed |
| `3` | relaxed | relaxed | relaxed |
| `4` | relaxed | relaxed | fixed |
| `7` | fixed | fixed | relaxed |
| `8` | relaxed | fixed | relaxed |

### IVDW (DFT-D3 dispersion)

| `IVDW` | Method |
|--------|--------|
| `0` | None (default) |
| `11` | D3(zero-damping) |
| `12` | D3(Becke-Johnson) |
| `13` | D3(zero-damping) + ATM three-body |
| `14` | D3(Becke-Johnson) + ATM three-body |

### Molecular dynamics (IBRION = 0)

| Tag | Default | Description |
|-----|---------|-------------|
| `MDALGO` | `2` | `1` = NVE (VelocityVerlet); `2` = NVT Langevin thermostat |
| `TEBEG` | `0.0` | Starting temperature (K). Velocities initialised from Maxwell-Boltzmann distribution |
| `TEEND` | `-1` | Ending temperature (K) for linear ramp; `-1` = same as `TEBEG` (constant temperature) |
| `POTIM` | `0.5` | MD timestep (fs). Use ≤ 1.0 fs for systems containing hydrogen |
| `NBLOCK` | `1` | Write XDATCAR frame and trajectory snapshot every `NBLOCK` steps |
| `SMASS` | `-3.0` | Langevin friction coefficient (ps⁻¹). Values ≤ 0 use a default of 0.01 fs⁻¹ |

---

## Output files

### Relaxation (NSW > 0, IBRION ≠ 0)

| File | Description |
|------|-------------|
| `CONTCAR` | Final structure in VASP format (preserves Selective Dynamics if present in POSCAR) |
| `OSZICAR` | Per-step energy, ΔE, and Fmax |
| `OUTCAR` | Minimal OUTCAR with lattice, stress tensor, forces, and per-step energies |
| `XDATCAR` | Trajectory of ionic positions (one frame per ionic step) |
| `vasprun.xml` | Minimal XML with energies and final structure |
| `ase_files/mace.traj` | Full ASE binary trajectory |
| `ase_files/opt.log` | ASE optimizer log |

### Molecular dynamics (IBRION = 0)

| File | Description |
|------|-------------|
| `CONTCAR` | Final structure |
| `XDATCAR` | Trajectory in fractional coordinates (written every `NBLOCK` steps) |
| `ase_files/mace.traj` | Full ASE binary trajectory |
| `ase_files/md.log` | ASE MD log (step, time, energy, temperature) |

### Single-point (NSW = 0)

| File | Description |
|------|-------------|
| `OUTCAR` | Lattice, stress tensor, and forces |
| `OSZICAR` | Single-line energy summary |
| `vasprun.xml` | Full single-point XML compatible with ShengBTE and Phonopy |

---

## Examples

Ready-to-run examples are provided in the `examples/` directory. Copy an example folder to your working directory and run `vasp-mace` inside it.

| Example | System | Description |
|---------|--------|-------------|
| `example01_MgO/` | MgO (rock salt, conventional cell) | Variable-cell relaxation (`ISIF = 3`), no dispersion |
| `example02_hBN_D3-dispersion/` | h-BN (hexagonal) | Variable-cell relaxation with D3(BJ) dispersion (`IVDW = 12`) |
| `example03_CsPbI3_MA_MD/` | Cs₆₃MA·PbI₃ perovskite (4×4×4, 327 atoms) | NVT Langevin MD at 500 K with one methylammonium defect |
| `example04_PbTe_pressure/` | PbTe (rock salt) | Variable-cell relaxation under 15 kBar target pressure (`PSTRESS = 15`) |

### example01 — MgO variable-cell relaxation

```
NSW    = 100
ISIF   = 3
EDIFFG = -0.01
```

### example02 — h-BN with DFT-D3 dispersion

```
NSW    = 100
ISIF   = 3
EDIFFG = -0.01
IVDW   = 12
```

### example03 — CsPbI₃ perovskite MD with methylammonium

4×4×4 supercell (327 atoms) of cubic CsPbI₃ with one Cs site replaced by methylammonium (CH₃NH₃⁺). NVT Langevin thermostat at 500 K.

```
IBRION = 0
MDALGO = 2
NSW    = 200
TEBEG  = 500
POTIM  = 1.0
NBLOCK = 5
SMASS  = 1.0
```

### example04 — PbTe under pressure

```
NSW     = 100
ISIF    = 3
EDIFFG  = -0.01
PSTRESS = 15
```

---

## Development

```bash
pip install -e ".[dev]"   # installs pytest, black, ruff, mypy
ruff check vasp_mace/     # lint
black vasp_mace/          # format
```

---

## License and citation

MIT License © 2025 Ricardo Grau-Crespo.

If you use `vasp-mace` in your work, please cite:

**vasp-mace:**
- Grau-Crespo, R. *vasp-mace: a VASP simulator based on the MACE machine-learning interatomic potential* (2025). Zenodo. https://doi.org/10.5281/zenodo.19479246

```bibtex
@software{graucrespo2025vaspmace,
  author  = {Grau-Crespo, Ricardo},
  title   = {vasp-mace: a VASP simulator based on the MACE machine-learning interatomic potential},
  year    = {2025},
  url     = {https://github.com/rgraucrespo/vasp-mace},
  doi     = {10.5281/zenodo.19479246},
}
```

**MACE potentials:**
- Batatia, I.; Kovács, D. P.; Simm, G. N. C.; Ortner, C.; Csányi, G. “MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields”. Advances in Neural Information Processing Systems (NeurIPS), 2022.
- Batatia, I. et al. “A foundation model for atomistic materials chemistry.” The Journal of Chemical Physics 163, no. 18 (2025).

**VASP** (if referring to specific VASP formats or comparing against VASP results):
- Kresse, G.; Furthmüller, J. “Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set.” Computational Materials Science 6 (1996) 15–50.
- Kresse, G.; Furthmüller, J. “Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set.” Physical Review B 54 (1996) 11169–11186.


