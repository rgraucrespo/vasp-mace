# vasp-mace

**VASP-like interface for structure relaxation, molecular dynamics, and energy calculations using MACE machine-learning potentials**

`vasp-mace` emulates VASP for fast, low-cost atomistic simulations using pretrained MACE interatomic potentials, with optional empirical dispersion corrections (DFT-D3).
It reads VASP-style inputs (`POSCAR`, `INCAR`) and produces VASP-compatible outputs (`CONTCAR`, `OUTCAR`, `OSZICAR`, `XDATCAR`, `vasprun.xml`), enabling seamless integration with existing VASP workflows and post-processing tools.

---

## Features

- **Single-point** energy, force, and stress evaluation (`NSW = 0`)
- **Geometry relaxation** of atomic positions and/or unit cell, driven by MACE potentials
- **Molecular dynamics** (NVE, NVT Langevin/Nosé-Hoover/Andersen, NPT Langevin) with XDATCAR output
- **Nudged Elastic Band (NEB)**: minimum-energy path and transition-state search via ASE's MDMin optimizer; optional climbing-image NEB (`LCLIMB = .TRUE.`, VTST convention)
- **Selective dynamics**: per-atom coordinate fixing from POSCAR, preserved in CONTCAR
- **DFT-D3 dispersion correction** via `IVDW` in INCAR (zero-damping and Becke-Johnson variants, with optional three-body ATM term)
- **Multiple ISIF modes**: positions-only, full cell relaxation, constant-volume shape relaxation, volume-only
- **Force-based** (`EDIFFG < 0`) and **energy-based** (`EDIFFG > 0`) convergence criteria
- **Target pressure** support via `PSTRESS` (ISIF = 3)
- VASP-compatible output written to the run directory: `CONTCAR`, `OUTCAR`, `OSZICAR`, `XDATCAR`, `vasprun.xml`
- ASE trajectory and log written to `ase_files/` subdirectory, keeping the run directory clean with only VASP-like output files

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
conda create -n vasp_mace_env python=3.11 -y
conda activate vasp_mace_env
conda install -c conda-forge dftd4
pip install vasp-mace
```

> **DFT-D3 dispersion** (`IVDW` tag) requires `dftd4`, which is best installed via conda before `pip install vasp-mace`. If you do not need dispersion corrections, the conda step can be skipped.

### Development install (includes examples)

To get the example input files or contribute to the code, clone the repository instead:

```bash
git clone https://github.com/rgraucrespo/vasp-mace.git
cd vasp-mace
conda create -n vasp_mace_env python=3.11 -y
conda activate vasp_mace_env
conda install -c conda-forge dftd4
pip install -e .
```

### Model checkpoint

Download a pretrained MACE model checkpoint, for example:

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

The mode (single-point, relaxation, NEB, or MD) is determined automatically from the INCAR tags.

---

## INCAR parameters

Only the tags relevant to `vasp-mace` are parsed; all others are silently ignored.

### General

| Tag | Default | Description |
|-----|---------|-------------|
| `NSW` | `0` | Max ionic steps. `0` = single-point calculation |
| `IBRION` | `-1` | `-1` = use `--optimizer` CLI flag; `0` = MD; `1` = LBFGS; `2` = BFGS; `3` = FIRE. For NEB (`IMAGES > 0`): `1` and `2` both map to MDMin (the recommended NEB optimizer) |
| `EDIFFG` | `-0.05` | Convergence criterion. `< 0`: max force (eV/Å); `> 0`: energy change per ion (eV) |
| `ISIF` | `2` | Degrees of freedom to relax (see table below) |
| `PSTRESS` | `0.0` | Target hydrostatic pressure in kBar, applied when `ISIF = 3` |
| `IVDW` | `0` | Empirical dispersion correction (see table below) |

### ISIF modes

| `ISIF` | Positions | Cell shape | Cell volume | Note |
|--------|-----------|------------|-------------|------|
| `2` | relaxed | fixed | fixed | MD: NVT |
| `3` | relaxed | relaxed | relaxed | MD: NPT (`MDALGO=3`) |
| `4` | relaxed | relaxed | fixed | Relax only |
| `7` | fixed | fixed | relaxed | Relax only |
| `8` | relaxed | fixed | relaxed | Relax only |

### IVDW (DFT-D3 dispersion)

| `IVDW` | Method |
|--------|--------|
| `0` | None (default) |
| `11` | D3(zero-damping) |
| `12` | D3(Becke-Johnson) |
| `13` | D3(zero-damping) + ATM three-body |
| `14` | D3(Becke-Johnson) + ATM three-body |

### Nudged Elastic Band (IMAGES ≥ 1)

Triggered when `IMAGES > 0` in INCAR. No top-level `POSCAR` is used; instead, place endpoint and (optionally) intermediate images in numbered subdirectories:

```
00/POSCAR   ← reactant (fixed endpoint)
01/POSCAR   ← intermediate image 1  (optional; generated by IDPP if absent)
…
NN/POSCAR   ← product  (fixed endpoint)   NN = IMAGES + 1
```

If intermediate POSCARs are absent, all images are generated automatically by IDPP interpolation.

| Tag | Default | Description |
|-----|---------|-------------|
| `IMAGES` | `0` | Number of intermediate NEB images. `IMAGES ≥ 1` triggers NEB mode |
| `SPRING` | `-5.0` | Spring constant for NEB (eV/Å²). Use negative values (`SPRING < 0`, VASP convention for NEB); the spring constant is `k = |SPRING|`. Positive values correspond to the non-nudged elastic band and are not supported |
| `LCLIMB` | `.FALSE.` | Enable climbing-image NEB (CI-NEB). **Not a native VASP tag** — borrowed from the [VTST Tools](https://theory.cm.utexas.edu/vtsttools/neb.html) convention (see note below) |

> **`LCLIMB` and VTST convention**: In VASP with the optional VTST extension, CI-NEB is activated by `LCLIMB = .TRUE.`. Native VASP (without VTST) does not recognise this tag and always runs plain NEB. `vasp-mace` follows the VTST convention so that INCAR files from VTST-enabled VASP work without modification.  
> `SPRING` follows the VASP sign convention: negative values (`SPRING < 0`) indicate NEB, and the spring constant is `k = |SPRING|`. Positive values correspond to the non-nudged elastic band method and are not supported by `vasp-mace`. CI-NEB is controlled exclusively by `LCLIMB`, not by the sign of `SPRING`.

### Molecular dynamics (IBRION = 0)

| Tag | Default | Description |
|-----|---------|-------------|
| `MDALGO` | `3` | `1` = VelocityVerlet: NVE if `ANDERSEN_PROB = 0`, NVT Andersen if `ANDERSEN_PROB > 0`; `2` = NVT Nosé-Hoover; `3` = NVT Langevin (`ISIF=2`) or NPT Langevin (`ISIF=3`) |
| `TEBEG` | `0.0` | Starting temperature (K). Velocities initialised from Maxwell-Boltzmann distribution |
| `TEEND` | `-1` | Ending temperature (K) for linear ramp; `-1` = same as `TEBEG` (constant temperature) |
| `POTIM` | `0.5` | MD timestep (fs). Use ≤ 1.0 fs for systems containing hydrogen |
| `NBLOCK` | `1` | Write XDATCAR frame and trajectory snapshot every `NBLOCK` steps |
| `ANDERSEN_PROB` | `0.0` | Collision probability for Andersen thermostat (`MDALGO = 1`) |
| `LANGEVIN_GAMMA` | `10.0` | Friction coefficient(s) (ps⁻¹) for atoms in Langevin MD (`MDALGO = 3`). A single value applies to all atoms; multiple space-separated values are assigned per species in POSCAR order. Also reads from `SMASS` if `LANGEVIN_GAMMA` is missing |
| `LANGEVIN_GAMMA_L` | `10.0` | Friction coefficient (ps⁻¹) for the lattice in Langevin NPT (`MDALGO = 3, ISIF = 3`) |
| `PMASS` | `0` | Piston mass (amu) for Langevin NPT (`MDALGO = 3, ISIF = 3`). `0` = automatic (`N × 10000` amu) |
| `SMASS` | `-3.0` | Nose-Hoover mass or Langevin friction. Values > 0 are used if `LANGEVIN_GAMMA` is missing |

#### NVE — microcanonical ensemble

Use `MDALGO = 1` with `ANDERSEN_PROB = 0.0` (no collisions → pure VelocityVerlet integrator).

```
IBRION        = 0
MDALGO        = 1
ANDERSEN_PROB = 0.0
NSW           = 5000
TEBEG         = 300      # sets initial velocity distribution only
POTIM         = 1.0
NBLOCK        = 10
```

#### NVT — canonical ensemble

Three thermostat options are available:

**Andersen thermostat** (`MDALGO = 1`, `ANDERSEN_PROB > 0`): stochastic velocity rescaling at each step. Simple and robust; `ANDERSEN_PROB` controls how frequently velocities are reassigned from the Maxwell-Boltzmann distribution (typical range: 0.01–0.1).

```
IBRION        = 0
MDALGO        = 1
ANDERSEN_PROB = 0.05
NSW           = 5000
TEBEG         = 300
POTIM         = 1.0
NBLOCK        = 10
```

**Nosé-Hoover thermostat** (`MDALGO = 2`): deterministic extended-system thermostat; generates a correct NVT ensemble. `SMASS > 0` sets the coupling time (ps); the default (`SMASS ≤ 0`) uses a period of 40 MD steps.

```
IBRION = 0
MDALGO = 2
SMASS  = 1.0
NSW    = 5000
TEBEG  = 300
POTIM  = 1.0
NBLOCK = 10
```

**Langevin thermostat** (`MDALGO = 3`, `ISIF = 2`): stochastic friction + random force; well-suited for systems with slow equilibration. `LANGEVIN_GAMMA` accepts a single value (all atoms) or one value per species in POSCAR order.

```
IBRION         = 0
MDALGO         = 3
ISIF           = 2
LANGEVIN_GAMMA = 10.0 20.0   # per-species: species1=10, species2=20 ps^-1
NSW            = 5000
TEBEG          = 300
POTIM          = 1.0
NBLOCK         = 10
```

#### NPT — isothermal-isobaric ensemble

Use `MDALGO = 3` with `ISIF = 3`. The Langevin barostat controls the cell volume; `LANGEVIN_GAMMA_L` sets the lattice friction. Set `PSTRESS` to the target pressure in kBar (0 = ambient pressure). `PMASS` sets the barostat piston mass in amu (default: `N × 10000` amu).

```
IBRION           = 0
MDALGO           = 3
ISIF             = 3
LANGEVIN_GAMMA   = 10.0 20.0   # per-species: species1=10, species2=20 ps^-1
LANGEVIN_GAMMA_L = 10.0
PMASS            = 50000
PSTRESS          = 0.0
NSW              = 5000
TEBEG            = 300
POTIM            = 1.0
NBLOCK           = 10
```

---

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | `$MACE_MODEL_PATH` | Path to MACE `.model` checkpoint |
| `--device` | `auto` | `auto` (→ `cpu`), `cpu`, or `mps` |
| `--dtype` | `auto` | `auto` (→ `float64` on CPU), `float32`, or `float64` |
| `--optimizer` | `BFGS` | Fallback optimizer: `BFGS`, `FIRE`, or `LBFGS`. Overridden by `IBRION` if set in INCAR |


---

## Differences with respect to VASP

While `vasp-mace` aims for a high degree of compatibility, there are important technical differences to keep in mind:

- **Electronic Steps**: Since MACE is a machine-learning potential, there are no "electronic steps" in the DFT sense. For compatibility with tools like `vasprun.xml`, a single dummy electronic step is recorded per ionic step.
- **Nosé-Hoover Coupling**: In VASP, `SMASS` directly sets the thermostat mass ($Q$). In `vasp-mace`, if `SMASS > 0`, it is treated as a characteristic damping time in picoseconds ($t_{damp} = \text{SMASS} \times 1 \text{ ps}$). The default `SMASS = 0` (or $\le 0$) correctly maps to an oscillation period of 40 time steps, matching VASP's default behavior.
- **Langevin NPT Algorithm**: The NPT implementation in `vasp-mace` (`MDALGO = 3`, `ISIF = 3`) uses the stochastic barostat algorithm of Quigley and Probert (2004). This correctly samples the NPT ensemble but may fluctuate differently than VASP's internal implementation.
- **Piston Mass**: The lattice "piston mass" for NPT defaults to `N × 10000` amu, but can be set explicitly via the `PMASS` INCAR tag (in amu, matching VASP's convention).
- **Optimizers**: Relaxation (`IBRION = 1, 2, 3`) uses ASE's robust optimizers (LBFGS, BFGS, and FIRE) rather than VASP's internal RMM-DIIS or conjugate gradient routines.
- **NEB Optimizer**: NEB calculations always use ASE's `MDMin` optimizer (regardless of `IBRION`). `MDMin` projects velocities along the force direction and resets them when the velocity and force point in opposite directions, which prevents the divergence that plagues BFGS and FIRE under non-conservative spring forces. VASP with VTST uses a different quasi-Newton method.
- **`LCLIMB` tag**: Not a native VASP tag. It originates from the [VTST Tools](https://theory.cm.utexas.edu/vtsttools/neb.html) extension package for VASP. Native VASP (without VTST) does not recognise `LCLIMB` and always runs plain NEB.


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

### NEB (IMAGES ≥ 1)

Written per image directory (`00/`, `01/`, …, `NN/`):

| File | Description |
|------|-------------|
| `CONTCAR` | Final image geometry |
| `OSZICAR` | Per-step energy for this image |
| `OUTCAR` | Forces and energies for this image |
| `vasprun.xml` | Single-point or relaxation XML for this image |

Shared output in `ase_files/`:

| File | Description |
|------|-------------|
| `ase_files/neb_opt.log` | MDMin optimizer log |
| `ase_files/mace.traj` | Converged NEB band as ASE trajectory (one frame per image, reactant → product) |

No `XDATCAR` is produced for NEB runs.

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
| `example03_CsPbI3_MA_MD/` | Cs₆₃MA·PbI₃ perovskite (4×4×4, 327 atoms) | NVT Nosé-Hoover MD at 500 K with one methylammonium defect |
| `example04_PbTe_pressure/` | PbTe (rock salt) | Variable-cell relaxation under 15 kBar target pressure (`PSTRESS = 15`) |
| `example05_Si_NEB/` | Si (diamond cubic) | CI-NEB for Si interstitial migration (`LCLIMB = .TRUE.`, 4 intermediate images) |
| `example06_Pt_NEB/` | Pt (fcc-001 surface) | CI-NEB for Pt adatom collective jump (`LCLIMB = .TRUE.`, 3 intermediate images) |
| `example07_PbTe_MD/` | PbTe (rock salt, 512 atoms) | Sequential NVT → NPT Langevin MD; per-species `LANGEVIN_GAMMA` and explicit `PMASS` |

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

4×4×4 supercell (327 atoms) of cubic CsPbI₃ with one Cs site replaced by methylammonium (CH₃NH₃⁺). NVT Nosé-Hoover thermostat at 500 K.

```
IBRION = 0
MDALGO = 2
NSW    = 200
TEBEG  = 500
POTIM  = 0.5
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

### example05 — Si interstitial migration (CI-NEB)

4-image CI-NEB for a Si self-interstitial hop in the diamond cubic lattice. Intermediate images are provided as starting POSCARs (previously converged); IDPP interpolation is used automatically if they are absent.

```
NSW    = 100
EDIFFG = -0.01
IBRION = 1
ISIF   = 2
IMAGES = 4
SPRING = -5
LCLIMB = .TRUE.
```

### example06 — Pt adatom collective jump (CI-NEB)

3-image CI-NEB for a collective Pt adatom jump on the fcc-Pt(001) surface. Endpoint and intermediate POSCARs are provided.

```
NSW    = 100
EDIFFG = -0.01
IBRION = 1
ISIF   = 2
IMAGES = 3
SPRING = -5
LCLIMB = .TRUE.
```

### example07 — PbTe sequential NVT → NPT MD

512-atom PbTe supercell. Demonstrates per-species `LANGEVIN_GAMMA` (Pb and Te assigned different friction coefficients) and explicit `PMASS`. Run with the provided `run.sh`, which chains the two stages automatically and saves outputs to `nvt_output/` and `npt_output/`.

**Stage 1 — NVT equilibration** (`INCAR_NVT`):

```
IBRION         = 0
MDALGO         = 3
ISIF           = 2
LANGEVIN_GAMMA = 10.0 20.0   # Pb: 10 ps^-1, Te: 20 ps^-1
NSW            = 500
TEBEG          = 300
POTIM          = 1.0
NBLOCK         = 1
```

**Stage 2 — NPT production** (`INCAR_NPT`, starts from NVT `CONTCAR`):

```
IBRION           = 0
MDALGO           = 3
ISIF             = 3
LANGEVIN_GAMMA   = 10.0 20.0   # Pb: 10 ps^-1, Te: 20 ps^-1
LANGEVIN_GAMMA_L = 10.0
PMASS            = 50000
NSW              = 500
TEBEG            = 300
POTIM            = 1.0
NBLOCK           = 1
```

```bash
bash run.sh --model /path/to/model
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


