# vasp-mace

**VASP-like interface for structure relaxation, molecular dynamics, and energy calculations using MACE machine-learning potentials**

`vasp-mace` emulates VASP for fast, low-cost atomistic simulations using pretrained MACE interatomic potentials, with optional empirical dispersion corrections (DFT-D3).
It reads VASP-style inputs (`POSCAR`, `INCAR`) and produces VASP-compatible outputs (`CONTCAR`, `OUTCAR`, `OSZICAR`, `XDATCAR`, `vasprun.xml`), enabling seamless integration with existing VASP workflows and post-processing tools.

---

## Trademark and project notice

`vasp-mace` is an independent open-source project and is not affiliated with,
endorsed by, or sponsored by VASP Software GmbH. VASP is a trademark of VASP
Software GmbH. This project does not include, call, wrap, modify, or distribute
VASP, VASP source code, POTCAR files, PAW datasets, or any licensed VASP
components. It implements independent surrogate-model calculations and
reads/writes selected VASP-style input/output files solely for workflow
interoperability.

See [NOTICE.md](NOTICE.md) for the repository-level notice.

---

## Features

- **Single-point** energy, force, and stress evaluation (`NSW = 0`)
- **Geometry relaxation** of atomic positions and/or unit cell, driven by MACE potentials
- **Molecular dynamics** (NVE, NVT Langevin/Nosé-Hoover/Andersen, NPT Langevin) with XDATCAR output
- **Heat flux for Green-Kubo** (`ML_LHEAT = .TRUE.`): per-step VASP-compatible `ML_HEAT` from MACE MD via the unfolded-cell autograd backend (`mace-unfolded`); for 3D bulk solids only. Post-process with [`sportran`](https://www.sciencedirect.com/science/article/abs/pii/S0010465522001898) for thermal conductivity
- **Nudged Elastic Band (NEB)**: minimum-energy path and transition-state search via ASE's MDMin optimizer; optional climbing-image NEB (`LCLIMB = .TRUE.`, VTST convention)
- **Phonon calculations**: Γ-point force constants and frequencies via finite differences (`IBRION = 5`); symmetry-reduced displacements via phonopy (`IBRION = 6`), with VASP-compatible `DYNMAT` and `OUTCAR` output
- **Elastic constants**: full 6×6 elastic tensor, Voigt/Reuss/Hill polycrystalline averages (K, G, E, ν) via stress-strain finite differences — triggered by `ISIF ≥ 3` alongside `IBRION = 5/6`
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
| `IBRION` | `-1` | `-1` = use `--optimizer` CLI flag; `0` = MD; `1` = LBFGS; `2` = BFGS; `3` = FIRE; `5` = phonons (no symmetry); `6` = phonons (symmetry-reduced via phonopy). For NEB (`IMAGES > 0`): `1` and `2` both map to MDMin (the recommended NEB optimizer) |
| `EDIFFG` | `-0.05` | Convergence criterion. `< 0`: max force (eV/Å); `> 0`: energy change per ion (eV) |
| `ISIF` | `2` | Degrees of freedom to relax (see table below) |
| `PSTRESS` | `0.0` | Target hydrostatic pressure in kBar, applied when `ISIF = 3` |
| `IVDW` | `0` | Empirical dispersion correction (see table below) |

### ISIF modes

| `ISIF` | Positions | Cell shape | Cell volume | Note |
|--------|-----------|------------|-------------|------|
| `0`, `1`, `2` | relaxed | fixed | fixed | MD: NVT. `0` and `1` are treated as `2` (in VASP they differ only in how much of the stress tensor is computed; `vasp-mace` always computes the full stress) |
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

### Phonon calculations (IBRION = 5 or 6)

Finite-difference second derivatives (force constants and phonon frequencies) at the Γ-point.

| Tag | Default | Description |
|-----|---------|-------------|
| `POTIM` | `0.015` | Displacement amplitude (Å). In phonon mode this sets the finite-difference step, **not** the MD timestep. VASP uses 0.02 Å; either value is acceptable |
| `NFREE` | `2` | Displacements per degree of freedom: `2` = central differences (±δ, recommended); `1` = forward differences (+δ only) |
| `ISIF` | `2` | `2` = force constants and phonon frequencies only; `≥ 3` = also compute the elastic tensor (see below) |

**IBRION = 5** computes all N × 3 × NFREE displacements (no symmetry).  
**IBRION = 6** uses [phonopy](https://phonopy.github.io/phonopy/) to reduce the number of displacements via crystal symmetry. Install with `pip install vasp-mace[phonons]` or `pip install phonopy`. Falls back to IBRION = 5 if phonopy is not installed.

Output files produced (VASP-compatible format):

| File | Description |
|------|-------------|
| `DYNMAT` | Force-constant matrix in VASP DYNMAT format (central-difference half-forces) |
| `OUTCAR` | Phonon eigenvalues and eigenvectors (modes ordered high → low frequency); elastic tensor appended when `ISIF ≥ 3` |
| `OSZICAR` | One energy line per displaced configuration |
| `XDATCAR` | Initial + all displaced configurations |
| `CONTCAR` | Initial (equilibrium) structure |
| `ase_files/force_constants.npy` | Force constant tensor C[i,α,j,β] in eV/Å² (shape N×3×N×3) |
| `ase_files/phonopy_params.yaml` | Phonopy parameters file (IBRION = 6 only) |

> **Note**: IBRION = 5/6 performs single-point force evaluations only (no ionic relaxation). The structure should be pre-relaxed to a minimum before running phonon calculations.

### Elastic constants (IBRION = 5 or 6 with ISIF ≥ 3)

Setting `ISIF ≥ 3` alongside `IBRION = 5/6` activates the elastic tensor calculation, matching VASP behavior. After the phonon displacements are complete, 12 additional single-point calculations are performed (6 Voigt strain patterns × ±1% strain, central differences), and the resulting stress-strain relationship is used to build the full 6×6 elastic tensor C_ij.

The elastic tensor, together with Voigt, Reuss, and Hill polycrystalline averages, is appended to `OUTCAR` in VASP format (kBar units, XX YY ZZ XY YZ ZX column order):

```
 TOTAL ELASTIC MODULI (kBar)
 Direction    XX          YY          ZZ          XY          YZ          ZX
 -----------------------------------------------------------------------------------------
  XX       2469.846     877.267     877.267       0.000       0.000       0.000
  ...

 POLYCRYSTALLINE CONSTANTS (Voigt / Reuss / Hill):
               Bulk modulus K  Shear modulus G  Young mod. E  Poisson ratio
                        (GPa)           (GPa)         (GPa)
  Voigt              140.813         102.442
  Reuss              140.813          98.784
  Hill               140.813         100.613       243.779        0.2115
```

A human-readable summary is also printed to stdout (GPa, ASE Voigt ordering xx yy zz yz xz xy).

The polycrystalline averages follow the Voigt–Reuss–Hill scheme ([de Jong et al., *Scientific Data* 2015](https://doi.org/10.1038/sdata.2015.9), Table 2), where **S** = **C**⁻¹ is the Voigt compliance tensor:

| Quantity | Formula |
|----------|---------|
| Voigt bulk modulus | K_V = (C₁₁+C₂₂+C₃₃ + 2(C₁₂+C₁₃+C₂₃)) / 9 |
| Reuss bulk modulus | K_R = 1 / (S₁₁+S₂₂+S₃₃ + 2(S₁₂+S₁₃+S₂₃)) |
| Voigt shear modulus | G_V = (C₁₁+C₂₂+C₃₃ − C₁₂−C₁₃−C₂₃ + 3(C₄₄+C₅₅+C₆₆)) / 15 |
| Reuss shear modulus | G_R = 15 / (4(S₁₁+S₂₂+S₃₃) − 4(S₁₂+S₁₃+S₂₃) + 3(S₄₄+S₅₅+S₆₆)) |
| Hill bulk modulus | K_VRH = (K_V + K_R) / 2 |
| Hill shear modulus | G_VRH = (G_V + G_R) / 2 |
| Young's modulus | E = 9 K_VRH G_VRH / (3 K_VRH + G_VRH) |
| Poisson's ratio | ν = (3 K_VRH − 2 G_VRH) / (6 K_VRH + 2 G_VRH) |

These expressions are valid for all crystal systems (cubic to triclinic).

> **Internal strain tensor**: not computed (vasp-mace computes the macroscopic elastic tensor only).

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
| `ML_LHEAT` | `.FALSE.` | Write a VASP-compatible `ML_HEAT` file (and `ML_HEAT.json` sidecar) during MD. See [Heat flux (ML_HEAT)](#heat-flux-ml_heat) |
| `ML_HEAT_INTERVAL` | `1` | `vasp-mace` extension: write `ML_HEAT` every `ML_HEAT_INTERVAL` MD steps. `1` matches VASP's per-step cadence |

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

### Heat flux (ML_HEAT)

Setting `ML_LHEAT = .TRUE.` in INCAR enables per-step heat-flux output during a MACE MD run. `vasp-mace` writes a VASP-compatible `ML_HEAT` file in the run directory (one `NSTEP=… QXYZ= …` line per recorded step, units `eV·Å·fs⁻¹`), plus an `ase_files/ML_HEAT.json` sidecar describing the timestep, write interval, target temperature, cell volume at MD start, backend, model, dtype, and device. `ML_HEAT` itself is byte-compatible with VASP's `ML_LHEAT` output, so downstream analysis tools that read VASP `ML_HEAT` files work unchanged; the JSON sidecar lives under `ase_files/` because VASP does not produce it.

**Backend.** `vasp-mace` does not implement the heat flux directly. It wraps [`mace-unfolded`](https://github.com/pulgon-project/mace-unfolded) (Wieser *et al.*, *J. Chem. Theory Comput.* **22**, 513 (2026)), which evaluates the autograd-based potential heat flux on an unfolded nonperiodic environment. Install the optional dependency:

```bash
pip install -e ".[heat]"
```

The `[heat]` extra installs `mace-unfolded` and its `comms` runtime dependency directly from GitHub (neither is published on PyPI). PyPI installs of `vasp-mace` therefore cannot resolve the `[heat]` extra automatically; clone the repository for that path.

**Scope of the first release.**

- Only the **potential** flux is computed. Convective and gauge-fixed flavours are deferred (Wieser *et al.*, lines 498–504 of the implementation spec); the file format records `flux_type: "potential"` so downstream tools know what was written.
- Only fully periodic 3D bulk solids are accepted. Each perpendicular cell height must strictly exceed `2 × num_message_passing_layers × r_cutoff + 2 Å` (26 Å for MACE-MP-0). Slabs, wires, molecules, and small primitive cells are rejected with a clear `ValueError` rather than silently returning a wrong flux. This restriction matches the typical Green-Kubo workflow for thermal conductivity, which already needs supercells of this size for convergence.
- `mace-unfolded`'s forward-mode autodiff path is currently incompatible with `mace-torch ≥ 0.3.10` (a `prepare_graph` call sets `requires_grad_(True)` inside `model.forward`, which `functorch.jvp` forbids). The default backend setting is reverse-mode, which is several times slower per call but works on a current MACE checkpoint. On a CUDA GPU, reverse-mode is fast enough for production; on CPU it can take many minutes per call.

**Combining with `ISIF = 3` (NPT).** When `ML_LHEAT = .TRUE.` is combined with NPT (`MDALGO = 3, ISIF = 3`), the cell volume drifts during the run. The `volume_A3` field in `ase_files/ML_HEAT.json` records the *initial* cell volume only; downstream tools should re-derive the time-resolved volume from the trajectory if needed. `vasp-mace` prints a `[note]` reminding the user when this combination is selected.

**Post-processing.** `vasp-mace` itself does not compute thermal conductivity. Pass the resulting `ML_HEAT` to [`sportran`](https://www.sciencedirect.com/science/article/abs/pii/S0010465522001898) for Green-Kubo / cepstral analysis. The `ase_files/ML_HEAT.json` sidecar carries the metadata `sportran` needs (timestep, units, temperature, volume, dtype).

```
IBRION         = 0
MDALGO         = 3
ISIF           = 2
LANGEVIN_GAMMA = 10.0 20.0
NSW            = 10000
TEBEG          = 300
POTIM          = 1.0
NBLOCK         = 100
ML_LHEAT       = .TRUE.
ML_HEAT_INTERVAL = 1
```

See `examples/example10_heat_flux/` for a runnable starting point on PbTe.

---

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | `$MACE_MODEL_PATH` | Path to MACE `.model` checkpoint |
| `--device` | `auto` | `auto` (→ `cuda` if available, else `mps`, else `cpu`), `cpu`, `cuda`, `mps` |
| `--dtype` | `auto` | `auto` (→ `float64` on CPU, `float32` on GPU/MPS), `float32`, or `float64` |
| `--optimizer` | `BFGS` | Fallback optimizer: `BFGS`, `FIRE`, or `LBFGS`. Overridden by `IBRION` if set in INCAR |


### GPU acceleration

`vasp-mace` supports GPU-accelerated inference via PyTorch. Every energy and force evaluation is a neural-network forward pass, so a GPU can deliver 10–100× speedup over CPU depending on system size.

**NVIDIA (CUDA)**

Install a CUDA-enabled PyTorch build before installing `vasp-mace` (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command for your CUDA version), then run:

```bash
vasp-mace --device cuda
```

Or simply run `vasp-mace` without `--device` — it will auto-detect CUDA.

**Apple Silicon (MPS)**

```bash
vasp-mace --device mps
```

**Precision**

GPU mode defaults to `float32`, which is faster and sufficient for MACE inference. Use `--dtype float64` to override if higher precision is needed.

**NEB memory note**: NEB runs load one calculator per image. With many images and a large model, this multiplies GPU memory usage. If memory is tight, reduce the number of images or use `--dtype float32` (which is already the default on GPU).

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
- **Phonon calculations**: IBRION = 5/6 produce `DYNMAT`, `OUTCAR` (eigenvalues + eigenvectors in VASP format), `OSZICAR`, `XDATCAR`, and `CONTCAR`. The `OUTCAR` phonon section is VASP-compatible (modes ordered high → low frequency; `f  =` for real, `f/i=` for imaginary). VASP's electronic-iteration lines in `OSZICAR` are omitted (replaced by a single energy summary line per configuration). `vasprun.xml` is not written for phonon runs.
- **Elastic constants**: when `ISIF ≥ 3` alongside `IBRION = 5/6`, the elastic tensor is appended to `OUTCAR`. The internal strain tensor (coupling atomic relaxation to macroscopic strain) is not computed.


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
| `ML_HEAT` | Per-step heat-flux vector (VASP-compatible format, `eV·Å·fs⁻¹`). Only written when `ML_LHEAT = .TRUE.` |
| `ase_files/mace.traj` | Full ASE binary trajectory |
| `ase_files/md.log` | ASE MD log (step, time, energy, temperature) |
| `ase_files/ML_HEAT.json` | Heat-flux metadata sidecar (timestep, write interval, target temperature, cell volume at MD start, backend, model, dtype, device). Only written when `ML_LHEAT = .TRUE.` |

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

### Phonon calculations (IBRION = 5 or 6)

See the [Phonon calculations](#phonon-calculations-ibrion--5-or-6) INCAR section above for a description of output files.

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
| `example08_PbTe_phonons/` | PbTe (rock salt, 8 atoms) | Phonon calculation with symmetry reduction (`IBRION = 6`, `NFREE = 2`). Requires `pip install phonopy` |
| `example09_MgO_elastic/` | MgO (rock salt, 8 atoms) | Phonons + elastic tensor (`IBRION = 6`, `ISIF = 3`): full 6×6 C_ij with Voigt/Reuss/Hill averages appended to OUTCAR |
| `example10_heat_flux/` | PbTe (rock salt, 4×4×4, 512 atoms) | NVT Langevin MD with `ML_LHEAT = .TRUE.`: writes a VASP-compatible `ML_HEAT` plus an `ML_HEAT.json` sidecar for downstream Green-Kubo analysis with [`sportran`](https://www.sciencedirect.com/science/article/abs/pii/S0010465522001898). Requires `pip install -e ".[heat]"` |

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

### example08 — PbTe phonon calculation (IBRION = 6)

8-atom PbTe conventional cell. Symmetry-reduced phonon calculation: only 2 irreducible displacements are needed (vs. 48 for a brute-force IBRION = 5 run). Writes `DYNMAT`, `OUTCAR` (with frequencies and eigenvectors), `OSZICAR`, `XDATCAR`, `CONTCAR`, and `ase_files/phonopy_params.yaml`.

```
ISIF   = 0
IBRION = 6
NFREE  = 2
POTIM  = 0.02
NSW    = 1
```

Run with:

```bash
vasp-mace --model /path/to/model
```

Requires `phonopy` for symmetry reduction: `pip install phonopy` or `pip install vasp-mace[phonons]`.

### example09 — MgO phonons + elastic constants (IBRION = 6, ISIF = 3)

8-atom MgO rock-salt conventional cell. Combines symmetry-reduced phonon calculation with elastic tensor computation. Phonopy reduces the phonon displacements from 48 to just 4, then 12 strain calculations (6 Voigt patterns × ±1%) yield the full 6×6 elastic tensor. The OUTCAR contains both phonon eigenvectors and the elastic tensor block with Voigt/Reuss/Hill averages.

```
IBRION = 6     # phonons via symmetry-reduced finite differences
NFREE  = 2     # central differences
POTIM  = 0.015 # displacement amplitude (Å)
ISIF   = 3     # also compute elastic tensor
```

Expected results for MgO with MACE-MP: cubic symmetry (C11=C22=C33 ≈ 247 GPa, C12=C13=C23 ≈ 88 GPa, C44=C55=C66 ≈ 118 GPa), K_Hill ≈ 141 GPa, G_Hill ≈ 101 GPa.

Requires `phonopy`: `pip install phonopy` or `pip install vasp-mace[phonons]`.

### example10 — PbTe heat-flux MD for Green-Kubo

4×4×4 PbTe rock-salt supercell at `a = 6.55 Å` (512 atoms; perpendicular heights 26.2 Å, just above the 26 Å bound enforced by the heat-flux backend for MACE-MP-0). NVT Langevin MD with `ML_LHEAT = .TRUE.` produces an `ML_HEAT` file in the run directory (one `NSTEP=… QXYZ= …` line per MD step, units `eV·Å·fs⁻¹`) and an `ase_files/ML_HEAT.json` sidecar with the metadata that downstream Green-Kubo tools need.

```
IBRION           = 0
MDALGO           = 3
ISIF             = 2
LANGEVIN_GAMMA   = 10.0 20.0
NSW              = 100      # production: 10000+
TEBEG            = 300
POTIM            = 1.0
NBLOCK           = 10
ML_LHEAT         = .TRUE.
ML_HEAT_INTERVAL = 1
```

Requires the heat-flux backend: clone the repo, then `pip install -e ".[heat]"`. Pass the resulting `ML_HEAT` to [`sportran`](https://www.sciencedirect.com/science/article/abs/pii/S0010465522001898) for Green-Kubo / cepstral analysis.

---

## Development

```bash
pip install -e ".[dev]"   # installs pytest, black, ruff, mypy
ruff check vasp_mace/     # lint
black vasp_mace/          # format
```

### Tests

The repository includes a standard-library `unittest` suite built from the
example inputs. The default run is lightweight and checks that all example
`INCAR` files parse and all example `POSCAR` files load correctly:

```bash
python scripts/run_tests.py
```

To run through a Conda environment without relying on `conda activate`:

```bash
python scripts/run_tests.py --conda-env mace_env
```

MACE-backed example smoke tests are opt-in because they need a model checkpoint
and are slower. They copy examples into temporary directories, reduce the run
length, execute `vasp-mace`, and verify the expected output files:

```bash
python scripts/run_tests.py --conda-env mace_env --with-examples --model "$MACE_MODEL_PATH"
python scripts/run_tests.py --conda-env mace_env --with-examples --example-set all --model "$MACE_MODEL_PATH"
```

---

## License and citation

MIT License © 2025 Ricardo Grau-Crespo.

If you use `vasp-mace` in your work, please cite:

**vasp-mace:**
- Grau-Crespo, R. *vasp-mace: a VASP-style workflow interface for MACE machine-learning interatomic potentials* (2025). Zenodo. https://doi.org/10.5281/zenodo.19479246

```bibtex
@software{graucrespo2025vaspmace,
  author  = {Grau-Crespo, Ricardo},
  title   = {vasp-mace: a VASP-style workflow interface for MACE machine-learning interatomic potentials},
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
