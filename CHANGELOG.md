# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Example-based `unittest` suite covering example `INCAR` parsing, `POSCAR` loading, NEB image layouts, and opt-in MACE-backed smoke runs.
- `scripts/run_tests.py` helper for running tests directly or through `conda run -n <env>`, including quick/all example smoke-test modes.
- Prominent README and repository-level `NOTICE.md` disclaimer clarifying that `vasp-mace` is independent from VASP Software GmbH and does not include or distribute licensed VASP components.

### Changed
- Bumped the minimum supported `mace-torch` dependency from `0.3.6` to `0.3.15`.
- Expanded PEP 484 type annotations and NumPy-style docstrings across public package APIs and shared dataclasses.

## [2.1.0] - 2026-05-08

### Added
- Elastic tensor calculation triggered by `ISIF ≥ 3` alongside `IBRION = 5/6`, matching VASP behaviour.
  - Applies 6 Voigt strain patterns ±1% (12 single-point calculations) and finite-differences stress → C_ij.
  - Full 6×6 elastic tensor in VASP OUTCAR format (kBar, XX YY ZZ XY YZ ZX column order) appended to the existing `OUTCAR`.
  - Voigt, Reuss, and Hill polycrystalline averages: bulk modulus K, shear modulus G, Young's modulus E, Poisson ratio ν. Formulas are valid for all crystal systems (cubic to triclinic).
  - Human-readable stdout summary in GPa (ASE Voigt ordering: xx yy zz yz xz xy).
  - Informational note printed when `PSTRESS > 0` is set alongside `IBRION = 5/6`, clarifying that pressure must be incorporated during the geometry relaxation step.
- `example09_MgO_elastic`: Γ-point phonons + elastic tensor for the 8-atom MgO conventional cell (`IBRION = 6`, `ISIF = 3`).

## [2.0.0] - 2026-04-22

### Added
- Phonon calculations at the Γ-point via finite-difference force constants (`IBRION = 5` and `IBRION = 6`).
- `IBRION = 6` symmetry-reduced displacements via [phonopy](https://phonopy.github.io/phonopy/), significantly reducing the number of required force evaluations.
- `POTIM` tag in phonon mode sets the finite-difference displacement amplitude (default 0.015 Å).
- `NFREE` tag selects central (`2`) or forward (`1`) finite differences.
- `DYNMAT` output file in VASP format (half-force central-difference matrix).
- Phonon eigenvalues and eigenvectors written to `OUTCAR` in VASP-compatible format (modes ordered high → low frequency; `f  =` for real, `f/i=` for imaginary modes).
- `OSZICAR` records one energy line per displaced configuration.
- `XDATCAR` contains the equilibrium structure followed by all displaced configurations.
- `ase_files/force_constants.npy` saves the full force-constant tensor (shape N×3×N×3, eV/Å²).
- `ase_files/phonopy_params.yaml` saves phonopy parameters for `IBRION = 6` runs.
- `phonopy` optional dependency: `pip install vasp-mace[phonons]`.
- `example08_PbTe_phonons`: symmetry-reduced Γ-point phonons for the 8-atom PbTe conventional cell.
- CUDA device support: `--device cuda` is now a valid CLI choice for NVIDIA GPUs.
- Auto-detection of available accelerator in `--device auto` mode: prefers CUDA, then MPS, then falls back to CPU.
- Graceful fallback to CPU/float64 with a printed warning if CUDA or MPS model loading fails (e.g. current e3nn/MPS float64 incompatibility on Apple Silicon).
- GPU acceleration section in README covering CUDA setup, MPS, precision defaults, and NEB memory notes.

### Changed
- `ISIF = 0` and `ISIF = 1` are now coerced to `ISIF = 2` with an informational message, matching VASP's effective behaviour (they differ only in stress-tensor completeness, which MACE always computes in full).

## [1.4.1] - 2026-04-20

### Added
- `pyproject.toml` with full PyPI metadata; package is now installable via `pip install vasp-mace`.
- GitHub Actions workflow for automated PyPI publishing on tagged releases.
- Inline comments in all example `INCAR` files explaining each tag.

### Changed
- README installation section updated to reflect `pip install vasp-mace` as the primary install path.

## [1.4.0] - 2026-04-20

### Added
- Per-species Langevin friction: `LANGEVIN_GAMMA` now accepts multiple space-separated values assigned in POSCAR species order (e.g. `LANGEVIN_GAMMA = 10.0 20.0`).
- `PMASS` tag for explicit barostat piston mass (amu) in Langevin NPT (`MDALGO = 3`, `ISIF = 3`); defaults to `N × 10000` amu.
- `example07_PbTe_MD`: 512-atom PbTe sequential NVT → NPT Langevin MD with per-species friction and explicit `PMASS`, driven by a `run.sh` script.

## [1.3.0] - 2026-04-13

### Added
- Nudged Elastic Band (NEB) mode, triggered by `IMAGES > 0` in INCAR.
- Climbing-image NEB (`LCLIMB = .TRUE.`) following the VTST convention.
- `SPRING` tag for NEB spring constant (VASP sign convention: negative values indicate NEB).
- Automatic IDPP interpolation of intermediate images when they are absent from the numbered subdirectories.
- NEB output per image directory: `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`.
- Shared NEB output in `ase_files/`: `neb_opt.log` and `mace.traj` (one frame per image).
- `example05_Si_NEB`: CI-NEB for Si self-interstitial migration (4 images).
- `example06_Pt_NEB`: CI-NEB for collective Pt adatom jump on fcc-Pt(001) (3 images).

### Changed
- NEB always uses ASE `MDMin` optimizer regardless of `IBRION`, to avoid divergence from non-conservative spring forces.
- Documentation expanded with NEB section, VTST convention notes, and updated "Differences with respect to VASP" table.

## [1.2.1] - 2026-04-12

### Fixed
- `XDATCAR` now repeats the cell header at every frame for cell-relaxing runs (`ISIF = 3/4/7/8`), making trajectories readable by VASP-compatible analysis tools.

## [1.2.0] - 2026-04-09

### Added
- Molecular dynamics mode (`IBRION = 0`) with full `MDALGO` support:
  - `MDALGO = 1`: NVE (VelocityVerlet, `ANDERSEN_PROB = 0`) and NVT Andersen thermostat (`ANDERSEN_PROB > 0`).
  - `MDALGO = 2`: NVT Nosé-Hoover thermostat; `SMASS > 0` sets damping time (ps); default maps to 40-step oscillation period matching VASP.
  - `MDALGO = 3`: NVT Langevin (`ISIF = 2`) and NPT Langevin barostat (`ISIF = 3`) using the Quigley–Probert (2004) stochastic algorithm.
- `TEBEG` and `TEEND` tags for MD temperature (supports linear ramp).
- `NBLOCK` tag for XDATCAR frame and trajectory write frequency.
- `LANGEVIN_GAMMA` and `LANGEVIN_GAMMA_L` tags for atomic and lattice Langevin friction.
- `SMASS` tag as fallback for Langevin friction if `LANGEVIN_GAMMA` is absent.
- `ANDERSEN_PROB` tag for Andersen collision probability.
- NPT XDATCAR writes an updated lattice header for every recorded frame.
- DFT-D3 empirical dispersion correction via `IVDW` tag (zero-damping, Becke-Johnson, and ATM three-body variants), implemented through the `dftd4` library.
- `ISIF = 4`: relax positions and cell shape at constant volume (ExpCellFilter).
- `ISIF = 7`: relax volume only with positions fixed.
- `ISIF = 8`: relax positions and volume at fixed cell shape.
- `PSTRESS` tag for target hydrostatic pressure in kBar (`ISIF = 3`).
- `XDATCAR` output for relaxation runs (one frame per ionic step).
- `CITATION.cff` with references for MACE potentials and VASP.
- `example03_CsPbI3_MA_MD`: NVT Nosé-Hoover MD on a 327-atom perovskite supercell.
- `example04_PbTe_pressure`: variable-cell relaxation under 15 kBar target pressure.

### Changed
- Output writers fully rewritten to match VASP file formats (`OUTCAR`, `OSZICAR`, `CONTCAR`, `vasprun.xml`).
- Standard output now reports potential energy and energy change (`dE`) at every ionic/MD step.
- Stress values in terminal output reported in kBar to match VASP convention.
- Simplified stress summary: prints `max|σ|` or `max|σ−pI|` (only when `PSTRESS > 0`) in a single line.
- Single-point (`NSW = 0`) prints a concise summary: energy, Fmax, and max stress.

### Fixed
- `NameError` in `vasp_mace/types_.py` caused by missing `import numpy as np`.

## [0.1.0] - 2025-10-31

### Added
- Single-point energy, force, and stress evaluation (`NSW = 0`).
- Geometry relaxation of atomic positions and/or unit cell (`ISIF = 2/3`).
- BFGS, FIRE, and LBFGS optimizers selectable via `--optimizer` CLI flag.
- Force-based (`EDIFFG < 0`) and energy-based (`EDIFFG > 0`) convergence criteria.
- Selective dynamics: per-atom coordinate fixing from POSCAR, preserved in CONTCAR.
- VASP-compatible outputs: `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`.
- ASE trajectory (`ase_files/mace.traj`) and optimizer log (`ase_files/opt.log`).
- `--device auto|cpu|mps` and `--dtype auto|float32|float64` CLI flags.
- `MACE_MODEL_PATH` environment variable for model checkpoint path.
- `example01_MgO`: variable-cell relaxation of MgO rock-salt structure.
- `example02_hBN_D3-dispersion`: variable-cell relaxation of h-BN with D3(BJ) dispersion.

[Unreleased]: https://github.com/rgraucrespo/vasp-mace/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/rgraucrespo/vasp-mace/compare/v1.4.1...v2.0.0
[1.4.1]: https://github.com/rgraucrespo/vasp-mace/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/rgraucrespo/vasp-mace/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/rgraucrespo/vasp-mace/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/rgraucrespo/vasp-mace/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/rgraucrespo/vasp-mace/compare/v0.1.0...v1.2.0
[0.1.0]: https://github.com/rgraucrespo/vasp-mace/releases/tag/v0.1.0
