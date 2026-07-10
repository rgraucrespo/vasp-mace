# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.9.0] - 2026-07-10

### Added
- Elastic-property output now includes lower and upper Hashin-Shtrikman shear
  modulus bounds, their midpoint, and the corresponding Poisson ratios in both
  stdout and `OUTCAR`, in a single polycrystalline-moduli table shared with the
  Voigt, Reuss, and Hill approximations.

## [2.8.0] - 2026-06-30

### Fixed
- Corrected the `IVDW=13` INCAR validation: it is DFT-D4 (VASP >= 6.2), not
  "D3 + ATM three-body" as previously labelled (`IVDW=13` now selects D4; see
  Added). `IVDW=14` is treated as an ordinary unsupported value.
- Pinned the development Black dependency so CI formatting checks use the same
  formatter version as the repository.

### Added
- **Implicit solvation** via `LSOL = .TRUE.`, a density-free surrogate (not
  VASPsol's Poisson-Boltzmann model), added on top of MACE. Two terms:
  - **Nonpolar/cavitation** `E = TAU * SASA` — PBC-aware Shrake-Rupley
    solvent-accessible surface area; `TAU` is the surface tension in meV/Å²
    (VASPsol convention, default 0.525).
  - **Polar/electrostatic** Generalized-Born (OBC effective Born radii) using
    the shared EEQ charges (`vasp_mace.charges`) and `EB_K` (solvent
    dielectric, default 78.4 = water). Cross terms use the minimum-image
    distance, not a periodic lattice sum (which is an ill-defined Madelung sum
    for ionic slabs). `EB_K = 1` disables the polar term.

  Slab/cluster/molecule only: a dense 3D-periodic cell with no vacuum is
  rejected. Forces are finite differences of the solvation energy (so the GB
  charge-position coupling is automatic); stress is not yet included (zero,
  with a warning). Rejected together with `ML_LHEAT` for heat-flux
  consistency. Validated against the VASPsol PbS(100) example (see
  `examples/example12_PbS_100_solvation/`): solvation shift −0.18 eV vs the
  VASPsol reference −0.08 eV.
- `examples/example12_PbS_100_solvation/`: PbS(100) slab implicit-solvation
  single-point (`LSOL`, `EB_K = 80`), after the VASPsol `PbS_100` example.
- **DFT-D4 dispersion correction** via `IVDW=13` (xc=PBE), backed by the
  optional `dftd4` package. Periodic-capable, so it works for 3D bulk systems
  as well as molecules/slabs, and returns energy, forces, and stress. Added on
  top of the MACE calculator with `SumCalculator`, mirroring the D3 path.
- `vasp_mace.charges.eeq_charges`: shared EEQ partial-charge engine (from the
  same `dftd4` backend). Neutral systems only for now; this is the charge
  source that will also drive the planned implicit-solvation term.
- `requirements/dftd4.txt`: optional DFT-D4 dispersion backend (`dftd4`).
- `examples/example11_hBN_D4-dispersion/`: hexagonal h-BN variable-cell
  relaxation with DFT-D4 (`IVDW=13`), mirroring the D3 example02. Reproduces
  the expected geometry (a ≈ 2.51 Å, B-N ≈ 1.45 Å, interlayer ≈ 3.4 Å) and
  EEQ charges (B ≈ +0.20, N ≈ -0.20).

### Changed
- Heat-flux backend: when `forward=True`, the `mace-unfolded` call is now
  wrapped so `mace-torch`'s in-place `requires_grad_` inside `functorch.jvp`
  no longer raises. This clears the first of two upstream blockers on
  forward-mode autodiff; the default stays `forward=False` (reverse mode),
  which is unchanged, because mace-unfolded's forward path still crashes
  serialising its `None` `sigma_*` terms. Verified numerically identical to
  reverse mode (to machine precision) once that second bug is patched.

## [2.7.0] - 2026-06-20

### Added
- `RANDOM_SEED` INCAR support for reproducible MD velocity initialization and
  stochastic Andersen/Langevin/NPT random terms.
- MD runs now write a VASP-style `OSZICAR` with per-step temperature, total
  energy, potential energy, and kinetic energy.
- GitHub Actions CI for Python 3.9-3.12 that checks Black formatting and runs
  the default test suite on pushes and pull requests.

### Changed
- Moved optional heat-backend dependency pins from root-level
  `requirements-heat.txt` to `requirements/heat.txt`.

### Fixed
- INCAR parsing now accepts Fortran-style `D`/`d` exponents for supported
  numeric tags and raises a clear `ValueError` for malformed supported tag
  values instead of silently falling back to defaults.
- NEB now rejects `IMAGES > 0` with `NSW < 1` at parse/driver entry so runs do
  not fail later while writing empty per-image outputs.
- MD temperature reporting now uses ASE's constrained degrees-of-freedom count
  instead of assuming `3 × N` mobile Cartesian components.
- `run_md` now preserves pre-existing nonzero velocities on programmatic
  `Atoms` inputs instead of always replacing them with a Maxwell-Boltzmann draw.
- `read_poscar(..., apply_selective_dynamics=False)` now actually clears
  Selective Dynamics constraints created by ASE, and the fallback
  `FixCartesian` construction for raw Selective Dynamics arrays uses the
  correct ASE API.
- The NEB module documentation now describes `ase_files/mace.traj` as the final
  NEB band, matching the implementation and README.

## [2.6.1] - 2026-06-17

### Changed
- The Nosé-Hoover chain `TEEND` ramp now rescales the existing thermostat
  masses by the temperature ratio (`Q ∝ kT`) instead of reproducing ASE's
  internal mass-matrix formula, so it stays correct across ASE versions that
  change the degrees-of-freedom count or `tdamp` convention.
- Capped the ASE dependency at `<4` so a major ASE release is a deliberate
  opt-in rather than silently exercising untested thermostat internals.

## [2.6.0] - 2026-06-16

### Changed
- Formatted the remaining Python files so `black --check vasp_mace tests scripts`
  passes.
- Ignored local-only heat-flux tarball artifacts and session handoff prompt
  files so they no longer appear as untracked repository changes.

### Fixed
- Raised the ASE dependency floor to `3.24.0` because `vasp_mace.md`
  imports `ase.md.nose_hoover_chain.NoseHooverChainNVT`, which is absent in
  ASE `3.22.x` and `3.23.x`.
- Synced `vasp_mace.__version__` with the `pyproject.toml` package version and
  added a packaging regression test for version drift.
- `TEEND` temperature ramps now update Andersen, Nosé-Hoover, and Langevin
  thermostat targets instead of applying only to Langevin runs; pure NVE now
  warns because there is no thermostat target to ramp.
- Symmetry-reduced phonon runs now save `ase_files/force_constants.npy` with
  the same `(N, 3, N, 3)` layout as brute-force phonon runs, matching the
  documented `C[i, alpha, j, beta]` convention.
- NEB now rejects non-negative `SPRING` values at parse/driver entry instead
  of silently converting them with `abs(SPRING)`, matching the documented
  negative-`SPRING` VASP NEB convention.

## [2.5.1] - 2026-05-26

### Documentation
- Installation section now lists `pip install "vasp-mace[phonons]"` as an optional step for `IBRION = 6` (symmetry-reduced phonons via phonopy), matching the existing `torch-dftd` callout pattern.

## [2.5.0] - 2026-05-26

### Fixed
- `IVDW > 0` was a silent no-op: `load_calc` passed `dispersion=True` into `MACECalculator(...)`, but mace-torch 0.3.x swallows that kwarg via `Calculator.__init__`'s catch-all `**kwargs` and never applies D3. Energies and forces were pure MACE regardless of `IVDW`. `load_calc` now constructs a `torch_dftd.TorchDFTD3Calculator` (xc=PBE, cutoff=40 Bohr) and returns `SumCalculator([mace_calc, d3_calc])` whenever `IVDW > 0`, mirroring how `mace_mp()` wires dispersion internally. The damping flavor follows the IVDW value: `11` → zero damping, `12` → Becke-Johnson. Reported against a MoS₂ test where `IVDW = 12` produced bit-identical results to `IVDW = 0`. Verified on the h-BN example with `mace-mp-0b3-medium`: `ΔE ≈ −561 meV` (4-atom cell) between `IVDW=0` and `IVDW=12`, attractive as expected for a layered vdW solid.

### Changed
- `IVDW` validation is now stricter: ATM three-body variants (`13`/`14`) are explicitly rejected at INCAR parse time with a message pointing to `11`/`12`. Only `0`, `11`, and `12` are accepted. The D3 xc functional is hardcoded to PBE.
- `load_calc` signature: the `dispersion: bool` parameter has been replaced by `ivdw: int`. Call sites in `cli.py` and `neb.py` were updated; `run_neb` now takes `ivdw=` instead of `dispersion=`.
- `tests/test_examples.py`: the `example02_dispersion_hbn` smoke case now gates on `torch_dftd` (the real dependency) instead of `dftd4`.

### Added
- `tests/test_ivdw_dispersion.py`: regression coverage that proves D3 is now actually being summed in (energies and forces shift on a Cu pair) and that the IVDW rejection paths fire as documented.

## [2.4.0] - 2026-05-11

### Changed
- `ML_LHEAT` is now restricted to fixed-cell NVE production MD (`IBRION = 0`, `MDALGO = 1`, `ANDERSEN_PROB = 0.0`, `ISIF = 2`). Thermostatted/barostatted MD should be used only for equilibration with `ML_LHEAT = .FALSE.`.
- `ML_LHEAT` now rejects `IVDW > 0` because the current heat-flux backend computes only the MACE potential contribution; allowing D3-corrected MD would produce an inconsistent MACE-only heat flux.
- `examples/example10_heat_flux/INCAR` is now an NVE Green-Kubo production input, with a separate `INCAR_NVT_EQUIL` for optional Langevin equilibration before the heat-flux run.
- `ML_LHEAT` now reuses the main MD calculator's raw MACE torch model when available, avoiding a second checkpoint load and duplicate model copy in memory. The heat-flux backend also suppresses upstream `POSCAR_unfolding` artefact writes and keeps one scratch directory as a fallback guard instead of creating a temporary directory for every MD step.
- The PyPI package no longer exposes a `[heat]` extra because its backend dependencies are GitHub-only. Optional ML_HEAT dependencies are now documented in `requirements-heat.txt`, and the publish workflow rejects direct URL dependencies in built metadata.

### Fixed
- `make_heat_flux_calculator`: removed stale `pbc` entry from the `settings` docstring. `MACEUnfoldedHeatFluxCalculator` does not accept a `pbc` argument (fully-periodic 3D is baked in); passing `pbc` in `settings` would have raised `TypeError` at runtime.
- `run_md`: `validate_3d_bulk_cell` is now called immediately after the heat-flux backend is constructed, before the MD loop starts. Previously the validation only fired at the first `compute()` call — after one full MD step had already executed.
- `MLHeatWriter._fh`: corrected type annotation from `Optional[object]` to `Optional[IO[str]]`.

### Repository
- Added `tests/data/` to `.gitignore`; the `mace_unfolded_reference.npz` regression file is model-checkpoint-specific and must be generated locally (see `tests/test_mace_unfolded_regression.py`).

## [2.3.1] - 2026-05-10

### Changed
- Moved the `ML_HEAT.json` sidecar from the run-directory root into `ase_files/ML_HEAT.json`, matching the project convention that `ase_files/` holds outputs VASP itself does not produce (`md.log`, `mace.traj`, `force_constants.npy`, …). `ML_HEAT` continues to be written at the run-directory root because it is the file VASP's `ML_LHEAT` workflow generates and downstream tools (e.g. `sportran`) expect it there.

### Documentation
- Added `ML_LHEAT` heat flux to the top-of-README Features list.
- README "Molecular dynamics" output-files table now lists `ML_HEAT` and `ase_files/ML_HEAT.json` explicitly.

## [2.3.0] - 2026-05-10

### Added
- VASP-style `ML_LHEAT` INCAR keyword (default `.FALSE.`) and the `ML_HEAT_INTERVAL` vasp-mace extension (default `1`); both parsed into `IncarConfig`. When `ML_LHEAT = .TRUE.` is set on an MD run (`IBRION = 0`), `vasp-mace` writes a VASP-compatible `ML_HEAT` file (one `NSTEP=… QXYZ= …` line per `ML_HEAT_INTERVAL` step, units `eV·Å·fs⁻¹`) plus an `ML_HEAT.json` sidecar with timestep, write interval, target temperature, cell volume at MD start, backend, model path, dtype, and device. Setting `ML_LHEAT = .TRUE.` outside MD is ignored with a clear `[warn]`; combining it with `ISIF = 3` (NPT) prints a `[note]` because the recorded volume becomes representative only.
- New `vasp_mace.heat` subpackage providing `MLHeatWriter`, `read_ml_heat`, and `write_ml_heat` for the VASP-compatible `ML_HEAT` file format. The reader tolerates Fortran-style `D` exponents so it can parse files produced by either VASP or vasp-mace.
- `vasp_mace.heat.heat_flux.HeatFluxCalculator` abstract interface and `make_heat_flux_calculator` factory; `MACEUnfoldedHeatFluxCalculator` adapter wrapping [`mace-unfolded`](https://github.com/pulgon-project/mace-unfolded) (Wieser *et al.*, *J. Chem. Theory Comput.* **22**, 513 (2026)) for the autograd-based unfolded-cell **potential** heat flux. Convective and gauge-fixed flavours are deferred. `mace_unfolded` ships as an opt-in extra (`pip install -e ".[heat]"`); because `mace_unfolded` and its logging dependency [`comms`](https://github.com/sirmarcel/comms) are both GitHub-only, the `[heat]` extra installs them via direct git URLs. Default install (without `[heat]`) is unchanged.
- Scope-restriction precondition: the first release deliberately supports only fully periodic 3D bulk solids in supercells where every perpendicular cell height exceeds `2 × num_message_passing_layers × r_cutoff + cell_size_margin` (default 2 Å, i.e. 26 Å for MACE-MP-0). Slabs, wires, molecules, and small primitive cells are rejected with a clear `ValueError` rather than silently returning a wrong flux. Exposed as `vasp_mace.heat.validate_3d_bulk_cell`.
- `examples/example10_heat_flux/`: 512-atom 4×4×4 PbTe NVT Langevin MD with `ML_LHEAT = .TRUE.` as a runnable starting point for Green-Kubo workflows. The cell is sized at `a = 6.55 Å` so the 26.2 Å perpendicular heights satisfy the precondition.
- README "Heat flux (ML_HEAT)" section documenting the VASP file-format compatibility, the `[heat]` extra install, the 3D-bulk-only scope, the potential-flux-only restriction, the `ISIF = 3` caveat, and the `sportran` post-processing path.
- `tests/test_ml_heat_io.py`: round-trip tests for the writer/reader, parsing of the verbatim VASP example block, and INCAR parsing of `ML_LHEAT`/`ML_HEAT_INTERVAL`.
- Opt-in `tests/test_mace_heat_flux_smoke.py` and `tests/test_mace_unfolded_regression.py` (gated on `RUN_VASP_MACE_EXAMPLES=1`, `MACE_MODEL_PATH`, and the `[heat]` extra) verifying that the heat-flux backend produces a finite 3-vector and matches a saved reference flux on a 64-atom PbTe fixture.
- `tests/test_heat_flux_cell_check.py`: default-suite coverage for `validate_3d_bulk_cell` (height math, partial-pbc rejection, triclinic cells, negative-margin escape hatch).

### Changed
- `vasp_mace.md.run_md` gained optional `model_path`, `device`, and `dtype` arguments so the heat-flux backend can load the MACE checkpoint directly (it operates below the ASE calculator interface to access per-atom energies).
- Updated citation title to "vasp-mace: a VASP-style workflow interface for MACE machine-learning interatomic potentials".

### Notes for maintainers
- Direct-URL refs in the `[heat]` extra (`mace_unfolded` and `comms` from GitHub) cannot be uploaded to PyPI. Strip the `[heat]` extra from `pyproject.toml` before `python -m build` + PyPI upload, or vendor the deps. Local `pip install -e ".[heat]"` works as-is.
- `mace-unfolded`'s forward-mode (functorch JVP) path is currently broken with `mace-torch ≥ 0.3.10`. `MACEUnfoldedHeatFluxCalculator` defaults to reverse-mode autodiff; once upstream restores forward-mode compatibility, flipping the default is several times faster per call.

## [2.2.0] - 2026-05-09

### Added
- `LICENSE` file (MIT) and updated `pyproject.toml` to reference it; `NOTICE.md` restructured to lead with the licence statement.
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

[Unreleased]: https://github.com/rgraucrespo/vasp-mace/compare/v2.9.0...HEAD
[2.9.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.8.0...v2.9.0
[2.8.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/rgraucrespo/vasp-mace/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.5.1...v2.6.0
[2.5.1]: https://github.com/rgraucrespo/vasp-mace/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.3.1...v2.4.0
[2.3.1]: https://github.com/rgraucrespo/vasp-mace/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/rgraucrespo/vasp-mace/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/rgraucrespo/vasp-mace/compare/v1.4.1...v2.0.0
[1.4.1]: https://github.com/rgraucrespo/vasp-mace/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/rgraucrespo/vasp-mace/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/rgraucrespo/vasp-mace/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/rgraucrespo/vasp-mace/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/rgraucrespo/vasp-mace/compare/v0.1.0...v1.2.0
[0.1.0]: https://github.com/rgraucrespo/vasp-mace/releases/tag/v0.1.0
