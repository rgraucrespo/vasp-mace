# -*- coding: utf-8 -*-
"""
Molecular dynamics driver for vasp-mace.

Supports:
  MDALGO=1 : NVE via VelocityVerlet, or NVT via Andersen thermostat
  MDALGO=2 : NVT via Nosé-Hoover thermostat
  MDALGO=3 : NVT via Langevin, or NPT via Langevin barostat

Outputs:
  XDATCAR              -- incremental trajectory in fractional coordinates
  ase_files/md.log     -- ASE MD log
  ase_files/mace.traj  -- ASE binary trajectory (every NBLOCK steps)
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import numpy as np
from ase import Atoms
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.io.trajectory import Trajectory
from ase.units import fs as ASE_FS, kB, GPa

from .types_ import IncarConfig, MDRecord
from .io_vasp import (
    write_xdatcar_header,
    append_xdatcar_frame,
    write_md_outcar_header,
    append_md_outcar_step,
)


class LangevinNPT(MolecularDynamics):
    """
    Simplified Langevin NPT (isothermal-isobaric) integrator.
    Based on Quigley and Probert (2004) algorithm.

    Parameters
    ----------
    friction : float
        Atomic friction coefficient in 1/ASE_time (not fs^-1)
    barostat_friction : float
        Lattice friction coefficient in 1/ASE_time (not fs^-1)
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        externalstress: float,
        friction: np.ndarray,
        barostat_friction: float,
        piston_mass: float,
        logfile: Optional[str] = None,
        trajectory: Optional[Any] = None,
        loginterval: int = 1,
    ) -> None:
        MolecularDynamics.__init__(
            self, atoms, timestep, trajectory, logfile, loginterval
        )
        self.temp_K = temperature_K
        # externalstress: scalar pressure (eV/A^3, compression positive)
        self.p_ext = externalstress
        self.friction = friction  # 1/ASE_time (passed as friction_fs / ASE_FS)
        self.gamma_L = (
            barostat_friction  # 1/ASE_time (passed as friction_L_fs / ASE_FS)
        )
        self.W = piston_mass  # eV * fs^2
        self.v_eps = 0.0  # strain rate
        self.rng = np.random.default_rng()

    def step(self, forces: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance the NPT trajectory by one integration step.

        Parameters
        ----------
        forces
            Optional precomputed force array in eV/Å. If omitted, forces are
            evaluated from the attached calculator.

        Returns
        -------
        numpy.ndarray
            Force array evaluated at the post-position-update geometry.
        """
        atoms = self.atoms
        dt = self.dt
        m = atoms.get_masses()[:, np.newaxis]
        v = atoms.get_velocities()
        if forces is None:
            f = atoms.get_forces(md=True)
        else:
            f = forces

        # Current internal pressure (eV/A^3)
        # stress is [xx, yy, zz, yz, xz, xy] tensile positive
        stress = atoms.get_stress(include_ideal_gas=True)
        p_int = -np.mean(stress[:3])  # compression positive

        # Stochastic terms
        sigma_at = np.sqrt(2.0 * self.friction * kB * self.temp_K / (m * dt))
        rnd_at = self.rng.standard_normal(v.shape) * sigma_at

        sigma_L = np.sqrt(2.0 * self.gamma_L * kB * self.temp_K / (self.W * dt))
        rnd_L = self.rng.standard_normal() * sigma_L

        # 1. Update velocities (half step)
        v += 0.5 * dt * (f / m - self.friction * v + rnd_at - self.v_eps * v)
        vol = atoms.get_volume()
        self.v_eps += (
            0.5
            * dt
            * ((p_int - self.p_ext) * vol / self.W - self.gamma_L * self.v_eps + rnd_L)
        )

        # 2. Update positions and cell (full step)
        pos = atoms.get_positions()
        pos += dt * (v + self.v_eps * pos)
        atoms.set_positions(pos)

        cell = atoms.get_cell()
        cell = cell * (1.0 + self.v_eps * dt)
        atoms.set_cell(cell, scale_atoms=False)
        vol_new = atoms.get_volume()  # updated volume for second barostat half-step

        # 3. Update forces and stress
        f = atoms.get_forces(md=True)
        stress = atoms.get_stress(include_ideal_gas=True)
        p_int = -np.mean(stress[:3])

        # 4. Update velocities (second half step)
        v += 0.5 * dt * (f / m - self.friction * v + rnd_at - self.v_eps * v)
        self.v_eps += (
            0.5
            * dt
            * (
                (p_int - self.p_ext) * vol_new / self.W
                - self.gamma_L * self.v_eps
                + rnd_L
            )
        )

        atoms.set_velocities(v)
        return f

def _per_atom_friction(atoms: Atoms, langevin_gamma: np.ndarray) -> np.ndarray:
    """Return per-atom friction (ps⁻¹, shape (N,)) from LANGEVIN_GAMMA.

    If langevin_gamma has one value, it is broadcast to all atoms.
    If it has multiple values, they are treated as per-species in the order
    species first appear in the atoms object (matching POSCAR convention).
    """
    N = len(atoms)
    if langevin_gamma.size == 1:
        return np.full(N, langevin_gamma[0])

    symbols = atoms.get_chemical_symbols()
    seen: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.append(s)

    if langevin_gamma.size != len(seen):
        print(
            f"[warn] LANGEVIN_GAMMA has {langevin_gamma.size} values but "
            f"{len(seen)} species ({', '.join(seen)}). Using first value for all atoms."
        )
        return np.full(N, langevin_gamma[0])

    gamma_map = {s: langevin_gamma[i] for i, s in enumerate(seen)}
    return np.array([gamma_map[s] for s in symbols])


ASE_OUT_DIR = "ase_files"


def _validate_ml_lheat_md_config(cfg: IncarConfig) -> None:
    """Validate ML_LHEAT constraints before creating MD output files."""
    if not cfg.ML_LHEAT:
        return
    if cfg.IVDW > 0:
        raise ValueError(
            "ML_LHEAT=.TRUE. is not supported with IVDW > 0. The current "
            "heat-flux backend computes only the MACE potential contribution, "
            "so adding D3 dispersion to the MD forces would make ML_HEAT "
            "inconsistent. Disable IVDW for heat-flux production, or run the "
            "dispersion-corrected MD without ML_LHEAT."
        )
    if not (
        cfg.MDALGO == 1
        and abs(float(cfg.ANDERSEN_PROB)) <= 1.0e-15
        and cfg.ISIF == 2
    ):
        raise ValueError(
            "ML_LHEAT=.TRUE. is restricted to fixed-cell NVE production MD. "
            "Use IBRION=0, MDALGO=1, ANDERSEN_PROB=0.0, and ISIF=2. "
            "Run NVT/NPT equilibration first with ML_LHEAT=.FALSE., then "
            "start a separate NVE heat-flux run."
        )


def _write_ml_heat_json(
    path: str,
    atoms: Atoms,
    cfg: IncarConfig,
    heat_calc: Any,
    model_path: str,
) -> None:
    """Write the ``ML_HEAT.json`` sidecar.

    Captures the metadata that downstream Green-Kubo/cepstral tools (e.g.
    ``sportran``) need in order to interpret the accompanying ``ML_HEAT``
    file: heat-flux units, integration timestep, write interval, target
    temperature, cell volume at MD start, and the backend/model/device that
    produced the flux.
    """
    payload = {
        "format": "vasp_mace_ml_heat/1",
        "heat_flux_units": "eV*Angstrom/fs",
        "ml_lheat": True,
        "flux_type": getattr(heat_calc, "flux_type", "potential"),
        "time_step_fs": float(cfg.POTIM),
        "write_interval": int(cfg.ML_HEAT_INTERVAL),
        "effective_heat_flux_dt_fs": float(cfg.POTIM) * int(cfg.ML_HEAT_INTERVAL),
        "temperature_K": float(cfg.TEBEG),
        "volume_A3": float(atoms.get_volume()),
        "backend": "mace_unfolded",
        "model_path": model_path,
        "dtype": getattr(heat_calc, "dtype", None),
        "device": getattr(heat_calc, "device", None),
        "shared_model": bool(getattr(heat_calc, "uses_shared_model", False)),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_md(
    atoms: Atoms,
    calc: Any,
    cfg: IncarConfig,
    model_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
) -> List[MDRecord]:
    """Run NVE, NVT, or NPT molecular dynamics.

    Parameters
    ----------
    atoms
        Structure to propagate. The object is modified in place.
    calc
        ASE-compatible calculator attached to ``atoms``.
    cfg
        Parsed INCAR configuration with ``IBRION=0`` and MD-related fields.
        ``MDALGO=1`` selects NVE or Andersen NVT, ``MDALGO=2`` selects
        Nosé-Hoover NVT, and ``MDALGO=3`` selects Langevin NVT or Langevin NPT
        when ``ISIF=3``.
    model_path
        Path to the MACE checkpoint. Only required when
        ``cfg.ML_LHEAT`` is true so the heat-flux backend can load the
        model directly (it operates below the ASE calculator interface).
    device, dtype
        Forwarded to the heat-flux backend so it follows the same
        device/dtype resolution as the main calculator.

    Returns
    -------
    list of MDRecord
        One summary record per MD step.

    Raises
    ------
    ValueError
        If ``cfg.MDALGO`` is not supported, or if ``cfg.ML_LHEAT`` is set
        but ``model_path`` is missing.
    """
    atoms.calc = calc
    N = len(atoms)
    _validate_ml_lheat_md_config(cfg)

    # Resolve TEEND: -1 means same as TEBEG (no ramp)
    T_start = cfg.TEBEG
    T_end = cfg.TEEND if cfg.TEEND >= 0.0 else cfg.TEBEG
    do_ramp = (cfg.MDALGO == 3) and (T_end != T_start)

    # Friction coefficients: build per-atom array (ps^-1), then convert to fs^-1
    gamma_per_atom = _per_atom_friction(atoms, cfg.LANGEVIN_GAMMA)  # shape (N,), ps^-1
    friction_per_atom_ase = gamma_per_atom / 1000.0 / ASE_FS  # ASE time units^-1
    friction_L_fs = cfg.LANGEVIN_GAMMA_L / 1000.0  # lattice friction: ps^-1 -> fs^-1

    os.makedirs(ASE_OUT_DIR, exist_ok=True)
    md_log = os.path.join(ASE_OUT_DIR, "md.log")
    traj_path = os.path.join(ASE_OUT_DIR, "mace.traj")

    # Initialise velocities from Maxwell-Boltzmann at TEBEG
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)

    # Create integrator
    if cfg.MDALGO == 1:
        # Andersen thermostat (NVT) or NVE if ANDERSEN_PROB=0
        if cfg.ANDERSEN_PROB > 0:
            dyn = Andersen(
                atoms,
                timestep=cfg.POTIM * ASE_FS,
                temperature_K=T_start,
                andersen_prob=cfg.ANDERSEN_PROB,
                logfile=md_log,
            )
        else:
            # Pure NVE: VelocityVerlet
            dyn = VelocityVerlet(atoms, timestep=cfg.POTIM * ASE_FS, logfile=md_log)
    elif cfg.MDALGO == 2:
        # Nose-Hoover NVT
        # VASP SMASS=0 corresponds to period ~ 40*POTIM
        # ASE tdamp = Period / 2pi
        if cfg.SMASS > 0:
            # User provided a mass-like parameter; here we take it as damping time in ps
            # to be somewhat consistent with other front-ends, but warn.
            tdamp = cfg.SMASS * 1000.0 * ASE_FS
        else:
            # VASP default for SMASS=0
            tdamp = (40.0 * cfg.POTIM * ASE_FS) / (2.0 * np.pi)

        dyn = NoseHooverChainNVT(
            atoms,
            timestep=cfg.POTIM * ASE_FS,
            temperature_K=T_start,
            tdamp=tdamp,
            logfile=md_log,
        )
    elif cfg.MDALGO == 3:
        if cfg.ISIF == 3:
            # NPT: Langevin
            pressure_GPa = cfg.PSTRESS * 0.1
            p_ext = pressure_GPa * GPa  # eV/A^3 (compression positive)

            # Piston mass (amu; numerically equals eV·ASE_time² in ASE units since 1 amu·Å² = 1 eV·ASE_time²).
            # Use PMASS from INCAR if set, otherwise default to N × 10000 amu.
            piston_mass = cfg.PMASS if cfg.PMASS > 0.0 else float(N) * (100.0**2)

            # LangevinNPT uses friction in 1/ASE_time; reshape to (N,1) for broadcasting
            friction_npt = friction_per_atom_ase.reshape(-1, 1)
            dyn = LangevinNPT(
                atoms,
                timestep=cfg.POTIM * ASE_FS,
                temperature_K=T_start,
                externalstress=p_ext,
                friction=friction_npt,
                barostat_friction=friction_L_fs
                / ASE_FS,  # lattice friction: fs^-1 -> ASE_time^-1
                piston_mass=piston_mass,
                logfile=md_log,
            )
        else:
            # NVT: Langevin; reshape to (N,1) so fr/masses stays (N,1) not (N,N)
            dyn = Langevin(
                atoms,
                timestep=cfg.POTIM * ASE_FS,
                temperature_K=T_start,
                friction=friction_per_atom_ase.reshape(-1, 1),
                logfile=md_log,
            )
    else:
        raise ValueError(
            f"MDALGO={cfg.MDALGO} is not supported. Use 1 (Andersen/NVE), 2 (Nose-Hoover), or 3 (Langevin)."
        )

    # Write XDATCAR header
    # For Langevin NPT (MDALGO=3, ISIF=3), the cell changes so the header is updated per frame.
    # For all other MD modes (NVE, NVT), the cell is fixed — write the header once.
    cell_relaxing = cfg.MDALGO == 3 and cfg.ISIF == 3
    if not cell_relaxing:
        write_xdatcar_header("XDATCAR", atoms)
    else:
        with open("XDATCAR", "w"):
            pass

    write_md_outcar_header("OUTCAR", atoms, getattr(cfg, "raw", {}))

    # Open trajectory file; write at each recorded step
    traj = Trajectory(traj_path, "w", atoms)

    # Heat-flux writer + calculator (opt-in via ML_LHEAT). Built before the
    # main loop so that a model-load or precondition failure is reported
    # immediately, not after several MD steps.
    heat_writer = None
    heat_calc = None
    if cfg.ML_LHEAT:
        if not model_path:
            raise ValueError(
                "ML_LHEAT=.TRUE. requires a model checkpoint path to load the "
                "heat-flux backend; pass it via the CLI dispatcher"
            )
        from .heat import MLHeatWriter, make_heat_flux_calculator, validate_3d_bulk_cell

        # Pin the heat-flux backend to float64 even when the main calculator
        # runs at float32 on a GPU: mace-unfolded has a known float32
        # dtype-mismatch bug (positions stay float64 inside `prepare_graph`
        # even after `set_default_dtype("float32")`), so float32 raises
        # `expected scalar type Float but found Double` mid-run. Float64 is
        # also what the stage-2 regression test relies on.
        if dtype not in ("auto", "float64"):
            print(
                f"[note] ML_LHEAT: heat-flux backend forced to float64 "
                f"(main calculator stays at --dtype={dtype})."
            )
        raw_torch_model = None
        calc_models = getattr(calc, "models", None)
        if calc_models:
            raw_torch_model = calc_models[0]

        heat_settings = {"device": device, "dtype": "float64"}
        if raw_torch_model is not None:
            heat_settings["torch_model"] = raw_torch_model
        heat_calc = make_heat_flux_calculator(model_path, settings=heat_settings)
        validate_3d_bulk_cell(
            atoms,
            heat_calc.r_cutoff,
            heat_calc.num_message_passing,
            heat_calc.cell_size_margin,
        )
        heat_writer = MLHeatWriter("ML_HEAT")
        _write_ml_heat_json(
            os.path.join(ASE_OUT_DIR, "ML_HEAT.json"),
            atoms=atoms,
            cfg=cfg,
            heat_calc=heat_calc,
            model_path=model_path,
        )
        print(
            f"[info] ML_LHEAT enabled: writing ML_HEAT every "
            f"{cfg.ML_HEAT_INTERVAL} step(s) using {heat_calc.__class__.__name__} "
            f"(device={heat_calc.device}, dtype={heat_calc.dtype})."
        )

    records: list[MDRecord] = []

    try:
        for step in range(1, cfg.NSW + 1):
            # Temperature ramp for Langevin (linear interpolation)
            if do_ramp and cfg.NSW > 1:
                frac = (step - 1) / (cfg.NSW - 1)
                T_target = T_start + frac * (T_end - T_start)
                if hasattr(dyn, "set_temperature"):
                    dyn.set_temperature(temperature_K=T_target)
                elif hasattr(dyn, "temp_K"):
                    dyn.temp_K = T_target

            dyn.run(1)  # single MD step

            E_pot = atoms.get_potential_energy()
            E_kin = atoms.get_kinetic_energy()
            # Instantaneous temperature from kinetic energy: T = 2*Ekin / (3*N*kB)
            T_inst = 2.0 * E_kin / (3.0 * N * kB)
            E_tot = E_pot + E_kin

            rec = MDRecord(
                n=step, energy_pot=E_pot, energy_kin=E_kin, temperature=T_inst
            )
            records.append(rec)
            append_md_outcar_step("OUTCAR", atoms, step, E_pot, E_kin, T_inst)

            # Stress/Pressure info for stdout if ISIF=3
            stress_info = ""
            if cell_relaxing:
                try:
                    stress = atoms.get_stress(include_ideal_gas=True, voigt=True)
                    p_int = -np.mean(stress[:3])
                    stress_info = f" | P={p_int/GPa*10.0:.2f} kBar"
                except Exception:
                    pass

            print(
                f"[step {step}] T={T_inst:.2f} K | "
                f"Epot={E_pot:.6f} eV | "
                f"Ekin={E_kin:.6f} eV | "
                f"Etot={E_tot:.6f} eV{stress_info}"
            )

            if step % cfg.NBLOCK == 0:
                append_xdatcar_frame(
                    "XDATCAR", atoms, step, update_header=cell_relaxing
                )
                traj.write()

            # Heat flux: compute and append after the integrator step so the
            # velocities reflect the post-step state, matching VASP's per-step
            # ML_HEAT cadence.
            if heat_writer is not None and step % cfg.ML_HEAT_INTERVAL == 0:
                qxyz = heat_calc.compute(atoms, atoms.get_velocities())
                heat_writer.write(step, qxyz)
    finally:
        traj.close()
        if heat_writer is not None:
            heat_writer.close()
        if heat_calc is not None and hasattr(heat_calc, "close"):
            heat_calc.close()

    return records
