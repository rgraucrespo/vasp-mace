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

import os

import numpy as np
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.io.trajectory import Trajectory
from ase.units import fs as ASE_FS, kB, GPa


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
    def __init__(self, atoms, timestep, temperature_K, externalstress,
                 friction, barostat_friction, piston_mass,
                 logfile=None, trajectory=None, loginterval=1):
        MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile, loginterval)
        self.temp_K = temperature_K
        # externalstress: scalar pressure (eV/A^3, compression positive)
        self.p_ext = externalstress
        self.friction = friction  # 1/ASE_time (passed as friction_fs / ASE_FS)
        self.gamma_L = barostat_friction  # 1/ASE_time (passed as friction_L_fs / ASE_FS)
        self.W = piston_mass  # eV * fs^2
        self.v_eps = 0.0  # strain rate
        self.rng = np.random.default_rng()

    def step(self, forces=None):
        atoms = self.atoms
        dt = self.dt
        N = len(atoms)
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
        v += 0.5 * dt * (f/m - self.friction * v + rnd_at - self.v_eps * v)
        vol = atoms.get_volume()
        self.v_eps += 0.5 * dt * ((p_int - self.p_ext) * vol / self.W - self.gamma_L * self.v_eps + rnd_L)

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
        v += 0.5 * dt * (f/m - self.friction * v + rnd_at - self.v_eps * v)
        self.v_eps += 0.5 * dt * ((p_int - self.p_ext) * vol_new / self.W - self.gamma_L * self.v_eps + rnd_L)

        atoms.set_velocities(v)
        return f

from .types_ import MDRecord
from .io_vasp import write_xdatcar_header, append_xdatcar_frame

ASE_OUT_DIR = "ase_files"


def run_md(atoms, calc, cfg):
    """Run NVE, NVT, or NPT molecular dynamics.

    Parameters
    ----------
    atoms : ase.Atoms
    calc  : ASE calculator (already attached to atoms by caller)
    cfg   : IncarConfig with IBRION=0 and MD-related fields
            MDALGO=1 → NVE (VelocityVerlet) or NVT Andersen (ANDERSEN_PROB > 0)
            MDALGO=2 → NVT Nosé-Hoover
            MDALGO=3, ISIF=2 → NVT Langevin; ISIF=3 → NPT Langevin

    Returns
    -------
    list[MDRecord]
    """
    atoms.calc = calc
    N = len(atoms)

    # Resolve TEEND: -1 means same as TEBEG (no ramp)
    T_start = cfg.TEBEG
    T_end = cfg.TEEND if cfg.TEEND >= 0.0 else cfg.TEBEG
    do_ramp = (cfg.MDALGO == 3) and (T_end != T_start)

    # Friction coefficients: convert from ps^-1 to fs^-1 (divide by 1000)
    gamma_ps = cfg.LANGEVIN_GAMMA[0] if cfg.LANGEVIN_GAMMA.size > 0 else 10.0
    friction_fs = gamma_ps / 1000.0  # atomic friction: ps^-1 -> fs^-1
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

            # Piston mass: scaled by number of atoms to balance barostat coupling.
            # A larger mass produces slower cell fluctuations. Default: ~100^2 per atom (eV*fs^2).
            # This is a sensible heuristic; future versions may allow user control.
            piston_mass = 1.0 * N * (100.0**2) # eV * fs^2
            
            # LangevinNPT uses friction in 1/ASE_time (same convention as ASE Langevin)
            dyn = LangevinNPT(
                atoms,
                timestep=cfg.POTIM * ASE_FS,
                temperature_K=T_start,
                externalstress=p_ext,
                friction=friction_fs / ASE_FS,  # atomic friction: fs^-1 -> ASE_time^-1
                barostat_friction=friction_L_fs / ASE_FS,  # lattice friction: fs^-1 -> ASE_time^-1
                piston_mass=piston_mass,
                logfile=md_log,
            )
        else:
            # NVT: Langevin
            # ASE's Langevin expects friction in 1/ASE_time (ASE uses its own time unit)
            # Convert fs^-1 to ASE time^-1 by dividing by ASE_FS (which is ~24.2 eV*fs/eV)
            dyn = Langevin(
                atoms,
                timestep=cfg.POTIM * ASE_FS,
                temperature_K=T_start,
                friction=friction_fs / ASE_FS,
                logfile=md_log,
            )
    else:
        raise ValueError(f"MDALGO={cfg.MDALGO} is not supported. Use 1 (Andersen/NVE), 2 (Nose-Hoover), or 3 (Langevin).")

    # Write XDATCAR header
    # For Langevin NPT (MDALGO=3, ISIF=3), the cell changes so the header is updated per frame.
    # For all other MD modes (NVE, NVT), the cell is fixed — write the header once.
    cell_relaxing = (cfg.MDALGO == 3 and cfg.ISIF == 3)
    if not cell_relaxing:
        write_xdatcar_header("XDATCAR", atoms)
    else:
        open("XDATCAR", "w").close()

    # Open trajectory file; write at each recorded step
    traj = Trajectory(traj_path, "w", atoms)

    records: list[MDRecord] = []

    for step in range(1, cfg.NSW + 1):
        # Temperature ramp for Langevin (linear interpolation)
        if do_ramp and cfg.NSW > 1:
            frac = (step - 1) / (cfg.NSW - 1)
            T_target = T_start + frac * (T_end - T_start)
            if hasattr(dyn, 'set_temperature'):
                dyn.set_temperature(temperature_K=T_target)
            elif hasattr(dyn, 'temp_K'):
                dyn.temp_K = T_target

        dyn.run(1)  # single MD step

        E_pot = atoms.get_potential_energy()
        E_kin = atoms.get_kinetic_energy()
        # Instantaneous temperature from kinetic energy: T = 2*Ekin / (3*N*kB)
        T_inst = 2.0 * E_kin / (3.0 * N * kB)
        E_tot = E_pot + E_kin

        rec = MDRecord(n=step, energy_pot=E_pot, energy_kin=E_kin, temperature=T_inst)
        records.append(rec)

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
            append_xdatcar_frame("XDATCAR", atoms, step, update_header=cell_relaxing)
            traj.write()

    traj.close()
    return records
