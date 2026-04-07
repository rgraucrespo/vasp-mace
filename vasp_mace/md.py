# -*- coding: utf-8 -*-
"""
Molecular dynamics driver for vasp-mace.

Supports:
  MDALGO=1 : NVE (NVT-less) via ASE VelocityVerlet
  MDALGO=2 : NVT Langevin thermostat

Outputs:
  XDATCAR              -- incremental trajectory in fractional coordinates
  ase_files/md.log     -- ASE MD log
  ase_files/mace.traj  -- ASE binary trajectory (every NBLOCK steps)
"""

from __future__ import annotations

import os

import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.units import fs as ASE_FS, kB

from .types_ import MDRecord
from .io_vasp import write_xdatcar_header, append_xdatcar_frame

ASE_OUT_DIR = "ase_files"


def run_md(atoms, calc, cfg):
    """Run NVE or NVT Langevin MD.

    Parameters
    ----------
    atoms : ase.Atoms
    calc  : ASE calculator (already attached to atoms by caller)
    cfg   : IncarConfig with IBRION=0 and MD-related fields

    Returns
    -------
    list[MDRecord]
    """
    atoms.calc = calc
    N = len(atoms)

    # Resolve TEEND: -1 means same as TEBEG (no ramp)
    T_start = cfg.TEBEG
    T_end = cfg.TEEND if cfg.TEEND >= 0.0 else cfg.TEBEG
    do_ramp = (cfg.MDALGO == 2) and (T_end != T_start)

    # Langevin friction: SMASS > 0 → convert ps^-1 to fs^-1; else default 0.01 fs^-1
    friction = (cfg.SMASS / 1000.0) if cfg.SMASS > 0 else 0.01  # fs^-1

    os.makedirs(ASE_OUT_DIR, exist_ok=True)
    md_log = os.path.join(ASE_OUT_DIR, "md.log")
    traj_path = os.path.join(ASE_OUT_DIR, "mace.traj")

    # Initialise velocities from Maxwell-Boltzmann at TEBEG
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)

    # Create integrator
    if cfg.MDALGO == 1:
        # NVE: VelocityVerlet
        dyn = VelocityVerlet(atoms, timestep=cfg.POTIM * ASE_FS, logfile=md_log)
    else:
        # NVT: Langevin
        dyn = Langevin(
            atoms,
            timestep=cfg.POTIM * ASE_FS,
            temperature_K=T_start,
            friction=friction / ASE_FS,  # ASE Langevin expects friction in 1/ASE_time
            logfile=md_log,
        )

    # Write XDATCAR header (static cell assumed for NVT/NVE)
    write_xdatcar_header("XDATCAR", atoms)

    # Open trajectory file; write at each recorded step
    traj = Trajectory(traj_path, "w", atoms)

    records: list[MDRecord] = []

    for step in range(1, cfg.NSW + 1):
        # Temperature ramp for Langevin (linear interpolation)
        if do_ramp and cfg.NSW > 1:
            frac = (step - 1) / (cfg.NSW - 1)
            T_target = T_start + frac * (T_end - T_start)
            dyn.set_temperature(temperature_K=T_target)

        dyn.run(1)  # single MD step

        E_pot = atoms.get_potential_energy()
        E_kin = atoms.get_kinetic_energy()
        # Instantaneous temperature from kinetic energy: T = 2*Ekin / (3*N*kB)
        T_inst = 2.0 * E_kin / (3.0 * N * kB)
        E_tot = E_pot + E_kin

        rec = MDRecord(n=step, energy_pot=E_pot, energy_kin=E_kin, temperature=T_inst)
        records.append(rec)

        print(
            f"[step {step}] T={T_inst:.2f} K | "
            f"Epot={E_pot:.6f} eV | "
            f"Ekin={E_kin:.6f} eV | "
            f"Etot={E_tot:.6f} eV"
        )

        if step % cfg.NBLOCK == 0:
            append_xdatcar_frame("XDATCAR", atoms, step)
            traj.write()

    traj.close()
    return records
