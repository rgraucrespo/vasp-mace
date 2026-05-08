# -*- coding: utf-8 -*-
"""
Cell/atomic relaxation driver for vasp-mace.

- ISIF=2: relax atomic positions only (stress is still computed & logged).
- ISIF=3: relax cell + atoms via UnitCellFilter towards hydrostatic target pressure (PSTRESS).
- EDIFFG < 0: force-based stop; fmax includes cell DOF forces from UnitCellFilter.
- EDIFFG > 0: energy-per-ion stop (plus a gentle force cap).

Logging prints max|σ| and max|σ−pI| so PSTRESS runs are easy to diagnose.
"""

from __future__ import annotations

import os
import numpy as np

ASE_OUT_DIR = "ase_files"
from ase.optimize import BFGS, FIRE, LBFGS
from ase.io.trajectory import Trajectory
from ase.units import GPa

# Prefer ase.filters (modern); fall back to older location if needed
try:
    from ase.filters import UnitCellFilter, ExpCellFilter
except Exception:  # pragma: no cover
    from ase.constraints import UnitCellFilter  # very old ASE

    try:
        from ase.filters import ExpCellFilter
    except Exception:
        from ase.constraints import ExpCellFilter

from ase.constraints import FixAtoms

from .logging_utils import StepLogger
from .io_vasp import write_xdatcar_header, append_xdatcar_frame

EV_A3_TO_KBAR = 1602.1766208  # 1 eV/Å^3 in kBar


def run_relax(atoms, calc, cfg, optimizer: str = "BFGS", pressure_GPa: float = 0.0):
    """Run relaxation according to INCAR-like cfg.

    Parameters
    ----------
    atoms : ase.Atoms
    calc  : ASE calculator
    cfg   : has attributes EDIFFG, NSW, ISIF, IBRION
    optimizer : str, fallback optimizer name {'BFGS','FIRE','LBFGS'} used only when
                IBRION was not explicitly set in the raw INCAR (i.e. 'IBRION' not in cfg.raw).
    pressure_GPa : float, hydrostatic target pressure (from PSTRESS in INCAR, converted to GPa)
    """
    atoms.calc = calc

    # Optimizer selection: IBRION in raw INCAR takes precedence over --optimizer CLI flag
    if "IBRION" in cfg.raw:
        # IBRION drives optimizer: 1=LBFGS, 3=FIRE, 2=BFGS (default)
        _ibrion_map = {1: "LBFGS", 2: "BFGS", 3: "FIRE"}
        opt_name = _ibrion_map.get(cfg.IBRION, "BFGS")
    else:
        opt_name = (optimizer or "BFGS").upper()
    Optim = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}.get(opt_name, BFGS)

    # Map EDIFFG to stopping criteria
    if cfg.EDIFFG < 0:
        f_tol = abs(cfg.EDIFFG)  # eV/Å
        ediff_tol = None
    else:
        f_tol = 0.05  # eV/Å (gentle cap; energy drives convergence)
        ediff_tol = cfg.EDIFFG  # eV per ion

    # Target object based on ISIF
    if cfg.ISIF == 2:
        # Relax positions only; cell fixed
        target = atoms
    elif cfg.ISIF == 3:
        # Relax positions + full cell (shape + volume) — VASP ISIF=3.
        # UnitCellFilter without hydrostatic_strain allows all 6 cell DOFs to change.
        target = UnitCellFilter(
            atoms,
            scalar_pressure=pressure_GPa * GPa,
        )
    elif cfg.ISIF == 4:
        # Relax positions + cell shape; volume is conserved — VASP ISIF=4
        target = ExpCellFilter(atoms, constant_volume=True)
    elif cfg.ISIF == 7:
        # Volume only, positions fixed: add FixAtoms on top of any existing constraints
        existing = list(atoms.constraints) if atoms.constraints else []
        atoms.set_constraint(existing + [FixAtoms(indices=list(range(len(atoms))))])
        target = UnitCellFilter(
            atoms,
            hydrostatic_strain=True,
            scalar_pressure=pressure_GPa * GPa,
        )
    elif cfg.ISIF == 8:
        # Relax positions + volume; cell shape is fixed (isotropic scaling only)
        target = UnitCellFilter(
            atoms,
            hydrostatic_strain=True,
            scalar_pressure=pressure_GPa * GPa,
        )
    else:
        raise ValueError(
            f"ISIF={cfg.ISIF} is not supported. "
            f"Supported values: 2 (positions only), 3 (full cell+atoms), "
            f"4 (cell shape+atoms, fixed volume), 7 (volume only, positions fixed), "
            f"8 (positions+volume, fixed shape)."
        )

    os.makedirs(ASE_OUT_DIR, exist_ok=True)
    logger = StepLogger()
    traj = Trajectory(os.path.join(ASE_OUT_DIR, "mace.traj"), "w", atoms)
    dyn = Optim(target, logfile=os.path.join(ASE_OUT_DIR, "opt.log"))

    # For cell-relaxing runs (ISIF >= 3), the cell changes at each step so
    # the XDATCAR header must be repeated per frame (VASP convention).
    # For positions-only (ISIF == 2), write the header once up front.
    cell_relaxing = cfg.ISIF != 2
    if not cell_relaxing:
        write_xdatcar_header("XDATCAR", atoms)
    else:
        open(
            "XDATCAR", "w"
        ).close()  # create/truncate; frames will self-contain headers

    converged = False
    for n in range(1, cfg.NSW + 1):
        # one ionic step per loop to emulate VASP-style ionic iterations
        dyn.run(fmax=f_tol, steps=1)

        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        rec = logger.log(n=n, energy=E, forces=F, atoms=atoms)

        # fmax for convergence: use the optimizer's combined forces (includes cell DOFs
        # for ISIF=3/4/7 via UnitCellFilter/ExpCellFilter) so the cell is not ignored.
        if target is atoms:
            fmax_opt = rec.fmax
        else:
            fmax_opt = float(np.max(np.linalg.norm(target.get_forces(), axis=1)))

        # Stress acquisition + PSTRESS target construction
        try:
            stress = atoms.get_stress(
                voigt=True
            )  # (6,) eV/Å^3  [xx, yy, zz, yz, xz, xy]
            if abs(pressure_GPa) > 1e-9:
                p_eva3 = pressure_GPa * GPa
                target_voigt = np.array(
                    [-p_eva3, -p_eva3, -p_eva3, 0.0, 0.0, 0.0], dtype=float
                )
                stress_err = np.asarray(stress, dtype=float) - target_voigt
                max_err = float(np.max(np.abs(stress_err)))
                stress_str = f"max|σ-pI|={max_err*EV_A3_TO_KBAR:.3f} kBar"
            else:
                max_sig = float(np.max(np.abs(stress)))
                stress_str = f"max|σ|={max_sig*EV_A3_TO_KBAR:.3f} kBar"

            print(
                f"[step {n}] E={rec.energy:.6f} eV | dE={rec.dE:.6e} eV | "
                f"Fmax={fmax_opt:.3f} eV/Å | {stress_str}"
            )
        except Exception:
            print(
                f"[step {n}] E={rec.energy:.6f} eV | dE={rec.dE:.6e} eV | Fmax={fmax_opt:.3f} eV/Å"
            )

        traj.write()
        append_xdatcar_frame("XDATCAR", atoms, n, update_header=cell_relaxing)

        # --- Convergence tests ---
        if ediff_tol is None:
            # Force-based (EDIFFG < 0): fmax_opt includes cell DOF forces for cell relaxations
            if fmax_opt <= f_tol:
                converged = True
                break
        else:
            # Energy-per-ion (EDIFFG > 0) + a gentle force cap
            ions = max(len(atoms), 1)
            dE_per_ion = abs(rec.dE) / ions
            if dE_per_ion <= ediff_tol and fmax_opt <= f_tol:
                converged = True
                break

    traj.close()
    return logger.steps, converged
