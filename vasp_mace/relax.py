# -*- coding: utf-8 -*-
"""
Cell/atomic relaxation driver for vasp-mace.

- ISIF=2: relax atomic positions only (stress is still computed & logged).
- ISIF=3: relax cell + atoms towards hydrostatic target pressure from INCAR:PSTRESS.
- EDIFFG < 0: force-based stop. If ISIF=3 we ALSO require per-component |σ−pI| <= |EDIFFG|*NIONS/VOL.
- EDIFFG > 0: energy-per-ion stop (plus a gentle force cap).

Logging prints max|σ| and max|σ−pI| so PSTRESS runs are easy to diagnose.
"""

from __future__ import annotations

import numpy as np
from ase.optimize import BFGS, FIRE, LBFGS
from ase.io.trajectory import Trajectory
from ase.units import GPa

# Prefer ase.filters (modern); fall back to older location if needed
try:
    from ase.filters import UnitCellFilter
except Exception:  # pragma: no cover
    from ase.constraints import UnitCellFilter  # very old ASE

from .logging_utils import StepLogger

EV_A3_TO_GPA = 160.21766208  # 1 eV/Å^3 in GPa


def run_relax(atoms, calc, cfg, optimizer: str = "BFGS", pressure_GPa: float = 0.0):
    """Run relaxation according to INCAR-like cfg.

    Parameters
    ----------
    atoms : ase.Atoms
    calc  : ASE calculator
    cfg   : has attributes EDIFFG, NSW, ISIF
    optimizer : str, one of {'BFGS','FIRE','LBFGS'}
    pressure_GPa : float, hydrostatic target pressure (from PSTRESS in INCAR, converted to GPa)
    """
    atoms.calc = calc

    # Optimizer selection
    opt_name = (optimizer or "BFGS").upper()
    Optim = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}.get(opt_name, BFGS)

    # Map EDIFFG to stopping criteria
    if cfg.EDIFFG < 0:
        f_tol = abs(cfg.EDIFFG)   # eV/Å
        ediff_tol = None
    else:
        f_tol = 0.05              # eV/Å (gentle cap; energy drives convergence)
        ediff_tol = cfg.EDIFFG    # eV per ion

    # Target object based on ISIF
    if cfg.ISIF == 3:
        target = UnitCellFilter(
            atoms,
            scalar_pressure=pressure_GPa * GPa,  # ASE expects eV/Å^3; GPa constant converts
            hydrostatic_strain=True,
        )
    elif cfg.ISIF == 2:
        target = atoms
    else:
        print(f"[warn] ISIF={cfg.ISIF} not supported; using ISIF=2.")
        cfg.ISIF = 2
        target = atoms

    logger = StepLogger()
    traj = Trajectory("mace.traj", "w", atoms)
    dyn = Optim(target, logfile="opt.log")  # ASE optimizer

    # VASP-style stress tolerance (only relevant for ISIF=3 with EDIFFG<0)
    stress_tol = None
    if cfg.ISIF == 3 and ediff_tol is None:
        nions = len(atoms)
        vol = max(atoms.get_volume(), 1e-12)
        stress_tol = f_tol * nions / vol  # eV/Å^3

    converged = False
    for n in range(1, cfg.NSW + 1):
        # one ionic step per loop to emulate VASP-style ionic iterations
        dyn.run(fmax=f_tol, steps=1)

        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        rec = logger.log(n=n, energy=E, forces=F)

        # Stress acquisition + PSTRESS target construction
        try:
            stress = atoms.get_stress(voigt=True)  # (6,) eV/Å^3  [xx, yy, zz, yz, xz, xy]
            p_eva3 = pressure_GPa * GPa
            target_voigt = np.array([-p_eva3, -p_eva3, -p_eva3, 0.0, 0.0, 0.0], dtype=float)
            stress_err = np.asarray(stress, dtype=float) - target_voigt

            max_sig = float(np.max(np.abs(stress)))
            max_err = float(np.max(np.abs(stress_err)))
            print(
                f"[step {n}] Fmax={rec.fmax:.3f} eV/Å | "
                f"max|σ|={max_sig:.3e} eV/Å³ ({max_sig*EV_A3_TO_GPA:.3f} GPa) | "
                f"max|σ−pI|={max_err:.3e} eV/Å³ ({max_err*EV_A3_TO_GPA:.3f} GPa)"
            )
        except Exception:
            stress = None
            stress_err = None
            print(f"[step {n}] Fmax={rec.fmax:.3f} eV/Å")

        traj.write()

        # --- Convergence tests ---
        if ediff_tol is None:
            # Force-based (EDIFFG < 0)
            if cfg.ISIF == 2:
                if rec.fmax <= f_tol:
                    converged = True
                    break
            elif cfg.ISIF == 3:
                # Require both forces and per-component stress error within tol
                if stress_err is not None and stress_tol is not None:
                    stress_ok = bool(np.all(np.abs(stress_err) <= stress_tol))
                else:
                    # If we cannot evaluate stress error, fall back to forces only
                    stress_ok = (stress_tol is None)
                if rec.fmax <= f_tol and stress_ok:
                    converged = True
                    break
        else:
            # Energy-per-ion (EDIFFG > 0) + a gentle force cap
            ions = max(len(atoms), 1)
            dE_per_ion = abs(rec.dE) / ions
            if dE_per_ion <= ediff_tol and rec.fmax <= f_tol:
                converged = True
                break

    traj.close()
    return logger.steps, converged
