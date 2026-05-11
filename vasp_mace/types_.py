"""Shared dataclasses used by vasp-mace run modes."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class IncarConfig:
    """Parsed subset of VASP-style INCAR settings used by vasp-mace.

    Attributes
    ----------
    EDIFFG
        Convergence criterion. Negative values are force tolerances in eV/Å;
        positive values are energy-change tolerances in eV per ion for
        relaxation. NEB treats positive values as a force tolerance.
    NSW
        Maximum number of ionic, MD, or NEB optimization steps. ``0`` selects a
        single-point calculation for non-NEB runs.
    ISIF
        Degrees of freedom to relax. ``2`` fixes the cell, ``3`` relaxes
        positions and the full cell, ``4`` relaxes shape at fixed volume,
        ``7`` relaxes volume only, and ``8`` relaxes positions plus isotropic
        volume.
    PSTRESS
        Target hydrostatic pressure in kBar.
    IBRION
        Run-mode/optimizer selector: ``0`` for MD, ``1``/``2``/``3`` for
        relaxation optimizers, and ``5``/``6`` for phonons.
    IVDW
        DFT-D3 dispersion selector. Supported values are ``0``, ``11``, ``12``,
        ``13``, and ``14``.
    TEBEG
        Initial MD temperature in K.
    TEEND
        Final MD temperature in K for a linear ramp. ``-1`` means use
        ``TEBEG``.
    POTIM
        MD timestep in fs, or phonon displacement amplitude in Å for
        ``IBRION=5/6``.
    NBLOCK
        MD trajectory/XDATCAR output interval in steps.
    MDALGO
        MD integrator selector: ``1`` for NVE/Andersen, ``2`` for
        Nosé-Hoover, and ``3`` for Langevin/NPT Langevin.
    ANDERSEN_PROB
        Andersen thermostat collision probability per MD step.
    LANGEVIN_GAMMA
        Atomic Langevin friction coefficients in ps^-1. A scalar applies to all
        atoms; multiple values are assigned by species in POSCAR order.
    LANGEVIN_GAMMA_L
        Lattice Langevin friction coefficient in ps^-1 for NPT MD.
    SMASS
        Nosé-Hoover damping parameter, or fallback Langevin friction when
        ``LANGEVIN_GAMMA`` is absent.
    PMASS
        NPT piston mass in amu. ``0`` requests the automatic default.
    NFREE
        Number of finite-difference displacements per phonon degree of freedom:
        ``1`` for forward differences or ``2`` for central differences.
    IMAGES
        Number of intermediate NEB images. Values greater than zero select NEB
        mode.
    SPRING
        NEB spring constant tag. Negative values follow VASP's NEB convention;
        vasp-mace uses ``abs(SPRING)`` as the spring constant in eV/Å^2.
    LCLIMB
        Whether to use climbing-image NEB.
    ML_LHEAT
        VASP-style logical: when true, MACE MD writes a VASP-compatible
        ``ML_HEAT`` file with per-step heat-flux components. Supported only
        for fixed-cell NVE production MD (``IBRION = 0``, ``MDALGO = 1``,
        ``ANDERSEN_PROB = 0.0``, ``ISIF = 2``) and ``IVDW = 0``; ignored for
        non-MD modes.
    ML_HEAT_INTERVAL
        vasp-mace extension: write ``ML_HEAT`` every ``ML_HEAT_INTERVAL`` MD
        steps. Defaults to ``1`` (every step) to match VASP behaviour.
    raw
        Raw INCAR key/value strings after comment stripping and key
        upper-casing.
    """

    EDIFFG: float  # <0 force tol (eV/Å); >0 energy tol (eV)
    NSW: int  # max ionic steps
    ISIF: int  # 2 positions-only, 3 variable-cell, 4 shape+atoms, 7 volume-only, 8 positions+volume
    PSTRESS: float  # pressure in kB
    IBRION: int  # -1 none, 0 MD; relaxation: 1 LBFGS, 2 BFGS, 3 FIRE; NEB: 1/2 MDMin, 3 FIRE
    IVDW: int  # 0 none, 11 D3(zero), 12 D3(BJ), 13 D3(zero)+ATM, 14 D3(BJ)+ATM
    TEBEG: float  # MD: starting temperature (K)
    TEEND: float  # MD: ending temperature (K); -1 = same as TEBEG
    POTIM: float  # MD: timestep (fs)
    NBLOCK: int  # MD: write XDATCAR / trajectory every NBLOCK steps
    MDALGO: int  # MD: 1 NVE/Andersen, 2 Nose-Hoover, 3 Langevin
    ANDERSEN_PROB: float  # MD: collision probability for Andersen thermostat
    LANGEVIN_GAMMA: (
        np.ndarray
    )  # MD: friction coefficients (ps^-1) for Langevin; one value per species or a scalar
    LANGEVIN_GAMMA_L: float  # MD: lattice friction coefficient (ps^-1) for Langevin NPT
    SMASS: float  # MD: Nose-Hoover mass or Langevin friction (ps^-1); <=0 uses default 0.01 fs^-1
    PMASS: float = (
        0.0  # MD: piston mass (amu) for Langevin NPT; 0 = auto (N × 10000 amu)
    )
    NFREE: int = 2  # phonons: displacements per DOF (1 = +only, 2 = ±central)
    IMAGES: int = 0  # NEB: number of intermediate images
    SPRING: float = (
        -5.0
    )  # NEB: spring constant; SPRING < 0 (NEB convention); k = |SPRING| eV/Å^2
    LCLIMB: bool = False  # NEB: enable climbing-image (CI-NEB)
    ML_LHEAT: bool = False  # VASP-style: write ML_HEAT during MD
    ML_HEAT_INTERVAL: int = 1  # vasp-mace extension: write every N MD steps
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class MDRecord:
    """Single molecular-dynamics step summary.

    Attributes
    ----------
    n
        One-based MD step index.
    energy_pot
        Potential energy in eV.
    energy_kin
        Kinetic energy in eV.
    temperature
        Instantaneous temperature in K.
    """

    n: int  # step index (1-based)
    energy_pot: float  # potential energy (eV)
    energy_kin: float  # kinetic energy (eV)
    temperature: float  # instantaneous temperature (K)
