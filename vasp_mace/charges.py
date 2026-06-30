"""Shared EEQ partial-charge engine, backed by ``dftd4``.

The Fortran-backed ``dftd4`` package computes electronegativity-equilibration
(EEQ) charges as part of evaluating DFT-D4. The same EEQ charges are reused by
the implicit-solvation term, so this module is the single ("shared") charge
engine for both DFT-D4 and solvation.

The total system charge is fixed to zero: vasp-mace currently supports only
neutral systems. Charged systems are deferred because they would also need an
electrolyte/screening model and POTCAR-derived valence counts (see
not_for_release/solvation_design.md).
"""

from typing import Any

import numpy as np

# vasp_mace/charges.py

# Total system charge. Hardcoded neutral; see module docstring.
_TOTAL_CHARGE = 0.0


def eeq_charges(atoms: Any) -> np.ndarray:
    """Return per-atom EEQ partial charges for a neutral system.

    Parameters
    ----------
    atoms
        ASE ``Atoms``. Periodic cells are handled via the lattice and per-axis
        PBC flags; non-periodic systems (molecules/clusters, or slabs treated
        as clusters) are also supported.

    Returns
    -------
    numpy.ndarray
        Partial charges in units of the elementary charge, shape
        ``(len(atoms),)``, summing to zero for the neutral system.

    Raises
    ------
    RuntimeError
        If the optional ``dftd4`` package is not installed.
    """
    try:
        from dftd4.interface import DispersionModel
    except ImportError as exc:
        raise RuntimeError(
            "EEQ charges require the optional dftd4 package. Install it with "
            "`pip install dftd4` (see requirements/dftd4.txt)."
        ) from exc
    from ase.units import Bohr

    numbers = atoms.get_atomic_numbers()
    # dftd4's interface works in atomic units (Bohr).
    positions = atoms.get_positions() / Bohr
    pbc = atoms.get_pbc()

    if pbc.any():
        lattice = np.asarray(atoms.get_cell()) / Bohr
        model = DispersionModel(
            numbers, positions, charge=_TOTAL_CHARGE, lattice=lattice, periodic=pbc
        )
    else:
        model = DispersionModel(numbers, positions, charge=_TOTAL_CHARGE)

    props = model.get_properties()
    return np.asarray(props["partial charges"], dtype=float)
