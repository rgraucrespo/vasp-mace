"""Read/write POSCAR, CONTCAR, and XDATCAR files."""

import os
import tempfile

import numpy as np
from ase import Atoms
from ase.constraints import FixCartesian
from ase.io import read, write


def read_poscar(path: str = "POSCAR", apply_selective_dynamics: bool = True) -> Atoms:
    """Read a POSCAR/CONTCAR file as an ASE ``Atoms`` object.

    Parameters
    ----------
    path
        POSCAR-like file to read.
    apply_selective_dynamics
        If ``True``, convert VASP Selective Dynamics flags into ASE
        ``FixCartesian`` constraints so relaxations respect fixed components.

    Returns
    -------
    ase.Atoms
        Structure read from ``path``. When Selective Dynamics flags are present,
        the original boolean array is preserved in
        ``atoms.arrays["selective_dynamics"]``.
    """
    atoms = read(path)

    if apply_selective_dynamics and "selective_dynamics" in atoms.arrays:
        sd = np.asarray(
            atoms.arrays["selective_dynamics"], dtype=bool
        )  # shape (N,3), T=free, F=fixed
        constraints = []

        for i, flags in enumerate(sd):
            fixed_mask = ~flags  # True where component is FIXED
            if fixed_mask.any():
                constraints.append(FixCartesian(fixed_mask, indices=[i]))

        if constraints:
            existing = atoms.constraints
            if existing is None:
                atoms.set_constraint(constraints)
            else:
                if isinstance(existing, (list, tuple)):
                    atoms.set_constraint(list(existing) + constraints)
                else:
                    atoms.set_constraint([existing] + constraints)

        atoms.arrays["selective_dynamics"] = sd

    return atoms


def write_contcar(path: str, atoms: Atoms) -> None:
    """Write a VASP-style CONTCAR file.

    Parameters
    ----------
    path
        Destination path.
    atoms
        Structure to write. ASE constraints are emitted as Selective Dynamics
        flags when possible.

    Notes
    -----
    The file is written through a temporary path and atomically renamed so a
    failed write does not leave a truncated output file.
    """
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        write(
            tmp,
            atoms,
            format="vasp",
            direct=True,
            vasp5=True,
            sort=False,
            ignore_constraints=False,
        )
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# XDATCAR
# ---------------------------------------------------------------------------


def _xdatcar_header_lines(atoms: Atoms) -> str:
    """Return the XDATCAR header block (title, scale, cell, species, counts)."""
    symbols = atoms.get_chemical_symbols()
    species = []
    counts = []
    for sym in symbols:
        if not species or species[-1] != sym:
            species.append(sym)
            counts.append(1)
        else:
            counts[-1] += 1
    cell = atoms.get_cell()
    lines = [
        atoms.get_chemical_formula(),
        "   1.00000000",
    ]
    for v in cell:
        lines.append(f"  {v[0]: .9f}  {v[1]: .9f}  {v[2]: .9f}")
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(c) for c in counts))
    return "\n".join(lines) + "\n"


def write_xdatcar_header(path: str, atoms: Atoms) -> None:
    """Write the fixed-cell XDATCAR header.

    Parameters
    ----------
    path
        Destination XDATCAR path.
    atoms
        Structure whose cell, formula, species order, and species counts are
        written to the header.
    """
    with open(path, "w") as f:
        f.write(_xdatcar_header_lines(atoms))


def append_xdatcar_frame(
    path: str, atoms: Atoms, step: int, update_header: bool = False
) -> None:
    """Append one frame of fractional coordinates to XDATCAR.

    Parameters
    ----------
    path
        XDATCAR path to append to.
    atoms
        Structure to serialize as fractional coordinates.
    step
        Configuration index written after ``Direct configuration=``.
    update_header : bool
        If True, prepend the full lattice header before the configuration line.
        Must be True for cell-relaxing runs (ISIF >= 3) so each frame carries
        the current cell vectors, matching real VASP XDATCAR format.
    """
    scaled = atoms.get_scaled_positions()
    with open(path, "a") as f:
        if update_header:
            f.write(_xdatcar_header_lines(atoms))
        f.write(f"Direct configuration=     {step}\n")
        for pos in scaled:
            f.write(f"  {pos[0]: .9f}  {pos[1]: .9f}  {pos[2]: .9f}\n")
