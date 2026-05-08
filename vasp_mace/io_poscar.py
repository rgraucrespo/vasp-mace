"""Read/write POSCAR, CONTCAR, and XDATCAR files."""

import os
import tempfile

import numpy as np
from ase.constraints import FixCartesian
from ase.io import read, write


def read_poscar(path: str = "POSCAR", apply_selective_dynamics: bool = True):
    """Read POSCAR/CONTCAR with ASE.

    If 'Selective dynamics' flags are present and apply_selective_dynamics=True,
    convert them into ASE FixCartesian constraints so relaxations respect them.
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


def write_contcar(path: str, atoms):
    """Write a VASP-style CONTCAR, preserving Selective Dynamics flags if present.

    Writes to a temp file first and renames atomically so a failed write
    never leaves a truncated or corrupt file at *path*.
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


def _xdatcar_header_lines(atoms) -> str:
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


def write_xdatcar_header(path: str, atoms) -> None:
    """Write the XDATCAR header once (for MD / fixed-cell runs)."""
    with open(path, "w") as f:
        f.write(_xdatcar_header_lines(atoms))


def append_xdatcar_frame(
    path: str, atoms, step: int, update_header: bool = False
) -> None:
    """Append one frame of fractional coordinates to XDATCAR.

    Parameters
    ----------
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
