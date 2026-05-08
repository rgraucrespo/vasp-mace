"""Write OSZICAR, OUTCAR (relaxation, single-point, MD) files."""

from typing import List, Optional

import numpy as np

from .logging_utils import StepRecord

_EV_A3_TO_KB = 1602.1766  # 1 eV/Å³ → kBar


def _fmt_fort(val: float, ndigits: int = 8) -> str:
    """Fortran-style scientific notation: sign + '.' + ndigits + 'E' + ±XX.

    e.g. -47.1299506 → '-.47129951E+02'  (ndigits=8)
    """
    if val == 0.0:
        return f" .{'0' * ndigits}E+00"
    sign = "-" if val < 0 else " "
    mag = abs(val)
    exp = int(np.floor(np.log10(mag))) + 1
    raw = mag * 10 ** (ndigits - exp)
    mantissa_int = round(raw)
    if mantissa_int >= 10**ndigits:
        mantissa_int //= 10
        exp += 1
    return f"{sign}.{mantissa_int:0{ndigits}d}E{exp:+03d}"


def _stress_kB(stress_voigt):
    """Convert ASE Voigt stress (eV/Å³, tensile-positive) to VASP kBar components
    (compressive-positive). Returns (XX, YY, ZZ, XY, YZ, ZX)."""
    s = stress_voigt
    c = _EV_A3_TO_KB
    return (-s[0] * c, -s[1] * c, -s[2] * c, -s[5] * c, -s[3] * c, -s[4] * c)


def _rec_basis(cell):
    """Reciprocal lattice (rows), VASP convention: G·a = δ  (no 2π)."""
    return np.linalg.inv(cell).T


# ---------------------------------------------------------------------------
# OSZICAR
# ---------------------------------------------------------------------------


def write_oszicar(path: str, steps: List[StepRecord]) -> None:
    """VASP-style OSZICAR: one ionic-step line per step.

    Format (per real VASP):
        N F= -.XXXXXXXE+XX E0= -.XXXXXXXE+XX  d E =-.XXXXXXE+XX
    F = free energy (= E0 for MACE, no entropy).
    d E for step 1 is the energy itself (VASP convention).
    """
    with open(path, "w") as f:
        for s in steps:
            E_str = _fmt_fort(s.energy, 8)
            dE_val = s.energy if s.n == 1 else s.dE
            dE_str = _fmt_fort(dE_val, 6)
            f.write(f"   {s.n:3d} F= {E_str} E0= {E_str}  d E ={dE_str}\n")


# ---------------------------------------------------------------------------
# OUTCAR — relaxation / single-point
# ---------------------------------------------------------------------------


def _write_step_block(f, rec: StepRecord, atoms=None):
    """Write one ionic-step block to an already-open OUTCAR file handle."""
    cell = rec.cell if rec.cell is not None else np.array(atoms.get_cell())
    pos = rec.positions if rec.positions is not None else atoms.get_positions()
    frc = rec.forces if rec.forces is not None else atoms.get_forces()
    sv = rec.stress

    vol = float(np.linalg.det(cell))

    f.write(
        f"\n --------------------------------------- "
        f"Iteration{rec.n:6d}(   1)  "
        f"---------------------------------------\n\n"
    )

    if sv is not None:
        XX, YY, ZZ, XY, YZ, ZX = _stress_kB(sv)
        tXX, tYY, tZZ = -sv[0] * vol, -sv[1] * vol, -sv[2] * vol
        tXY, tYZ, tZX = -sv[5] * vol, -sv[3] * vol, -sv[4] * vol
        ext_p = (XX + YY + ZZ) / 3.0
        f.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
        f.write(
            "  Direction    XX          YY          ZZ"
            "          XY          YZ          ZX\n"
        )
        f.write("  " + "-" * 86 + "\n")
        f.write(
            f"  Total   {tXX:11.5f} {tYY:11.5f} {tZZ:11.5f}"
            f" {tXY:11.5f} {tYZ:11.5f} {tZX:11.5f}\n"
        )
        f.write(
            f"  in kB   {XX:11.5f} {YY:11.5f} {ZZ:11.5f}"
            f" {XY:11.5f} {YZ:11.5f} {ZX:11.5f}\n"
        )
        f.write(
            f"  external pressure ={ext_p:12.2f} kB"
            f"  Pullay stress =        0.00 kB\n\n"
        )

    rec_b = _rec_basis(cell)
    lens_d = np.linalg.norm(cell, axis=1)
    lens_r = np.linalg.norm(rec_b, axis=1)
    f.write(" VOLUME and BASIS-vectors are now :\n")
    f.write(" " + "-" * 77 + "\n")
    f.write(f"  volume of cell :{vol:12.2f}\n")
    f.write(
        "      direct lattice vectors" "                 reciprocal lattice vectors\n"
    )
    for i in range(3):
        f.write(
            f"  {cell[i,0]:12.9f} {cell[i,1]:12.9f} {cell[i,2]:12.9f}"
            f"   {rec_b[i,0]:12.9f} {rec_b[i,1]:12.9f} {rec_b[i,2]:12.9f}\n"
        )
    f.write("\n  length of vectors\n")
    f.write(
        f"  {lens_d[0]:12.9f} {lens_d[1]:12.9f} {lens_d[2]:12.9f}"
        f"   {lens_r[0]:12.9f} {lens_r[1]:12.9f} {lens_r[2]:12.9f}\n\n"
    )

    f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
    f.write(" " + "-" * 83 + "\n")
    for p, fv in zip(pos, frc):
        f.write(
            f" {p[0]:12.5f} {p[1]:12.5f} {p[2]:12.5f}"
            f"    {fv[0]:13.6f} {fv[1]:13.6f} {fv[2]:13.6f}\n"
        )
    f.write(" " + "-" * 83 + "\n")
    drift = np.sum(frc, axis=0)
    f.write(
        f"    total drift:                               "
        f"{drift[0]:13.6f} {drift[1]:13.6f} {drift[2]:13.6f}\n\n"
    )

    f.write("\n  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
    f.write("  ---------------------------------------------------\n")
    f.write(f"  free  energy   TOTEN  =     {rec.energy:20.8f} eV\n\n")
    f.write(
        f"  energy  without entropy=    {rec.energy:20.8f}"
        f"  energy(sigma->0) =    {rec.energy:20.8f}\n\n"
    )
    f.write(" " + "-" * 102 + "\n")


def _write_outcar_header(f, atoms, incar_raw):
    """Write the OUTCAR file header (species summary, INCAR echo)."""
    symbols = atoms.get_chemical_symbols()
    nions = len(atoms)

    f.write(" vasp-mace (MACE/ASE interface) -- VASP-compatible OUTCAR\n")
    f.write(f" NIONS =   {nions}\n\n")

    if incar_raw:
        f.write(" INCAR:\n")
        for k, v in incar_raw.items():
            f.write(f"   {k} = {v}\n")
        f.write("\n")

    seen = {}
    counts = []
    for sym in symbols:
        if sym not in seen:
            seen[sym] = len(seen)
            counts.append(0)
        counts[seen[sym]] += 1

    f.write(" ions per type = " + "  ".join(str(c) for c in counts) + "\n")
    f.write(
        " POMASS = "
        + "  ".join(f"{atoms.get_masses()[symbols.index(sp)]:.3f}" for sp in seen)
        + "\n\n"
    )
    f.write(" " + "-" * 102 + "\n\n")


def write_outcar(
    path: str,
    atoms,
    steps: List[StepRecord],
    incar_raw: Optional[dict] = None,
    converged: bool = True,
    elapsed: float = 0.0,
    cpu_time: Optional[float] = None,
) -> None:
    """Write a VASP-style OUTCAR for a relaxation or single-point run.

    Each ionic step gets a full block: stress, cell, positions/forces, energy.
    Timing info is appended at the end.
    """
    with open(path, "w") as f:
        _write_outcar_header(f, atoms, incar_raw)
        for rec in steps:
            _write_step_block(f, rec, atoms)
        if converged:
            f.write(
                "\n reached required accuracy"
                " - stopping structural energy minimisation\n"
            )
        else:
            f.write("\n maximum ionic steps (NSW) exceeded\n")

    write_outcar_tail(path, elapsed, cpu_time, mode="a")


def write_outcar_tail(
    path: str, elapsed: float, cpu_time: float = None, mode: str = "a"
) -> None:
    """Append (or write) the VASP-style timing footer to an OUTCAR file."""
    cpu = cpu_time if cpu_time is not None else elapsed
    with open(path, mode) as f:
        f.write("\n\n General timing and accounting informations for this job:\n")
        f.write(" ========================================================\n\n")
        f.write(f"                  Total CPU time used (sec):{cpu:13.3f}\n")
        f.write(f"                         Elapsed time (sec):{elapsed:13.3f}\n\n")
        f.write(f"                   Maximum memory used (kb):          N/A\n")
        f.write(f"                   Average memory used (kb):          N/A\n")


def write_outcar_like(
    path,
    atoms,
    steps,
    stress=None,
    incar_raw=None,
    converged=True,
    elapsed=0.0,
    cpu_time=None,
):
    """Thin wrapper kept for backward compatibility; delegates to write_outcar."""
    upgraded = []
    for s in steps:
        if isinstance(s, StepRecord):
            upgraded.append(s)
        else:
            sv = np.array(stress) if stress is not None else None
            upgraded.append(
                StepRecord(
                    n=s.n,
                    energy=s.energy,
                    dE=s.dE,
                    fmax=s.fmax,
                    positions=atoms.get_positions().copy(),
                    forces=(
                        atoms.get_forces().copy()
                        if hasattr(atoms, "get_forces")
                        else None
                    ),
                    stress=sv,
                    cell=np.array(atoms.get_cell()).copy(),
                )
            )
    write_outcar(
        path,
        atoms,
        upgraded,
        incar_raw=incar_raw,
        converged=converged,
        elapsed=elapsed,
        cpu_time=cpu_time,
    )


# ---------------------------------------------------------------------------
# OUTCAR — MD (incremental writers)
# ---------------------------------------------------------------------------


def write_md_outcar_header(path: str, atoms, incar_raw: Optional[dict] = None) -> None:
    """Write the OUTCAR header for an MD run (called once before the loop)."""
    with open(path, "w") as f:
        _write_outcar_header(f, atoms, incar_raw)


def append_md_outcar_step(
    path: str,
    atoms,
    n: int,
    energy_pot: float,
    energy_kin: float,
    temperature: float,
) -> None:
    """Append one MD ionic-step block to OUTCAR (called every step)."""
    cell = np.array(atoms.get_cell())
    pos = atoms.get_positions()
    vol = float(np.linalg.det(cell))
    rec_b = _rec_basis(cell)
    lens_d = np.linalg.norm(cell, axis=1)
    lens_r = np.linalg.norm(rec_b, axis=1)

    try:
        frc = atoms.get_forces()
    except Exception:
        frc = np.zeros_like(pos)

    try:
        sv = atoms.get_stress(voigt=True)
    except Exception:
        sv = None

    with open(path, "a") as f:
        f.write(
            f"\n --------------------------------------- "
            f"Iteration{n:6d}(   1)  "
            f"---------------------------------------\n\n"
        )

        if sv is not None:
            XX, YY, ZZ, XY, YZ, ZX = _stress_kB(sv)
            tXX, tYY, tZZ = -sv[0] * vol, -sv[1] * vol, -sv[2] * vol
            tXY, tYZ, tZX = -sv[5] * vol, -sv[3] * vol, -sv[4] * vol
            ext_p = (XX + YY + ZZ) / 3.0
            f.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
            f.write(
                "  Direction    XX          YY          ZZ"
                "          XY          YZ          ZX\n"
            )
            f.write("  " + "-" * 86 + "\n")
            f.write(
                f"  Total   {tXX:11.5f} {tYY:11.5f} {tZZ:11.5f}"
                f" {tXY:11.5f} {tYZ:11.5f} {tZX:11.5f}\n"
            )
            f.write(
                f"  in kB   {XX:11.5f} {YY:11.5f} {ZZ:11.5f}"
                f" {XY:11.5f} {YZ:11.5f} {ZX:11.5f}\n"
            )
            f.write(
                f"  external pressure ={ext_p:12.2f} kB"
                f"  Pullay stress =        0.00 kB\n\n"
            )

        f.write(" VOLUME and BASIS-vectors are now :\n")
        f.write(" " + "-" * 77 + "\n")
        f.write(f"  volume of cell :{vol:12.2f}\n")
        f.write(
            "      direct lattice vectors"
            "                 reciprocal lattice vectors\n"
        )
        for i in range(3):
            f.write(
                f"  {cell[i,0]:12.9f} {cell[i,1]:12.9f} {cell[i,2]:12.9f}"
                f"   {rec_b[i,0]:12.9f} {rec_b[i,1]:12.9f} {rec_b[i,2]:12.9f}\n"
            )
        f.write("\n  length of vectors\n")
        f.write(
            f"  {lens_d[0]:12.9f} {lens_d[1]:12.9f} {lens_d[2]:12.9f}"
            f"   {lens_r[0]:12.9f} {lens_r[1]:12.9f} {lens_r[2]:12.9f}\n\n"
        )

        f.write(
            " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        )
        f.write(" " + "-" * 83 + "\n")
        for p, fv in zip(pos, frc):
            f.write(
                f" {p[0]:12.5f} {p[1]:12.5f} {p[2]:12.5f}"
                f"    {fv[0]:13.6f} {fv[1]:13.6f} {fv[2]:13.6f}\n"
            )
        f.write(" " + "-" * 83 + "\n")
        drift = np.sum(frc, axis=0)
        f.write(
            f"    total drift:                               "
            f"{drift[0]:13.6f} {drift[1]:13.6f} {drift[2]:13.6f}\n\n"
        )

        e_tot = energy_pot + energy_kin
        f.write("\n  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
        f.write("  ---------------------------------------------------\n")
        f.write(f"  free  energy   TOTEN  =     {energy_pot:20.8f} eV\n\n")
        f.write(
            f"  energy  without entropy=    {energy_pot:20.8f}"
            f"  energy(sigma->0) =    {energy_pot:20.8f}\n\n"
        )
        f.write(f"  kinetic Energy EKIN   =     {energy_kin:20.8f} eV\n")
        f.write(
            f"  total energy   ETOTAL =     {e_tot:20.8f} eV"
            f"  temperature  {temperature:10.2f} K\n\n"
        )
        f.write(" " + "-" * 102 + "\n")
