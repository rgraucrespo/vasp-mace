from typing import List, Optional
import xml.etree.ElementTree as ET
import numpy as np
from ase.constraints import FixCartesian
from ase.io import read, write

from .logging_utils import StepRecord

# -----------------------------------------------------------------------
# Unit conversion
# -----------------------------------------------------------------------
_EV_A3_TO_KB = 1602.1766   # 1 eV/Å³  →  kBar


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _fmt_fort(val: float, ndigits: int = 8) -> str:
    """Fortran-style scientific notation: sign + '.' + ndigits + 'E' + ±XX.
    e.g. -47.1299506 → '-.47129951E+02'  (ndigits=8)
    """
    if val == 0.0:
        return f" .{'0' * ndigits}E+00"
    sign = '-' if val < 0 else ' '
    mag = abs(val)
    exp = int(np.floor(np.log10(mag))) + 1
    raw = mag * 10 ** (ndigits - exp)
    mantissa_int = round(raw)
    # carry propagation
    if mantissa_int >= 10 ** ndigits:
        mantissa_int //= 10
        exp += 1
    return f"{sign}.{mantissa_int:0{ndigits}d}E{exp:+03d}"


def _stress_kB(stress_voigt):
    """Convert ASE Voigt stress (eV/Å³, tensile-positive) to VASP kBar components
    (compressive-positive). Returns (XX, YY, ZZ, XY, YZ, ZX)."""
    s = stress_voigt
    c = _EV_A3_TO_KB
    # ASE Voigt: [xx, yy, zz, yz, xz, xy]; VASP order: XX YY ZZ XY YZ ZX
    return (-s[0]*c, -s[1]*c, -s[2]*c, -s[5]*c, -s[3]*c, -s[4]*c)


def _stress_matrix_kB(stress_voigt):
    """Return 3×3 stress matrix in kBar for vasprun.xml."""
    XX, YY, ZZ, XY, YZ, ZX = _stress_kB(stress_voigt)
    return [
        [XX, XY, ZX],
        [XY, YY, YZ],
        [ZX, YZ, ZZ],
    ]


def _rec_basis(cell):
    """Reciprocal lattice (rows), VASP convention: G·a = δ  (no 2π)."""
    return np.linalg.inv(cell).T


def _incar_xml_type_val(v_str: str):
    """Guess VASP vasprun.xml type tag and formatted value from raw INCAR string."""
    v = v_str.strip()
    if v.upper() in ('.TRUE.', '.FALSE.', 'T', 'F'):
        logical = 'T' if v.upper() in ('.TRUE.', 'T') else 'F'
        return 'logical', f' {logical}  '
    try:
        return 'int', f'   {int(v)}'
    except ValueError:
        pass
    try:
        fval = float(v)
        return None, f'      {fval:.8f}'
    except ValueError:
        pass
    return 'string', v


# -----------------------------------------------------------------------
# POSCAR / CONTCAR
# -----------------------------------------------------------------------

def read_poscar(path: str = "POSCAR", apply_selective_dynamics: bool = True):
    """
    Read POSCAR/CONTCAR with ASE.
    If 'Selective dynamics' flags are present and apply_selective_dynamics=True,
    convert them into ASE FixCartesian constraints so relaxations respect them.
    """
    atoms = read(path)  # ASE handles VASP POSCAR/CONTCAR

    if apply_selective_dynamics and "selective_dynamics" in atoms.arrays:
        sd = np.asarray(atoms.arrays["selective_dynamics"], dtype=bool)  # shape (N,3), T=free, F=fixed
        constraints = []

        # Build per-atom FixCartesian constraints for components marked F
        for i, flags in enumerate(sd):
            fixed_mask = ~flags  # True where component is FIXED
            if fixed_mask.any():
                constraints.append(FixCartesian(fixed_mask, indices=[i]))

        # Merge with any pre-existing constraints
        if constraints:
            existing = atoms.constraints
            if existing is None:
                atoms.set_constraint(constraints)
            else:
                if isinstance(existing, (list, tuple)):
                    atoms.set_constraint(list(existing) + constraints)
                else:
                    atoms.set_constraint([existing] + constraints)

        # Keep the SD flags so we can round-trip to CONTCAR
        atoms.arrays["selective_dynamics"] = sd

    return atoms


def write_contcar(path: str, atoms):
    """
    Write a VASP-style CONTCAR, preserving Selective Dynamics flags if present.
    """
    write(
        path,
        atoms,
        format="vasp",
        direct=True,
        vasp5=True,
        sort=False,
        ignore_constraints=False,
    )


# -----------------------------------------------------------------------
# OSZICAR
# -----------------------------------------------------------------------

def write_oszicar(path: str, steps: List[StepRecord]) -> None:
    """VASP-style OSZICAR: one ionic-step line per step.

    Format (per real VASP):
        N F= -.XXXXXXXE+XX E0= -.XXXXXXXE+XX  d E =-.XXXXXXE+XX
    F = free energy (= E0 for MACE, no entropy).
    d E for step 1 is the energy itself (VASP convention).
    """
    with open(path, "w") as f:
        for s in steps:
            E_str  = _fmt_fort(s.energy, 8)
            dE_val = s.energy if s.n == 1 else s.dE
            dE_str = _fmt_fort(dE_val, 6)
            f.write(f"   {s.n:3d} F= {E_str} E0= {E_str}  d E ={dE_str}\n")


# -----------------------------------------------------------------------
# OUTCAR
# -----------------------------------------------------------------------

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
    symbols = atoms.get_chemical_symbols()
    nions = len(atoms)

    with open(path, "w") as f:

        # ---- Header ----
        f.write(" vasp-mace (MACE/ASE interface) -- VASP-compatible OUTCAR\n")
        f.write(f" NIONS =   {nions}\n")
        f.write("\n")

        # ---- INCAR echo ----
        if incar_raw:
            f.write(" INCAR:\n")
            for k, v in incar_raw.items():
                f.write(f"   {k} = {v}\n")
            f.write("\n")

        # ---- Species summary ----
        f.write(f" ions per type = ")
        seen = {}
        counts = []
        for sym in symbols:
            if sym not in seen:
                seen[sym] = len(seen)
                counts.append(0)
            counts[seen[sym]] += 1
        f.write("  ".join(str(c) for c in counts) + "\n")
        f.write(" POMASS = " + "  ".join(f"{atoms.get_masses()[symbols.index(sp)]:.3f}"
                                         for sp in seen) + "\n\n")
        f.write(" " + "-" * 102 + "\n\n")

        # ---- Per-ionic-step blocks ----
        for rec in steps:
            cell = rec.cell if rec.cell is not None else np.array(atoms.get_cell())
            pos  = rec.positions if rec.positions is not None else atoms.get_positions()
            frc  = rec.forces    if rec.forces    is not None else atoms.get_forces()
            sv   = rec.stress    # may be None

            vol = float(np.linalg.det(cell))

            # Separator
            f.write(f"\n --------------------------------------- "
                    f"Iteration{rec.n:6d}(   1)  "
                    f"---------------------------------------\n\n")

            # -- Stress / Force on cell --
            if sv is not None:
                XX, YY, ZZ, XY, YZ, ZX = _stress_kB(sv)
                # Total = stress * volume (eV, compressive positive)
                tXX, tYY, tZZ = -sv[0]*vol, -sv[1]*vol, -sv[2]*vol
                tXY, tYZ, tZX = -sv[5]*vol, -sv[3]*vol, -sv[4]*vol
                ext_p = (XX + YY + ZZ) / 3.0

                f.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
                f.write("  Direction    XX          YY          ZZ"
                        "          XY          YZ          ZX\n")
                f.write("  " + "-" * 86 + "\n")
                f.write(f"  Total   {tXX:11.5f} {tYY:11.5f} {tZZ:11.5f}"
                        f" {tXY:11.5f} {tYZ:11.5f} {tZX:11.5f}\n")
                f.write(f"  in kB   {XX:11.5f} {YY:11.5f} {ZZ:11.5f}"
                        f" {XY:11.5f} {YZ:11.5f} {ZX:11.5f}\n")
                f.write(f"  external pressure ={ext_p:12.2f} kB"
                        f"  Pullay stress =        0.00 kB\n\n")

            # -- Volume and basis vectors --
            rec_b = _rec_basis(cell)
            lens_d = np.linalg.norm(cell, axis=1)
            lens_r = np.linalg.norm(rec_b, axis=1)
            f.write(" VOLUME and BASIS-vectors are now :\n")
            f.write(" " + "-" * 77 + "\n")
            f.write(f"  volume of cell :{vol:12.2f}\n")
            f.write("      direct lattice vectors"
                    "                 reciprocal lattice vectors\n")
            for i in range(3):
                f.write(f"  {cell[i,0]:12.9f} {cell[i,1]:12.9f} {cell[i,2]:12.9f}"
                        f"   {rec_b[i,0]:12.9f} {rec_b[i,1]:12.9f} {rec_b[i,2]:12.9f}\n")
            f.write("\n  length of vectors\n")
            f.write(f"  {lens_d[0]:12.9f} {lens_d[1]:12.9f} {lens_d[2]:12.9f}"
                    f"   {lens_r[0]:12.9f} {lens_r[1]:12.9f} {lens_r[2]:12.9f}\n\n")

            # -- Position / force --
            f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write(" " + "-" * 83 + "\n")
            for p, fv in zip(pos, frc):
                f.write(f" {p[0]:12.5f} {p[1]:12.5f} {p[2]:12.5f}"
                        f"    {fv[0]:13.6f} {fv[1]:13.6f} {fv[2]:13.6f}\n")
            f.write(" " + "-" * 83 + "\n")
            drift = np.sum(frc, axis=0)
            f.write(f"    total drift:                               "
                    f"{drift[0]:13.6f} {drift[1]:13.6f} {drift[2]:13.6f}\n\n")

            # -- Energy --
            f.write("\n  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
            f.write("  ---------------------------------------------------\n")
            f.write(f"  free  energy   TOTEN  =     {rec.energy:20.8f} eV\n\n")
            f.write(f"  energy  without entropy=    {rec.energy:20.8f}"
                    f"  energy(sigma->0) =    {rec.energy:20.8f}\n\n")

            f.write(" " + "-" * 102 + "\n")

        # ---- Convergence message ----
        if converged:
            f.write("\n reached required accuracy"
                    " - stopping structural energy minimisation\n")
        else:
            f.write("\n maximum ionic steps (NSW) exceeded\n")

        # ---- Timing ----
        cpu = cpu_time if cpu_time is not None else elapsed
        f.write("\n\n General timing and accounting informations for this job:\n")
        f.write(" ========================================================\n\n")
        f.write(f"                  Total CPU time used (sec):{cpu:13.3f}\n")
        f.write(f"                         Elapsed time (sec):{elapsed:13.3f}\n\n")
        f.write(f"                   Maximum memory used (kb):          N/A\n")
        f.write(f"                   Average memory used (kb):          N/A\n")


# Backward-compatible alias used for single-point in cli.py
def write_outcar_like(path, atoms, steps, stress=None,
                      incar_raw=None, converged=True, elapsed=0.0, cpu_time=None):
    """Thin wrapper kept for backward compatibility; delegates to write_outcar."""
    # For single-point SimpleNamespace records that lack the new fields,
    # upgrade them if a stress array was passed separately.
    upgraded = []
    for s in steps:
        if isinstance(s, StepRecord):
            upgraded.append(s)
        else:
            # Wrap SimpleNamespace → StepRecord
            sv = np.array(stress) if stress is not None else None
            upgraded.append(StepRecord(
                n=s.n, energy=s.energy, dE=s.dE, fmax=s.fmax,
                positions=atoms.get_positions().copy(),
                forces=atoms.get_forces().copy() if hasattr(atoms, 'get_forces') else None,
                stress=sv,
                cell=np.array(atoms.get_cell()).copy(),
            ))
    write_outcar(path, atoms, upgraded, incar_raw=incar_raw,
                 converged=converged, elapsed=elapsed, cpu_time=cpu_time)


# -----------------------------------------------------------------------
# vasprun.xml — relaxation
# -----------------------------------------------------------------------

def _xml_crystal_block(parent, cell):
    """Append <crystal> with basis, volume, rec_basis to *parent*."""
    rec_b = _rec_basis(cell)
    vol = float(np.linalg.det(cell))

    cryst = ET.SubElement(parent, "crystal")
    basis = ET.SubElement(cryst, "varray", attrib={"name": "basis"})
    for v in cell:
        ET.SubElement(basis, "v").text = f"  {v[0]:16.8f} {v[1]:16.8f} {v[2]:16.8f} "
    ET.SubElement(cryst, "i", attrib={"name": "volume"}).text = f"  {vol:16.8f} "
    rec = ET.SubElement(cryst, "varray", attrib={"name": "rec_basis"})
    for v in rec_b:
        ET.SubElement(rec, "v").text = f"  {v[0]:16.8f} {v[1]:16.8f} {v[2]:16.8f} "
    return cryst


def _xml_positions_block(parent, cell, positions):
    """Append <varray name='positions'> with fractional coords to *parent*."""
    cell_inv = np.linalg.inv(cell)
    frac = positions @ cell_inv
    pv = ET.SubElement(parent, "varray", attrib={"name": "positions"})
    for p in frac:
        ET.SubElement(pv, "v").text = f"  {p[0]:16.8f} {p[1]:16.8f} {p[2]:16.8f} "


def _xml_energy_block(parent, energy):
    """Append <energy> with e_fr_energy / e_wo_entrp / e_0_energy (all equal for MACE)."""
    en = ET.SubElement(parent, "energy")
    for name in ("e_fr_energy", "e_wo_entrp", "e_0_energy"):
        ET.SubElement(en, "i", attrib={"name": name}).text = f"  {energy:20.8f} "


def write_relax_vasprun_xml(
    path: str,
    atoms_initial,
    atoms,
    steps: List[StepRecord],
    incar_raw: Optional[dict] = None,
) -> None:
    """Write a VASP-compatible vasprun.xml for a relaxation run.

    Root is <modeling> with <generator>, <incar>, <atominfo>,
    <structure name='initialpos'>, per-step <calculation>, and
    <structure name='finalpos'>.
    """
    import datetime

    root = ET.Element("modeling")

    # ---- generator ----
    gen = ET.SubElement(root, "generator")
    now = datetime.datetime.now()
    for name, typ, val in [
        ("program",    "string", "vasp "),
        ("version",    "string", "mace-ase  "),
        ("platform",   "string", "mace-ase"),
        ("date",       "string", now.strftime("%Y %m %d")),
        ("time",       "string", now.strftime("%H:%M:%S")),
    ]:
        ET.SubElement(gen, "i", attrib={"name": name, "type": typ}).text = val

    # ---- incar ----
    # (also add a <parameters> block so pymatgen doesn't fail on missing attribute)
    if incar_raw:
        incar_el = ET.SubElement(root, "incar")
        for k, v_str in incar_raw.items():
            typ, val = _incar_xml_type_val(v_str)
            attrib = {"name": k}
            if typ is not None:
                attrib["type"] = typ
            ET.SubElement(incar_el, "i", attrib=attrib).text = val

    # ---- parameters (pymatgen needs at least NELM, IBRION, NSW, EDIFFG) ----
    raw = incar_raw or {}
    # Helper to get int from raw or default
    def _ri(key, default):
        try: return int(raw.get(key, default))
        except: return default
    def _rf(key, default):
        try: return float(raw.get(key, default))
        except: return default

    params_el = ET.SubElement(root, "parameters")
    sep_elec = ET.SubElement(params_el, "separator", attrib={"name": "electronic"})
    ET.SubElement(sep_elec, "i", attrib={"type": "int",     "name": "NELM"}).text    = f"   {_ri('NELM', 60)}"
    ET.SubElement(sep_elec, "i", attrib={"type": "logical", "name": "LCHIMAG"}).text = " F  "
    ET.SubElement(sep_elec, "i", attrib={               "name": "EDIFF"}).text       = f"      {_rf('EDIFF', 1e-4):.8f}"
    sep_ion  = ET.SubElement(params_el, "separator", attrib={"name": "ionic"})
    ET.SubElement(sep_ion,  "i", attrib={"type": "int",     "name": "NSW"}).text     = f"   {_ri('NSW', 0)}"
    ET.SubElement(sep_ion,  "i", attrib={"type": "int",     "name": "IBRION"}).text  = f"   {_ri('IBRION', -1)}"
    ET.SubElement(sep_ion,  "i", attrib={"type": "int",     "name": "ISIF"}).text    = f"   {_ri('ISIF', 2)}"
    ET.SubElement(sep_ion,  "i", attrib={               "name": "EDIFFG"}).text      = f"      {_rf('EDIFFG', -0.05):.8f}"
    ET.SubElement(sep_ion,  "i", attrib={"type": "int",     "name": "LORBIT"}).text  = f"   {_ri('LORBIT', 0)}"

    # ---- atominfo ----
    symbols = atoms.get_chemical_symbols()
    masses  = atoms.get_masses()

    # Build ordered species list preserving POSCAR order
    seen_sp = {}
    species_order = []
    for sym in symbols:
        if sym not in seen_sp:
            seen_sp[sym] = len(seen_sp)
            species_order.append(sym)

    type_counts = [0] * len(species_order)
    type_masses = {}
    for sym, m in zip(symbols, masses):
        type_counts[seen_sp[sym]] += 1
        if sym not in type_masses:
            type_masses[sym] = m

    ainfo = ET.SubElement(root, "atominfo")
    ET.SubElement(ainfo, "atoms").text = f"  {len(symbols):5d} "
    ET.SubElement(ainfo, "types").text = f"  {len(species_order):5d} "

    arr_atoms = ET.SubElement(ainfo, "array", attrib={"name": "atoms"})
    ET.SubElement(arr_atoms, "dimension", attrib={"dim": "1"}).text = "ion"
    ET.SubElement(arr_atoms, "field", attrib={"type": "string"}).text = "element"
    ET.SubElement(arr_atoms, "field", attrib={"type": "int"}).text = "atomtype"
    aset = ET.SubElement(arr_atoms, "set")
    for sym in symbols:
        rc = ET.SubElement(aset, "rc")
        ET.SubElement(rc, "c").text = sym
        ET.SubElement(rc, "c").text = f"   {seen_sp[sym]+1}"

    arr_types = ET.SubElement(ainfo, "array", attrib={"name": "atomtypes"})
    ET.SubElement(arr_types, "dimension", attrib={"dim": "1"}).text = "type"
    ET.SubElement(arr_types, "field", attrib={"type": "int"}).text = "atomspertype"
    ET.SubElement(arr_types, "field", attrib={"type": "string"}).text = "element"
    ET.SubElement(arr_types, "field").text = "mass"
    ET.SubElement(arr_types, "field").text = "valence"
    ET.SubElement(arr_types, "field", attrib={"type": "string"}).text = "pseudopotential"
    tset = ET.SubElement(arr_types, "set")
    for sp in species_order:
        rc = ET.SubElement(tset, "rc")
        ET.SubElement(rc, "c").text = f"  {type_counts[seen_sp[sp]]:5d}"
        ET.SubElement(rc, "c").text = sp
        ET.SubElement(rc, "c").text = f"  {type_masses[sp]:16.8f}"
        ET.SubElement(rc, "c").text = f"       0.00000000"   # valence (not tracked by MACE)
        ET.SubElement(rc, "c").text = f"  PAW_PBE {sp} (MACE)                    "

    # ---- initialpos ----
    init_cell = (steps[0].cell if steps and steps[0].cell is not None
                 else np.array(atoms_initial.get_cell()))
    init_pos  = (steps[0].positions if steps and steps[0].positions is not None
                 else atoms_initial.get_positions())

    initpos = ET.SubElement(root, "structure", attrib={"name": "initialpos"})
    _xml_crystal_block(initpos, init_cell)
    _xml_positions_block(initpos, init_cell, init_pos)

    # ---- per-step calculations ----
    for rec in steps:
        cell = rec.cell if rec.cell is not None else np.array(atoms.get_cell())
        pos  = rec.positions if rec.positions is not None else atoms.get_positions()
        frc  = rec.forces if rec.forces is not None else atoms.get_forces()
        sv   = rec.stress

        calc = ET.SubElement(root, "calculation")

        # scstep (one MACE evaluation = one "electronic step")
        sc = ET.SubElement(calc, "scstep")
        _xml_energy_block(sc, rec.energy)

        # structure
        struct = ET.SubElement(calc, "structure")
        _xml_crystal_block(struct, cell)
        _xml_positions_block(struct, cell, pos)

        # forces
        fv_el = ET.SubElement(calc, "varray", attrib={"name": "forces"})
        for fv in frc:
            ET.SubElement(fv_el, "v").text = (f"  {fv[0]:16.8f}"
                                               f" {fv[1]:16.8f}"
                                               f" {fv[2]:16.8f} ")

        # stress (3×3, kB, compressive-positive)
        if sv is not None:
            mat = _stress_matrix_kB(sv)
            sv_el = ET.SubElement(calc, "varray", attrib={"name": "stress"})
            for row in mat:
                ET.SubElement(sv_el, "v").text = (f"  {row[0]:16.8f}"
                                                   f" {row[1]:16.8f}"
                                                   f" {row[2]:16.8f} ")

        # energy summary at end of calculation block
        _xml_energy_block(calc, rec.energy)

    # ---- finalpos ----
    final_cell = np.array(atoms.get_cell())
    final_pos  = atoms.get_positions()
    finpos = ET.SubElement(root, "structure", attrib={"name": "finalpos"})
    _xml_crystal_block(finpos, final_cell)
    _xml_positions_block(finpos, final_cell, final_pos)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space=" ")
    except Exception:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)


# -----------------------------------------------------------------------
# vasprun.xml — single-point
# -----------------------------------------------------------------------

def write_single_vasprun_xml(path: str, atoms, forces, stress=None, energy=None) -> None:
    """Write a minimal VASP-compatible vasprun.xml for a single-point calculation."""
    root = ET.Element("modeling")
    calc = ET.SubElement(root, "calculation")

    cell = np.array(atoms.get_cell())
    struct = ET.SubElement(calc, "structure", attrib={"name": "initialpos"})
    _xml_crystal_block(struct, cell)
    _xml_positions_block(struct, cell, atoms.get_positions())

    farr = ET.SubElement(calc, "varray", attrib={"name": "forces"})
    for fv in np.asarray(forces, dtype=float):
        ET.SubElement(farr, "v").text = (f"  {float(fv[0]):.10e}"
                                          f" {float(fv[1]):.10e}"
                                          f" {float(fv[2]):.10e} ")

    if stress is not None:
        mat = _stress_matrix_kB(np.asarray(stress))
        sarr = ET.SubElement(calc, "varray", attrib={"name": "stress"})
        for row in mat:
            ET.SubElement(sarr, "v").text = (f"  {row[0]:16.8f}"
                                              f" {row[1]:16.8f}"
                                              f" {row[2]:16.8f} ")

    if energy is not None:
        _xml_energy_block(calc, float(energy))

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)


# -----------------------------------------------------------------------
# XDATCAR
# -----------------------------------------------------------------------

def write_xdatcar_header(path: str, atoms) -> None:
    """Write the XDATCAR header (system name + lattice vectors + species counts)."""
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
    with open(path, "w") as f:
        f.write(f"{atoms.get_chemical_formula()}\n")
        f.write("   1.00000000\n")
        for v in cell:
            f.write(f"  {v[0]: .9f}  {v[1]: .9f}  {v[2]: .9f}\n")
        f.write("  " + "  ".join(species) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")


def append_xdatcar_frame(path: str, atoms, step: int) -> None:
    """Append one frame of fractional coordinates to XDATCAR."""
    scaled = atoms.get_scaled_positions()
    with open(path, "a") as f:
        f.write(f"Direct configuration=     {step}\n")
        for pos in scaled:
            f.write(f"  {pos[0]: .9f}  {pos[1]: .9f}  {pos[2]: .9f}\n")
