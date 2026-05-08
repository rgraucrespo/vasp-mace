"""Write vasprun.xml files (relaxation and single-point)."""

import datetime
from typing import List, Optional
import xml.etree.ElementTree as ET

import numpy as np

from .logging_utils import StepRecord

_EV_A3_TO_KB = 1602.1766  # 1 eV/Å³ → kBar


def _rec_basis(cell):
    """Reciprocal lattice (rows), VASP convention: G·a = δ  (no 2π)."""
    return np.linalg.inv(cell).T


def _stress_matrix_kB(stress_voigt):
    """Return 3×3 stress matrix in kBar for vasprun.xml."""
    s = stress_voigt
    c = _EV_A3_TO_KB
    XX, YY, ZZ = -s[0] * c, -s[1] * c, -s[2] * c
    XY, YZ, ZX = -s[5] * c, -s[3] * c, -s[4] * c
    return [[XX, XY, ZX], [XY, YY, YZ], [ZX, YZ, ZZ]]


# ---------------------------------------------------------------------------
# INCAR parsing helpers (XML type inference)
# ---------------------------------------------------------------------------


def _ri(raw: dict, key: str, default: int) -> int:
    """Read an integer INCAR value with a fallback default."""
    try:
        return int(raw.get(key, default))
    except (ValueError, TypeError):
        return default


def _rf(raw: dict, key: str, default: float) -> float:
    """Read a float INCAR value with a fallback default."""
    try:
        return float(raw.get(key, default))
    except (ValueError, TypeError):
        return default


def _incar_xml_type_val(v_str: str):
    """Guess VASP vasprun.xml type tag and formatted value from raw INCAR string."""
    v = v_str.strip()
    if v.upper() in (".TRUE.", ".FALSE.", "T", "F"):
        logical = "T" if v.upper() in (".TRUE.", "T") else "F"
        return "logical", f" {logical}  "
    try:
        return "int", f"   {int(v)}"
    except ValueError:
        pass
    try:
        fval = float(v)
        return None, f"      {fval:.8f}"
    except ValueError:
        pass
    return "string", v


# ---------------------------------------------------------------------------
# XML building blocks
# ---------------------------------------------------------------------------


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


def _xml_generator(root):
    """Append <generator> block to root."""
    gen = ET.SubElement(root, "generator")
    now = datetime.datetime.now()
    for name, typ, val in [
        ("program", "string", "vasp "),
        ("version", "string", "mace-ase  "),
        ("platform", "string", "mace-ase"),
        ("date", "string", now.strftime("%Y %m %d")),
        ("time", "string", now.strftime("%H:%M:%S")),
    ]:
        ET.SubElement(gen, "i", attrib={"name": name, "type": typ}).text = val


def _xml_incar(root, incar_raw):
    """Append <incar> block to root."""
    if not incar_raw:
        return
    incar_el = ET.SubElement(root, "incar")
    for k, v_str in incar_raw.items():
        typ, val = _incar_xml_type_val(v_str)
        attrib = {"name": k}
        if typ is not None:
            attrib["type"] = typ
        ET.SubElement(incar_el, "i", attrib=attrib).text = val


def _xml_parameters(root, incar_raw):
    """Append <parameters> block (pymatgen needs at least NELM, IBRION, NSW, EDIFFG)."""
    raw = incar_raw or {}
    params_el = ET.SubElement(root, "parameters")
    sep_elec = ET.SubElement(params_el, "separator", attrib={"name": "electronic"})
    ET.SubElement(sep_elec, "i", attrib={"type": "int", "name": "NELM"}).text = (
        f"   {_ri(raw, 'NELM', 60)}"
    )
    ET.SubElement(sep_elec, "i", attrib={"type": "logical", "name": "LCHIMAG"}).text = (
        " F  "
    )
    ET.SubElement(sep_elec, "i", attrib={"name": "EDIFF"}).text = (
        f"      {_rf(raw, 'EDIFF', 1e-4):.8f}"
    )
    sep_ion = ET.SubElement(params_el, "separator", attrib={"name": "ionic"})
    ET.SubElement(sep_ion, "i", attrib={"type": "int", "name": "NSW"}).text = (
        f"   {_ri(raw, 'NSW', 0)}"
    )
    ET.SubElement(sep_ion, "i", attrib={"type": "int", "name": "IBRION"}).text = (
        f"   {_ri(raw, 'IBRION', -1)}"
    )
    ET.SubElement(sep_ion, "i", attrib={"type": "int", "name": "ISIF"}).text = (
        f"   {_ri(raw, 'ISIF', 2)}"
    )
    ET.SubElement(sep_ion, "i", attrib={"name": "EDIFFG"}).text = (
        f"      {_rf(raw, 'EDIFFG', -0.05):.8f}"
    )
    ET.SubElement(sep_ion, "i", attrib={"type": "int", "name": "LORBIT"}).text = (
        f"   {_ri(raw, 'LORBIT', 0)}"
    )


def _xml_atominfo(root, atoms):
    """Append <atominfo> block to root. Returns (seen_sp, species_order)."""
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()

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
    ET.SubElement(arr_types, "field", attrib={"type": "string"}).text = (
        "pseudopotential"
    )
    tset = ET.SubElement(arr_types, "set")
    for sp in species_order:
        rc = ET.SubElement(tset, "rc")
        ET.SubElement(rc, "c").text = f"  {type_counts[seen_sp[sp]]:5d}"
        ET.SubElement(rc, "c").text = sp
        ET.SubElement(rc, "c").text = f"  {type_masses[sp]:16.8f}"
        ET.SubElement(rc, "c").text = "       0.00000000"
        ET.SubElement(rc, "c").text = f"  PAW_PBE {sp} (MACE)                    "

    return seen_sp, species_order


def _xml_write(tree, path):
    """Write an ElementTree to *path*, indenting if possible."""
    try:
        ET.indent(tree, space=" ")
    except Exception:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Public writers
# ---------------------------------------------------------------------------


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
    root = ET.Element("modeling")
    _xml_generator(root)
    _xml_incar(root, incar_raw)
    _xml_parameters(root, incar_raw)
    _xml_atominfo(root, atoms)

    init_cell = (
        steps[0].cell
        if steps and steps[0].cell is not None
        else np.array(atoms_initial.get_cell())
    )
    init_pos = (
        steps[0].positions
        if steps and steps[0].positions is not None
        else atoms_initial.get_positions()
    )
    initpos = ET.SubElement(root, "structure", attrib={"name": "initialpos"})
    _xml_crystal_block(initpos, init_cell)
    _xml_positions_block(initpos, init_cell, init_pos)

    for rec in steps:
        cell = rec.cell if rec.cell is not None else np.array(atoms.get_cell())
        pos = rec.positions if rec.positions is not None else atoms.get_positions()
        frc = rec.forces if rec.forces is not None else atoms.get_forces()
        sv = rec.stress

        calc = ET.SubElement(root, "calculation")
        sc = ET.SubElement(calc, "scstep")
        _xml_energy_block(sc, rec.energy)

        struct = ET.SubElement(calc, "structure")
        _xml_crystal_block(struct, cell)
        _xml_positions_block(struct, cell, pos)

        fv_el = ET.SubElement(calc, "varray", attrib={"name": "forces"})
        for fv in frc:
            ET.SubElement(fv_el, "v").text = (
                f"  {fv[0]:16.8f} {fv[1]:16.8f} {fv[2]:16.8f} "
            )

        if sv is not None:
            mat = _stress_matrix_kB(sv)
            sv_el = ET.SubElement(calc, "varray", attrib={"name": "stress"})
            for row in mat:
                ET.SubElement(sv_el, "v").text = (
                    f"  {row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f} "
                )

        _xml_energy_block(calc, rec.energy)

    final_cell = np.array(atoms.get_cell())
    final_pos = atoms.get_positions()
    finpos = ET.SubElement(root, "structure", attrib={"name": "finalpos"})
    _xml_crystal_block(finpos, final_cell)
    _xml_positions_block(finpos, final_cell, final_pos)

    _xml_write(ET.ElementTree(root), path)


def write_single_vasprun_xml(
    path: str,
    atoms,
    forces,
    stress=None,
    energy=None,
    incar_raw: Optional[dict] = None,
) -> None:
    """Write a VASP-compatible vasprun.xml for a single-point calculation.

    Follows the same root-level structure as write_relax_vasprun_xml and the
    real VASP output: generator → incar → parameters → atominfo →
    structure[initialpos] → calculation → structure[finalpos].
    """
    root = ET.Element("modeling")
    _xml_generator(root)
    _xml_incar(root, incar_raw)
    _xml_parameters(root, incar_raw)
    _xml_atominfo(root, atoms)

    cell = np.array(atoms.get_cell())
    initpos = ET.SubElement(root, "structure", attrib={"name": "initialpos"})
    _xml_crystal_block(initpos, cell)
    _xml_positions_block(initpos, cell, atoms.get_positions())

    calc_el = ET.SubElement(root, "calculation")

    if energy is not None:
        sc = ET.SubElement(calc_el, "scstep")
        _xml_energy_block(sc, float(energy))

    struct = ET.SubElement(calc_el, "structure")
    _xml_crystal_block(struct, cell)
    _xml_positions_block(struct, cell, atoms.get_positions())

    farr = ET.SubElement(calc_el, "varray", attrib={"name": "forces"})
    for fv in np.asarray(forces, dtype=float):
        ET.SubElement(farr, "v").text = (
            f"  {float(fv[0]):16.8f} {float(fv[1]):16.8f} {float(fv[2]):16.8f} "
        )

    if stress is not None:
        mat = _stress_matrix_kB(np.asarray(stress))
        sarr = ET.SubElement(calc_el, "varray", attrib={"name": "stress"})
        for row in mat:
            ET.SubElement(sarr, "v").text = (
                f"  {row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f} "
            )

    if energy is not None:
        _xml_energy_block(calc_el, float(energy))

    finpos = ET.SubElement(root, "structure", attrib={"name": "finalpos"})
    _xml_crystal_block(finpos, cell)
    _xml_positions_block(finpos, cell, atoms.get_positions())

    _xml_write(ET.ElementTree(root), path)
