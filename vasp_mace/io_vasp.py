from typing import List, Optional
import xml.etree.ElementTree as ET
from .types_ import StepRecord
from .logging_utils import StepRecord
import numpy as np
from ase.constraints import FixCartesian
from ase.io import read, write

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
                # existing can be a single constraint or a list
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
    # ASE’s VASP writer will include the “Selective dynamics” section if we pass selective_dynamics=True
    # and atoms.arrays['selective_dynamics'] exists.
    write(
        path,
        atoms,
        format="vasp",
        direct=True,
        vasp5=True,
        sort=False,
        selective_dynamics=("selective_dynamics" in atoms.arrays),
        # ignore_constraints=False ensures constraints don’t get stripped silently
        ignore_constraints=False,
    )


def write_oszicar(path: str, steps: List[StepRecord]) -> None:
    """
    Minimal VASP-like OSZICAR: one line per ionic step.
    Columns: step, E (eV), dE (eV), Fmax (eV/Å)
    """
    with open(path, "w") as f:
        for s in steps:
            # VASP prints more fields; we keep it compact and stable for parsing
            f.write(f"  {s.n:4d}  E = {s.energy: .9f}  dE = {s.dE: .3e}  Fmax = {s.fmax: .4f}\n")


def write_outcar_like(path: str, atoms, steps: List[StepRecord], stress: Optional[list]) -> None:
    with open(path, "w") as f:
        f.write(" VASP pseudo-OUTCAR (MACE/ASE-generated, minimal)\n")
        f.write(f" Number of ions     NIONS = {len(atoms)}\n")
        a = atoms.get_cell()
        f.write(" Lattice vectors (Å):\n")
        for i in range(3): f.write(f"  {a[i,0]: .9f} {a[i,1]: .9f} {a[i,2]: .9f}\n")
        if stress is not None:
            s = stress
            f.write(" Stress tensor (eV/Å^3) (Voigt):\n")
            f.write(f"  {s[0]: .6e} {s[1]: .6e} {s[2]: .6e} {s[3]: .6e} {s[4]: .6e} {s[5]: .6e}\n")
        f.write("\n Energies per ionic step:\n")
        for s in steps:
            f.write(f"  step {s.n:4d} : E = {s.energy: .9f}  dE = {s.dE: .3e}  Fmax = {s.fmax: .4f}\n")
        f.write("\n POSITION (Å)              TOTAL-FORCE (eV/Å)\n")
        for pos, frc, sym in zip(atoms.get_positions(), atoms.get_forces(), atoms.get_chemical_symbols()):
            f.write(f" {pos[0]: .9f} {pos[1]: .9f} {pos[2]: .9f}    {frc[0]: .6f} {frc[1]: .6f} {frc[2]: .6f}  {sym}\n")

def write_contcar(path: str, atoms) -> None:
    write(path, atoms, format="vasp", direct=True, vasp5=True, sort=True)

def write_relax_vasprun_xml(path: str, atoms, steps: List[StepRecord]) -> None:
    root = ET.Element("vasprun", attrib={"version":"mace-ase-lite"})
    calc = ET.SubElement(root, "calculation")
    scsteps = ET.SubElement(calc, "scstep_list")
    for s in steps:
        sc = ET.SubElement(scsteps, "scstep", attrib={"number": str(s.n)})
        ET.SubElement(sc, "energy").text = f"{s.energy:.10f}"
        ET.SubElement(sc, "dE").text = f"{s.dE:.6e}"
        ET.SubElement(sc, "fmax").text = f"{s.fmax:.6f}"
    struct = ET.SubElement(calc, "structure", attrib={"name":"final"})
    latt = ET.SubElement(struct, "crystal")
    for v in atoms.cell:
        vec = ET.SubElement(latt, "varray", attrib={"name":"basis"})
        vec.text = f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}"
    pos = ET.SubElement(struct, "positions")
    for p in atoms.get_positions():
        ET.SubElement(pos, "v").text = f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_single_vasprun_xml(path: str, atoms, forces, stress=None, energy=None) -> None:
    root = ET.Element("modeling")
    calc = ET.SubElement(root, "calculation")

    struct = ET.SubElement(calc, "structure", attrib={"name": "initialpos"})
    cryst = ET.SubElement(struct, "crystal")
    basis = ET.SubElement(cryst, "varray", attrib={"name": "basis"})
    for v in atoms.cell:
        ET.SubElement(basis, "v").text = f"{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}"
    pos = ET.SubElement(struct, "varray", attrib={"name": "positions", "coord": "direct"})
    for p in atoms.get_scaled_positions():
        ET.SubElement(pos, "v").text = f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e}"

    farr = ET.SubElement(calc, "varray", attrib={"name": "forces"})
    for f in forces:
        ET.SubElement(farr, "v").text = f"{float(f[0]):.10e} {float(f[1]):.10e} {float(f[2]):.10e}"

    if stress is not None:
        sarr = ET.SubElement(calc, "varray", attrib={"name": "stress"})
        ET.SubElement(sarr, "v").text = " ".join(f"{float(x):.10e}" for x in stress)

    if energy is not None:
        en = ET.SubElement(calc, "energy")
        ET.SubElement(en, "i", attrib={"name": "e_fr_energy"}).text = f"{float(energy):.10e}"

    tree = ET.ElementTree(root)
    try: ET.indent(tree, space="  ")
    except Exception: pass
    tree.write(path, encoding="utf-8", xml_declaration=True)
