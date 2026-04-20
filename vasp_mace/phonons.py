import os
import math
import numpy as np
from ase import Atoms

from .io_vasp import write_contcar, write_xdatcar_header, append_xdatcar_frame

# ---------------------------------------------------------------------------
# Physical constants for phonon frequency conversion
# ---------------------------------------------------------------------------
# sqrt(eV / (Å² · amu))  →  THz    (angular frequency / 2π)
_THz_FACTOR = math.sqrt(1.60218e-19 / (1e-20 * 1.66054e-27)) / (2 * math.pi * 1e12)
_THz_TO_CM  = 33.3564   # cm⁻¹ per THz
_THz_TO_MEV =  4.13567  # meV   per THz


# ===========================================================================
# Public entry point
# ===========================================================================

def run_phonons(atoms: Atoms, calc, cfg) -> None:
    """
    IBRION=5: all N×3×NFREE Cartesian displacements (no symmetry).
    IBRION=6: symmetry-reduced displacements via phonopy; falls back to
              IBRION=5 if phonopy is not installed.

    Outputs (VASP-compatible):
      DYNMAT, XDATCAR, OSZICAR, OUTCAR, CONTCAR
      ase_files/force_constants.npy
      ase_files/phonopy_params.yaml   (IBRION=6 only)
    """
    os.makedirs("ase_files", exist_ok=True)

    if cfg.IBRION == 6:
        try:
            import phonopy  # noqa: F401
            _run_with_symmetry(atoms, calc, cfg)
            return
        except ImportError:
            print("[warn] phonopy not found; IBRION=6 symmetry reduction unavailable. "
                  "Install with `pip install phonopy`. "
                  "Falling back to IBRION=5 (all displacements).")

    _run_brute_force(atoms, calc, cfg)


# ===========================================================================
# IBRION=5: brute-force (all displacements)
# ===========================================================================

def _run_brute_force(atoms: Atoms, calc, cfg) -> None:
    delta = cfg.POTIM
    nfree = cfg.NFREE
    N     = len(atoms)
    n_disp_all = N * 3 * nfree   # total single-point calculations

    print(f"[info] Phonons: IBRION={cfg.IBRION}, POTIM={delta} Å, NFREE={nfree}")
    print(f"[info] {N} atoms × 3 directions × {nfree} = {n_disp_all} single-point calculations")
    if N > 150:
        print(f"[warn] {N} atoms → {n_disp_all} force evaluations. "
              "Phonon calculations work best on supercells of ~16–128 atoms.")

    # Reference calculation
    forces_eq = atoms.get_forces()
    energy_eq = atoms.get_potential_energy()
    _check_residual_forces(forces_eq)

    # Collect all displaced configurations
    # blocks_all: one per single-point calculation, in order
    blocks_all  = []   # (disp_atoms, disp_vec, forces, energy)
    count = 0
    for i in range(N):
        for alpha in range(3):
            for sign in ([+1, -1] if nfree == 2 else [+1]):
                count += 1
                disp_vec = np.zeros(3)
                disp_vec[alpha] = sign * delta
                sign_str = "+" if sign > 0 else "-"
                print(f"  [{count:4d}/{n_disp_all}] atom {i:4d}  {sign_str}{'xyz'[alpha]}", flush=True)
                disp_atoms, forces, energy = _displaced_calc(atoms, calc, i, disp_vec)
                blocks_all.append((disp_atoms, disp_vec.copy(), forces, energy))

    # Force constants C[i, α, j, β] = -∂F_{j,β}/∂u_{i,α}  (eV/Å²)
    forces_blocks = [(dv, f) for (_, dv, f, _) in blocks_all]  # (disp_vec, forces)
    C = _force_constants_brute(N, forces_blocks, delta, nfree, forces_eq)
    np.save("ase_files/force_constants.npy", C)

    # Build DYNMAT entries: one per (atom, direction), central-difference forces
    n_dynmat = N * 3
    dynmat_entries = _make_dynmat_entries_brute(N, forces_blocks, forces_eq, delta, nfree)

    # Compute phonon frequencies and eigenvectors
    freqs, eigvecs = _diagonalize(atoms, C)

    # --- Write output files ---
    n_types = len(set(atoms.get_chemical_symbols()))
    _write_dynmat("DYNMAT", atoms, dynmat_entries, n_types, n_dynmat)
    _write_xdatcar_phonons("XDATCAR", atoms, blocks_all)
    _write_oszicar_phonons("OSZICAR", energy_eq, [(e,) for (_, _, _, e) in blocks_all])
    _write_outcar_phonons("OUTCAR", atoms, freqs, eigvecs, cfg)
    write_contcar("CONTCAR", atoms)

    print(f"[done] {n_disp_all} displaced configurations complete. "
          f"Wrote DYNMAT, XDATCAR, OSZICAR, OUTCAR, CONTCAR, "
          f"ase_files/force_constants.npy.")


# ===========================================================================
# IBRION=6: symmetry-reduced via phonopy
# ===========================================================================

def _run_with_symmetry(atoms: Atoms, calc, cfg) -> None:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    delta = cfg.POTIM
    nfree = cfg.NFREE
    N     = len(atoms)
    n_full = N * 3 * nfree

    forces_eq = atoms.get_forces()
    energy_eq = atoms.get_potential_energy()
    _check_residual_forces(forces_eq)

    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.array(atoms.get_cell()),
        scaled_positions=atoms.get_scaled_positions(),
    )
    ph = Phonopy(ph_atoms, supercell_matrix=np.eye(3, dtype=int))
    ph.generate_displacements(distance=delta, is_plusminus=(nfree == 2))

    supercells   = ph.supercells_with_displacements
    n_disp       = len(supercells)
    reduction    = 100 * (1 - n_disp / n_full) if n_full > 0 else 0
    disp_infos   = ph.dataset["first_atoms"]

    print(f"[info] Phonons (symmetry): IBRION=6, POTIM={delta} Å, NFREE={nfree}")
    print(f"[info] {n_disp} irreducible displacements "
          f"(vs {n_full} without symmetry — {reduction:.0f}% reduction)")

    forces_list  = []
    energies     = []
    disp_structs = []   # ASE Atoms for each displaced supercell (for XDATCAR)
    for k, sc in enumerate(supercells):
        ase_sc = Atoms(
            symbols=sc.symbols,
            cell=sc.cell,
            scaled_positions=sc.scaled_positions,
            pbc=True,
        )
        ase_sc.calc = calc
        atom_idx = disp_infos[k]["number"]
        d        = disp_infos[k]["displacement"]
        alpha    = int(np.argmax(np.abs(d)))
        sign_str = "+" if d[alpha] > 0 else "-"
        print(f"  [{k+1:4d}/{n_disp}] atom {atom_idx:4d}  {sign_str}{'xyz'[alpha]}", flush=True)
        f = ase_sc.get_forces()
        e = ase_sc.get_potential_energy()
        forces_list.append(f)
        energies.append(e)
        disp_structs.append(ase_sc)

    ph.forces = forces_list
    ph.produce_force_constants()
    ph.save("ase_files/phonopy_params.yaml")

    # Save force constants (N, N, 3, 3) in phonopy convention
    np.save("ase_files/force_constants.npy", ph.force_constants)

    # Build DYNMAT entries from reconstructed phonopy force constants
    # (one entry per irreducible atom/direction, positive displacement)
    n_types        = len(set(atoms.get_chemical_symbols()))
    dynmat_entries = _make_dynmat_entries_phonopy(atoms, ph, disp_infos, delta)
    n_dynmat       = len(dynmat_entries)

    # Dynamical matrix: C_my[i,α,j,β] = ph.force_constants.transpose(0,2,1,3)[i,α,j,β]
    C = ph.force_constants.transpose(0, 2, 1, 3)   # (N, 3, N, 3)
    freqs, eigvecs = _diagonalize(atoms, C)

    # --- Write output files ---
    _write_dynmat("DYNMAT", atoms, dynmat_entries, n_types, n_dynmat)
    _write_xdatcar_phonons_from_sc(
        "XDATCAR", atoms, disp_structs)
    _write_oszicar_phonons("OSZICAR", energy_eq, [(e,) for e in energies])
    _write_outcar_phonons("OUTCAR", atoms, freqs, eigvecs, cfg)
    write_contcar("CONTCAR", atoms)

    print(f"[done] {n_disp} displaced configurations complete. "
          f"Wrote DYNMAT, XDATCAR, OSZICAR, OUTCAR, CONTCAR, "
          f"ase_files/force_constants.npy, ase_files/phonopy_params.yaml.")


# ===========================================================================
# Shared physics helpers
# ===========================================================================

def _check_residual_forces(forces_eq: np.ndarray) -> None:
    fmax = float(np.max(np.linalg.norm(forces_eq, axis=1)))
    if fmax > 0.05:
        print(f"[warn] Max equilibrium force = {fmax:.4f} eV/Å. "
              "Consider relaxing the structure first (IBRION=1/2/3).")


def _displaced_calc(atoms: Atoms, calc, atom_idx: int,
                    disp_vec: np.ndarray):
    """Return (displaced_atoms, forces, energy) for one displacement."""
    a = atoms.copy()
    a.calc = calc
    pos = a.get_positions()
    pos[atom_idx] += disp_vec
    a.set_positions(pos)
    forces = a.get_forces()
    energy = a.get_potential_energy()
    return a, forces, energy


def _force_constants_brute(N: int, forces_blocks: list, delta: float,
                            nfree: int, forces_eq: np.ndarray) -> np.ndarray:
    """
    C[i, α, j, β] = -∂F_{j,β}/∂u_{i,α}  (eV/Å²)
    Central differences (NFREE=2): -(F_+ − F_−) / 2δ
    Forward differences (NFREE=1): -(F_+ − F_eq) / δ
    """
    C = np.zeros((N, 3, N, 3))
    k = 0
    for i in range(N):
        for alpha in range(3):
            if nfree == 2:
                _, f_plus  = forces_blocks[k]
                _, f_minus = forces_blocks[k + 1]
                C[i, alpha] = -(f_plus - f_minus) / (2 * delta)
                k += 2
            else:
                _, f_plus = forces_blocks[k]
                C[i, alpha] = -(f_plus - forces_eq) / delta
                k += 1
    return C


def _make_dynmat_entries_brute(N: int, forces_blocks: list,
                                forces_eq: np.ndarray,
                                delta: float, nfree: int) -> list:
    """
    Return DYNMAT entries for brute-force runs: one per (atom, direction).
    Each entry: (atom_idx, alpha, disp_vec, dynmat_forces)
    dynmat_forces[j,β] = (F_+[j,β] - F_-[j,β])/2   for NFREE=2
                       = F_+[j,β]  - F_eq[j,β]        for NFREE=1
    """
    entries = []
    k = 0
    for i in range(N):
        for alpha in range(3):
            disp_vec = np.zeros(3)
            disp_vec[alpha] = delta          # positive direction
            if nfree == 2:
                _, f_plus  = forces_blocks[k]
                _, f_minus = forces_blocks[k + 1]
                dynmat_forces = (f_plus - f_minus) / 2
                k += 2
            else:
                _, f_plus = forces_blocks[k]
                dynmat_forces = f_plus - forces_eq
                k += 1
            entries.append((i, alpha, disp_vec, dynmat_forces))
    return entries


def _make_dynmat_entries_phonopy(atoms: Atoms, ph, disp_infos: list,
                                  delta: float) -> list:
    """
    Return DYNMAT entries for IBRION=6: one per irreducible (atom, direction),
    forces derived from the reconstructed phonopy force constants.

    dynmat_forces[j,β] = -ph.force_constants[j, atom_i, β, α] × δ
    """
    N = len(atoms)
    seen     = set()
    entries  = []
    for d in disp_infos:
        atom_idx = d["number"]
        disp_arr = np.array(d["displacement"])
        alpha    = int(np.argmax(np.abs(disp_arr)))
        key      = (atom_idx, alpha)
        if key in seen or disp_arr[alpha] < 0:
            continue
        seen.add(key)
        disp_vec = np.zeros(3)
        disp_vec[alpha] = delta
        dynmat_forces = np.zeros((N, 3))
        for j in range(N):
            for beta in range(3):
                # ph.force_constants[j, atom_idx, beta, alpha]
                #   = -∂F_{j,β}/∂u_{i,α}  (positive for self-term)
                # DYNMAT force = (F_+ - F_-)/2 ≈ -fc × δ
                dynmat_forces[j, beta] = -ph.force_constants[j, atom_idx, beta, alpha] * delta
        entries.append((atom_idx, alpha, disp_vec, dynmat_forces))
    return entries


def _diagonalize(atoms: Atoms, C: np.ndarray):
    """
    Build and diagonalize the dynamical matrix.

    C: (N, 3, N, 3) force constants, convention C[i,α,j,β] = -∂F_{j,β}/∂u_{i,α}

    Returns
    -------
    freqs   : (3N,) frequencies in THz; negative sign denotes imaginary modes
    eigvecs : (3N, N, 3) normalised eigenvectors (columns of the mass-weighted
              dynamical matrix, directly used as dx/dy/dz in OUTCAR)
    """
    N      = len(atoms)
    masses = atoms.get_masses()

    # Mass-weighted dynamical matrix D[iα, jβ] = C[i,α,j,β] / sqrt(m_i × m_j)
    D = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for alpha in range(3):
            for j in range(N):
                for beta in range(3):
                    D[i * 3 + alpha, j * 3 + beta] = (
                        C[i, alpha, j, beta] / math.sqrt(masses[i] * masses[j])
                    )
    D = (D + D.T) / 2  # enforce symmetry numerically

    omega2, v = np.linalg.eigh(D)   # v columns are eigenvectors, shape (3N, 3N)

    # Convert eigenvalues to THz (negative for imaginary modes)
    freqs = np.where(
        omega2 >= 0,
        np.sqrt(np.maximum(omega2, 0.0)) * _THz_FACTOR,
        -np.sqrt(np.maximum(-omega2, 0.0)) * _THz_FACTOR,
    )

    # Eigenvectors: reshape to (mode, atom, xyz)
    eigvecs = v.T.reshape(3 * N, N, 3)
    return freqs, eigvecs


# ===========================================================================
# Output file writers
# ===========================================================================

def _write_dynmat(path: str, atoms: Atoms, entries: list,
                  n_types: int, n_displacements: int) -> None:
    """
    Write DYNMAT in VASP format.

    Header : n_types  n_atoms  n_displacements
    Masses : one per species (3 decimal places)
    Per displacement:
        atom_idx(1-based)  direction(1/2/3)  dx  dy  dz
        N lines of Fx  Fy  Fz  [eV/Å, central-difference half-forces]
    """
    N      = len(atoms)
    masses = atoms.get_masses()

    # Unique species masses in POSCAR order
    species_masses = []
    seen_sym = []
    for sym, m in zip(atoms.get_chemical_symbols(), masses):
        if sym not in seen_sym:
            seen_sym.append(sym)
            species_masses.append(m)

    with open(path, "w") as f:
        f.write(f"  {n_types:3d}  {N:3d}  {n_displacements:3d}\n")
        f.write(" " + " ".join(f"{m:.3f}" for m in species_masses) + "\n")
        for atom_idx, alpha, disp_vec, dynmat_forces in entries:
            direction = alpha + 1    # 1-based: 1=x, 2=y, 3=z
            f.write(f"  {atom_idx + 1:3d}  {direction:3d}"
                    f"  {disp_vec[0]:6.4f}  {disp_vec[1]:6.4f}  {disp_vec[2]:6.4f}\n")
            for j in range(N):
                f.write(f"  {dynmat_forces[j, 0]:16.12f}"
                        f"  {dynmat_forces[j, 1]:16.12f}"
                        f"  {dynmat_forces[j, 2]:16.12f}\n")


def _write_xdatcar_phonons(path: str, atoms: Atoms,
                            blocks_all: list) -> None:
    """Write XDATCAR for brute-force run: initial + all displaced configs."""
    write_xdatcar_header(path, atoms)
    # Config 1: initial (undisplaced)
    append_xdatcar_frame(path, atoms, step=1)
    # Configs 2..: displaced
    for k, (disp_atoms, _, _, _) in enumerate(blocks_all):
        append_xdatcar_frame(path, disp_atoms, step=k + 2)


def _write_xdatcar_phonons_from_sc(path: str, atoms: Atoms,
                                    disp_structs: list) -> None:
    """Write XDATCAR for phonopy run: initial + displaced supercells."""
    write_xdatcar_header(path, atoms)
    append_xdatcar_frame(path, atoms, step=1)
    for k, sc in enumerate(disp_structs):
        append_xdatcar_frame(path, sc, step=k + 2)


def _write_oszicar_phonons(path: str, energy_eq: float,
                            energies: list) -> None:
    """Write OSZICAR: one line per configuration (initial + displaced)."""
    all_energies = [energy_eq] + [e[0] for e in energies]
    with open(path, "w") as f:
        prev = all_energies[0]
        for n, e in enumerate(all_energies, start=1):
            dE = e - prev if n > 1 else 0.0
            f.write(f"   {n} F= {e:17.8E} E0= {e:17.8E}  d E ={dE:12.5E}\n")
            prev = e


def _write_outcar_phonons(path: str, atoms: Atoms,
                           freqs: np.ndarray, eigvecs: np.ndarray,
                           cfg) -> None:
    """Write OUTCAR with phonon eigenvectors and eigenvalues in VASP format."""
    positions = atoms.get_positions()
    n_modes   = len(freqs)

    with open(path, "w") as f:
        f.write(" vasp-mace  phonon calculation (IBRION={cfg.IBRION})\n".format(cfg=cfg))
        f.write(f" NIONS = {len(atoms)}\n")
        f.write(f" POTIM = {cfg.POTIM:.4f}\n")
        f.write(f" NFREE = {cfg.NFREE}\n\n")

        f.write(" Eigenvectors and eigenvalues of the dynamical matrix\n")
        f.write(" ----------------------------------------------------\n\n\n")

        for m in range(n_modes):
            freq = freqs[m]
            is_imaginary = freq < 0
            label = "f/i=" if is_imaginary else "f  ="
            abs_f  = abs(freq)
            f2pi   = abs_f * 2 * math.pi
            f_cm   = abs_f * _THz_TO_CM
            f_meV  = abs_f * _THz_TO_MEV

            f.write(f"  {m + 1:3d} {label} {abs_f:12.6f} THz  "
                    f"{f2pi:12.6f} 2PiTHz  "
                    f"{f_cm:12.6f} cm-1  "
                    f"{f_meV:12.6f} meV\n")
            f.write("             X         Y         Z           "
                    "dx          dy          dz\n")
            ev = eigvecs[m]   # shape (N, 3)
            for i, (pos, d) in enumerate(zip(positions, ev)):
                f.write(f"  {pos[0]:10.6f}{pos[1]:10.6f}{pos[2]:10.6f}"
                        f"    {d[0]:10.6f}  {d[1]:10.6f}  {d[2]:10.6f}  \n")
            f.write("\n")
