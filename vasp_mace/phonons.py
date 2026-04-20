import os
import numpy as np
from ase import Atoms


def run_phonons(atoms: Atoms, calc, cfg) -> None:
    """
    IBRION=5: all N×3×NFREE displacements (no symmetry).
    IBRION=6: symmetry-reduced displacements via phonopy (requires `pip install phonopy`).
              Falls back to IBRION=5 behaviour if phonopy is not installed.
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


# ---------------------------------------------------------------------------
# IBRION=5: brute-force (all displacements)
# ---------------------------------------------------------------------------

def _run_brute_force(atoms: Atoms, calc, cfg) -> None:
    delta = cfg.POTIM
    nfree = cfg.NFREE
    N = len(atoms)
    n_disp = N * 3 * nfree

    print(f"[info] Phonons: IBRION={cfg.IBRION}, POTIM={delta} Å, NFREE={nfree}")
    print(f"[info] {N} atoms × 3 directions × {nfree} = {n_disp} single-point calculations")
    if N > 150:
        print(f"[warn] {N} atoms → {n_disp} force evaluations. "
              "Phonon calculations work best on supercells of ~16–128 atoms.")

    forces_eq = atoms.get_forces()
    _check_residual_forces(forces_eq)

    blocks = []
    count = 0
    for i in range(N):
        for alpha in range(3):
            for sign in ([+1, -1] if nfree == 2 else [+1]):
                count += 1
                disp_vec = np.zeros(3)
                disp_vec[alpha] = sign * delta
                sign_str = "+" if sign > 0 else "-"
                print(f"  [{count:4d}/{n_disp}] atom {i:4d}  {sign_str}{'xyz'[alpha]}", flush=True)
                forces = _displaced_forces(atoms, calc, i, disp_vec)
                blocks.append((i, disp_vec.copy(), forces))

    C = _force_constants_brute(N, blocks, delta, nfree, forces_eq)
    np.save("ase_files/force_constants.npy", C)
    _write_dynmat("DYNMAT", atoms, blocks)

    print(f"[done] {n_disp} displaced configurations complete. "
          "Wrote DYNMAT and ase_files/force_constants.npy.")


# ---------------------------------------------------------------------------
# IBRION=6: symmetry-reduced via phonopy
# ---------------------------------------------------------------------------

def _run_with_symmetry(atoms: Atoms, calc, cfg) -> None:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    delta = cfg.POTIM
    nfree = cfg.NFREE
    N = len(atoms)
    n_full = N * 3 * nfree

    forces_eq = atoms.get_forces()
    _check_residual_forces(forces_eq)

    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.array(atoms.get_cell()),
        scaled_positions=atoms.get_scaled_positions(),
    )
    ph = Phonopy(ph_atoms, supercell_matrix=np.eye(3, dtype=int))
    ph.generate_displacements(distance=delta, is_plusminus=(nfree == 2))

    supercells = ph.supercells_with_displacements
    n_disp = len(supercells)
    reduction = 100 * (1 - n_disp / n_full) if n_full > 0 else 0

    print(f"[info] Phonons (symmetry): IBRION=6, POTIM={delta} Å, NFREE={nfree}")
    print(f"[info] {n_disp} irreducible displacements "
          f"(vs {n_full} without symmetry — {reduction:.0f}% reduction)")

    forces_list = []
    disp_infos = ph.dataset["first_atoms"]
    for k, sc in enumerate(supercells):
        ase_sc = Atoms(
            symbols=sc.symbols,
            cell=sc.cell,
            scaled_positions=sc.scaled_positions,
            pbc=True,
        )
        ase_sc.calc = calc
        atom_idx = disp_infos[k]["number"]
        d = disp_infos[k]["displacement"]
        alpha = int(np.argmax(np.abs(d)))
        sign_str = "+" if d[alpha] > 0 else "-"
        print(f"  [{k+1:4d}/{n_disp}] atom {atom_idx:4d}  {sign_str}{'xyz'[alpha]}", flush=True)
        forces_list.append(ase_sc.get_forces())

    ph.forces = forces_list
    ph.produce_force_constants()

    # phonopy_params.yaml: displacements + force constants (phonopy-ready input)
    ph.save("phonopy_params.yaml")

    # force_constants.npy: shape (N, N, 3, 3), phonopy convention
    # fc[i, j, α, β] = -∂F_{i,α}/∂u_{j,β}
    np.save("ase_files/force_constants.npy", ph.force_constants)

    # DYNMAT: irreducible displacement–force pairs (VASP format)
    _write_dynmat_from_dataset("DYNMAT", atoms, disp_infos, forces_list)

    print(f"[done] {n_disp} displaced configurations complete. "
          "Wrote DYNMAT, ase_files/force_constants.npy, phonopy_params.yaml.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_residual_forces(forces_eq: np.ndarray) -> None:
    fmax = float(np.max(np.linalg.norm(forces_eq, axis=1)))
    if fmax > 0.05:
        print(f"[warn] Max equilibrium force = {fmax:.4f} eV/Å. "
              "Consider relaxing the structure first (IBRION=1/2/3).")


def _displaced_forces(atoms: Atoms, calc, atom_idx: int,
                      disp_vec: np.ndarray) -> np.ndarray:
    a = atoms.copy()
    a.calc = calc
    pos = a.get_positions()
    pos[atom_idx] += disp_vec
    a.set_positions(pos)
    return a.get_forces()


def _force_constants_brute(N: int, blocks: list, delta: float,
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
                _, _, f_plus  = blocks[k]
                _, _, f_minus = blocks[k + 1]
                C[i, alpha] = -(f_plus - f_minus) / (2 * delta)
                k += 2
            else:
                _, _, f_plus = blocks[k]
                C[i, alpha] = -(f_plus - forces_eq) / delta
                k += 1
    return C


def _write_dynmat(path: str, atoms: Atoms, blocks: list) -> None:
    """Write DYNMAT from brute-force blocks (all displacements)."""
    N = len(atoms)
    masses = atoms.get_masses()
    with open(path, "w") as f:
        f.write(f"   {N}   {3 * N}   0\n")
        for atom_idx, disp_vec, forces in blocks:
            f.write(f"  {masses[atom_idx]:.6f}"
                    f"  {disp_vec[0]:12.8f}  {disp_vec[1]:12.8f}  {disp_vec[2]:12.8f}\n")
            for j in range(N):
                f.write(f"  {forces[j, 0]:16.8f}"
                        f"  {forces[j, 1]:16.8f}"
                        f"  {forces[j, 2]:16.8f}\n")


def _write_dynmat_from_dataset(path: str, atoms: Atoms,
                               disp_infos: list, forces_list: list) -> None:
    """Write DYNMAT from phonopy displacement dataset (irreducible displacements)."""
    N = len(atoms)
    masses = atoms.get_masses()
    with open(path, "w") as f:
        f.write(f"   {N}   {3 * N}   0\n")
        for k, disp_info in enumerate(disp_infos):
            atom_idx = disp_info["number"]
            disp_vec = np.array(disp_info["displacement"])
            forces = forces_list[k]
            f.write(f"  {masses[atom_idx]:.6f}"
                    f"  {disp_vec[0]:12.8f}  {disp_vec[1]:12.8f}  {disp_vec[2]:12.8f}\n")
            for j in range(N):
                f.write(f"  {forces[j, 0]:16.8f}"
                        f"  {forces[j, 1]:16.8f}"
                        f"  {forces[j, 2]:16.8f}\n")
