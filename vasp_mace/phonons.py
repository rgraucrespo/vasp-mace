import os
import numpy as np
from ase import Atoms


def run_phonons(atoms: Atoms, calc, cfg) -> None:
    """
    IBRION=5/6: finite-difference force constants via Cartesian displacements.

    Each atom is displaced by ±POTIM (NFREE=2, central differences) or
    +POTIM (NFREE=1, forward differences). Forces are collected and written
    to DYNMAT (VASP format) and ase_files/force_constants.npy.

    IBRION=5: no symmetry (all N×3 displacements computed).
    IBRION=6: symmetry flag set; currently treated identically to IBRION=5
              (point-group reduction not yet implemented).
    """
    os.makedirs("ase_files", exist_ok=True)

    delta = cfg.POTIM  # displacement amplitude (Å)
    nfree = cfg.NFREE
    N = len(atoms)
    n_disp = N * 3 * nfree

    print(f"[info] Phonons: IBRION={cfg.IBRION}, POTIM={delta} Å, NFREE={nfree}")
    if cfg.IBRION == 6:
        print("[info] IBRION=6: symmetry reduction not yet implemented; "
              "computing all displacements as in IBRION=5.")
    print(f"[info] {N} atoms × 3 directions × {nfree} = {n_disp} single-point calculations")

    if N > 150:
        print(f"[warn] {N} atoms → {n_disp} force evaluations. "
              "Phonon calculations work best on supercells of ~16–128 atoms.")

    # Equilibrium forces: sanity check + NFREE=1 reference subtraction
    forces_eq = atoms.get_forces()
    fmax_eq = float(np.max(np.linalg.norm(forces_eq, axis=1)))
    if fmax_eq > 0.05:
        print(f"[warn] Max equilibrium force = {fmax_eq:.4f} eV/Å. "
              "Consider relaxing the structure first (IBRION=1/2/3).")

    # Run all displaced configurations
    blocks = []  # list of (atom_idx, disp_vec [3], forces [N,3])
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

    # Force constants C[i, alpha, j, beta] (eV/Å²)
    C = _force_constants(N, blocks, delta, nfree, forces_eq)
    np.save("ase_files/force_constants.npy", C)

    _write_dynmat("DYNMAT", atoms, blocks)

    print(f"[done] {n_disp} displaced configurations complete. "
          "Wrote DYNMAT and ase_files/force_constants.npy.")


def _displaced_forces(atoms: Atoms, calc, atom_idx: int, disp_vec: np.ndarray) -> np.ndarray:
    a = atoms.copy()
    a.calc = calc
    pos = a.get_positions()
    pos[atom_idx] += disp_vec
    a.set_positions(pos)
    return a.get_forces()


def _force_constants(N: int, blocks: list, delta: float, nfree: int,
                     forces_eq: np.ndarray) -> np.ndarray:
    """
    C[i, alpha, j, beta] = -dF_{j,beta}/du_{i,alpha}  (eV/Å²)

    Central differences (NFREE=2): C = -(F_+ - F_-) / (2*delta)
    Forward differences (NFREE=1): C = -(F_+ - F_eq) / delta
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
    """Write VASP DYNMAT format (displacement–force pairs)."""
    N = len(atoms)
    masses = atoms.get_masses()

    with open(path, "w") as f:
        f.write(f"   {N}   {3 * N}   0\n")
        for atom_idx, disp_vec, forces in blocks:
            f.write(f"  {masses[atom_idx]:.6f}"
                    f"  {disp_vec[0]:12.8f}"
                    f"  {disp_vec[1]:12.8f}"
                    f"  {disp_vec[2]:12.8f}\n")
            for j in range(N):
                f.write(f"  {forces[j, 0]:16.8f}"
                        f"  {forces[j, 1]:16.8f}"
                        f"  {forces[j, 2]:16.8f}\n")
