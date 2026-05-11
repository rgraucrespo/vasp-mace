"""Elastic tensor finite-difference calculations."""

from typing import Any

import numpy as np
from ase import Atoms

from .types_ import IncarConfig

STRAIN_AMP = 0.01  # dimensionless Voigt strain amplitude (1%)
EV_A3_TO_GPA = 160.21766  # 1 eV/Å³ in GPa
EV_A3_TO_KBAR = 1602.1766  # 1 eV/Å³ in kBar

# ASE Voigt ordering: [xx, yy, zz, yz, xz, xy]  (indices 0-5)
# VASP OUTCAR ordering: XX YY ZZ XY YZ ZX        (permute with [0,1,2,5,3,4])
_ASE_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]
_VASP_LABELS = ["XX", "YY", "ZZ", "XY", "YZ", "ZX"]
_TO_VASP = [0, 1, 2, 5, 3, 4]  # ASE index → VASP column/row position


def _strain_matrix(voigt_idx: int, delta: float) -> np.ndarray:
    """3×3 symmetric strain matrix for Voigt component voigt_idx (ASE ordering)."""
    eps = np.zeros((3, 3))
    if voigt_idx == 0:
        eps[0, 0] = delta
    elif voigt_idx == 1:
        eps[1, 1] = delta
    elif voigt_idx == 2:
        eps[2, 2] = delta
    elif voigt_idx == 3:  # yz shear: γ = δ → off-diag = δ/2
        eps[1, 2] = eps[2, 1] = delta / 2
    elif voigt_idx == 4:  # xz shear
        eps[0, 2] = eps[2, 0] = delta / 2
    elif voigt_idx == 5:  # xy shear
        eps[0, 1] = eps[1, 0] = delta / 2
    return eps


def _voigt_averages(C: np.ndarray):
    """Voigt upper-bound bulk and shear moduli (GPa) from 6×6 C (ASE Voigt)."""
    K_v = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    G_v = (
        C[0, 0]
        + C[1, 1]
        + C[2, 2]
        - C[0, 1]
        - C[0, 2]
        - C[1, 2]
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15
    return K_v, G_v


def _reuss_averages(C: np.ndarray):
    """Reuss lower-bound bulk and shear moduli (GPa) from 6×6 C (ASE Voigt)."""
    S = np.linalg.inv(C)
    K_r = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    G_r = 15.0 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )
    return K_r, G_r


def run_elastic(
    atoms: Atoms, calc: Any, cfg: IncarConfig, outcar_path: str = "OUTCAR"
) -> np.ndarray:
    """Compute and write the 6×6 elastic tensor.

    Applies 6 Voigt strain patterns ±STRAIN_AMP (12 single-point calculations),
    retrieves stress from the MACE calculator, and central-differences to Cij.
    Derives Voigt, Reuss, and Hill polycrystalline averages (K, G, E, ν).
    Appends results to outcar_path in VASP format.

    Parameters
    ----------
    atoms
        Equilibrium structure used as the reference cell. The object itself is
        not modified; strained copies are evaluated.
    calc
        ASE-compatible calculator used to evaluate stress.
    cfg
        Parsed INCAR configuration. Currently used for mode context and future
        extension points.
    outcar_path
        OUTCAR path to append the VASP-format elastic tensor block to.

    Returns
    -------
    numpy.ndarray
        Elastic tensor in GPa with ASE Voigt ordering ``xx, yy, zz, yz, xz,
        xy`` and shape ``(6, 6)``.
    """
    delta = STRAIN_AMP
    cell0 = np.array(atoms.get_cell())

    print(
        f"\n[info] Elastic constants: 6 Voigt strains × 2 = 12 calculations (δ={delta})"
    )

    C_eVA3 = np.zeros((6, 6))
    k = 0
    for j in range(6):
        eps = _strain_matrix(j, delta)
        stress_plus = stress_minus = None
        for sign in (+1, -1):
            k += 1
            label = _ASE_LABELS[j]
            sign_str = "+" if sign > 0 else "-"
            print(f"  [{k:2d}/12] strain {sign_str}{label}", flush=True)
            a = atoms.copy()
            a.calc = calc
            a.set_cell((np.eye(3) + sign * eps) @ cell0, scale_atoms=True)
            s = a.get_stress(voigt=True)  # eV/Å³, tensile-positive
            if sign > 0:
                stress_plus = s
            else:
                stress_minus = s
        C_eVA3[:, j] = (stress_plus - stress_minus) / (2 * delta)

    C_eVA3 = (C_eVA3 + C_eVA3.T) / 2  # enforce symmetry
    C_GPa = C_eVA3 * EV_A3_TO_GPA

    K_v, G_v = _voigt_averages(C_GPa)
    K_r, G_r = _reuss_averages(C_GPa)
    K_h = (K_v + K_r) / 2
    G_h = (G_v + G_r) / 2
    E_h = 9 * K_h * G_h / (3 * K_h + G_h)
    nu_h = (3 * K_h - 2 * G_h) / (6 * K_h + 2 * G_h)

    _print_elastic_summary(C_GPa, K_v, G_v, K_r, G_r, K_h, G_h, E_h, nu_h)
    _append_elastic_outcar(outcar_path, C_GPa, K_v, G_v, K_r, G_r, K_h, G_h, E_h, nu_h)
    print(f"[done] Elastic constants appended to {outcar_path}.")
    return C_GPa


def _print_elastic_summary(C, K_v, G_v, K_r, G_r, K_h, G_h, E_h, nu_h):
    header = "  ".join(f"{label:>8}" for label in _ASE_LABELS)
    print("\n Elastic tensor (GPa) — ASE Voigt ordering: xx yy zz yz xz xy")
    print(f" {'':8s}  {header}")
    for i, row_label in enumerate(_ASE_LABELS):
        row = "  ".join(f"{C[i,j]:8.2f}" for j in range(6))
        print(f" {row_label:8s}  {row}")
    print()
    print(f" Voigt:  K = {K_v:7.2f} GPa   G = {G_v:7.2f} GPa")
    print(f" Reuss:  K = {K_r:7.2f} GPa   G = {G_r:7.2f} GPa")
    print(
        f" Hill:   K = {K_h:7.2f} GPa   G = {G_h:7.2f} GPa   "
        f"E = {E_h:7.2f} GPa   ν = {nu_h:.4f}"
    )


def _append_elastic_outcar(outcar_path, C_GPa, K_v, G_v, K_r, G_r, K_h, G_h, E_h, nu_h):
    """Append elastic tensor and polycrystalline moduli to OUTCAR in VASP format."""
    # Reorder C from ASE Voigt to VASP Voigt: [0,1,2,5,3,4]
    p = _TO_VASP
    C_vasp = C_GPa[np.ix_(p, p)]  # reorder rows and columns
    C_kbar = C_vasp * 10.0  # GPa → kBar

    sep = " " + "-" * 89

    with open(outcar_path, "a") as f:
        f.write("\n")
        f.write(" TOTAL ELASTIC MODULI (kBar)\n")
        f.write(
            f" Direction {'':4s}"
            + "".join(f"{label:>12s}" for label in _VASP_LABELS)
            + "\n"
        )
        f.write(sep + "\n")
        for i, rl in enumerate(_VASP_LABELS):
            row = "".join(f"{C_kbar[i, j]:12.3f}" for j in range(6))
            f.write(f"  {rl:<10s}{row}\n")
        f.write(sep + "\n")
        f.write("\n")
        f.write(" POLYCRYSTALLINE CONSTANTS (Voigt / Reuss / Hill):\n")
        hdr = f"  {'':10s}{'Bulk modulus K':>16s}{'Shear modulus G':>16s}"
        hdr += f"{'Young mod. E':>14s}{'Poisson ratio':>14s}\n"
        f.write(hdr)
        f.write(f"  {'':10s}{'(GPa)':>16s}{'(GPa)':>16s}{'(GPa)':>14s}{' ':>14s}\n")
        f.write(f"  {'Voigt':<10s}{K_v:>16.3f}{G_v:>16.3f}{'':>14s}{'':>14s}\n")
        f.write(f"  {'Reuss':<10s}{K_r:>16.3f}{G_r:>16.3f}{'':>14s}{'':>14s}\n")
        f.write(f"  {'Hill':<10s}{K_h:>16.3f}{G_h:>16.3f}{E_h:>14.3f}{nu_h:>14.4f}\n")
        f.write("\n")
