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


def _isotropic_moduli(bulk_modulus: float, shear_modulus: float):
    """Return Young's modulus and Poisson ratio from bulk and shear moduli."""
    young_modulus = (
        9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
    )
    poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (
        6 * bulk_modulus + 2 * shear_modulus
    )
    return young_modulus, poisson_ratio


def _modulus_row(name: str, bulk_modulus: float, shear_modulus: float):
    """Build one isotropic elastic-modulus table row."""
    young_modulus, poisson_ratio = _isotropic_moduli(bulk_modulus, shear_modulus)
    return name, bulk_modulus, shear_modulus, young_modulus, poisson_ratio


def _hashin_shtrikman_moduli(C: np.ndarray):
    """Return lower, upper, and midpoint Hashin-Shtrikman modulus rows.

    The bounds use the isotropic-comparison-tensor formulation for a randomly
    oriented polycrystal, so they apply to the full elastic tensor rather than
    treating the Voigt and Reuss averages as material phases.
    """
    kelvin_scale = np.diag([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
    C_kelvin = kelvin_scale @ C @ kelvin_scale
    eigenvalues = np.linalg.eigvalsh(C_kelvin)
    if np.min(eigenvalues) <= 0:
        raise ValueError(
            "Hashin-Shtrikman bounds require a mechanically stable elastic tensor."
        )

    scale = float(np.max(eigenvalues))
    K_lower, G_lower = _optimize_hs_shear(C_kelvin, scale, positive_residual=True)
    K_upper, G_upper = _optimize_hs_shear(C_kelvin, scale, positive_residual=False)
    K_mid = (K_lower + K_upper) / 2
    G_mid = (G_lower + G_upper) / 2
    return (
        _modulus_row("Hashin-Shtrikman lower", K_lower, G_lower),
        _modulus_row("Hashin-Shtrikman upper", K_upper, G_upper),
        _modulus_row("Hashin-Shtrikman midpoint", K_mid, G_mid),
    )


def _optimize_hs_shear(
    C_kelvin: np.ndarray, scale: float, positive_residual: bool
) -> tuple:
    """Optimize an HS shear bound over isotropic comparison tensors."""
    # A coarse global scan followed by local refinements is sufficient for this
    # two-dimensional bounded problem and keeps NumPy as the only dependency.
    floor = scale * 1.0e-6
    k_min = g_min = floor
    k_max = g_max = scale
    best_k = best_g = None

    for count in (81, 31, 31, 31):
        k_values, g_values = np.meshgrid(
            np.linspace(k_min, k_max, count),
            np.linspace(g_min, g_max, count),
            indexing="ij",
        )
        candidate_bulk, candidate_shears, residual_eigenvalues = _hs_candidate_moduli(
            C_kelvin, k_values.ravel(), g_values.ravel()
        )
        nonsingular = np.min(np.abs(residual_eigenvalues), axis=1) > scale * 1.0e-10
        if positive_residual:
            valid = nonsingular & (
                np.min(residual_eigenvalues, axis=1) >= -scale * 1.0e-9
            )
            score = np.where(
                valid & np.isfinite(candidate_shears), candidate_shears, -np.inf
            )
            index = int(np.argmax(score))
        else:
            valid = nonsingular & (
                np.max(residual_eigenvalues, axis=1) <= scale * 1.0e-9
            )
            score = np.where(
                valid & np.isfinite(candidate_shears), candidate_shears, np.inf
            )
            index = int(np.argmin(score))

        if not np.isfinite(score[index]):
            raise ValueError(
                "Could not find a valid Hashin-Shtrikman comparison tensor."
            )

        best_k = k_values.ravel()[index]
        best_g = g_values.ravel()[index]
        step_k = (k_max - k_min) / (count - 1)
        step_g = (g_max - g_min) / (count - 1)
        k_min = max(floor, best_k - 2 * step_k)
        k_max = min(scale, best_k + 2 * step_k)
        g_min = max(floor, best_g - 2 * step_g)
        g_max = min(scale, best_g + 2 * step_g)

    candidate_bulk, candidate_shears, _ = _hs_candidate_moduli(
        C_kelvin, np.array([best_k]), np.array([best_g])
    )
    return float(candidate_bulk[0]), float(candidate_shears[0])


def _hs_candidate_moduli(C_kelvin: np.ndarray, K0: np.ndarray, G0: np.ndarray):
    """Evaluate HS bulk/shear moduli and comparison-tensor eigenvalues."""
    volumetric = np.zeros((6, 6))
    volumetric[:3, :3] = 1.0 / 3.0
    deviatoric = np.eye(6) - volumetric

    comparison = 3 * K0[:, None, None] * volumetric
    comparison += 2 * G0[:, None, None] * deviatoric
    residual = C_kelvin - comparison

    residual_values, residual_vectors = np.linalg.eigh(residual)
    residual_compliance = np.matmul(
        residual_vectors * (1.0 / residual_values)[:, None, :],
        np.swapaxes(residual_vectors, 1, 2),
    )

    alpha = -3.0 / (3.0 * K0 + 4.0 * G0)
    beta = -3.0 * (K0 + 2.0 * G0) / (5.0 * G0 * (3.0 * K0 + 4.0 * G0))
    gamma = (alpha - 3.0 * beta) / 9.0
    response = residual_compliance - beta[:, None, None] * np.eye(6)
    response -= 3.0 * gamma[:, None, None] * volumetric

    response_values, response_vectors = np.linalg.eigh(response)
    stress_response = np.matmul(
        response_vectors * (1.0 / response_values)[:, None, :],
        np.swapaxes(response_vectors, 1, 2),
    )
    contraction_1 = 3.0 * np.einsum("ij,nji->n", volumetric, stress_response)
    contraction_2 = np.trace(stress_response, axis1=1, axis2=2)
    B1 = (2.0 * contraction_1 - contraction_2) / 15.0
    B2 = (3.0 * contraction_2 - contraction_1) / 30.0
    bulk_response = 3.0 * B1 + 2.0 * B2
    bulk = K0 + bulk_response / (3.0 + alpha * bulk_response)
    shear = G0 + B2 / (1.0 + 2.0 * beta * B2)
    return bulk, shear, residual_values


def run_elastic(
    atoms: Atoms, calc: Any, cfg: IncarConfig, outcar_path: str = "OUTCAR"
) -> np.ndarray:
    """Compute and write the 6×6 elastic tensor.

    Applies 6 Voigt strain patterns ±STRAIN_AMP (12 single-point calculations),
    retrieves stress from the MACE calculator, and central-differences to Cij.
    Derives Voigt, Reuss, Hill, and Hashin-Shtrikman polycrystalline moduli.
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
    modulus_rows = [
        _modulus_row("Voigt", K_v, G_v),
        _modulus_row("Reuss", K_r, G_r),
        _modulus_row("Hill", K_h, G_h),
        *_hashin_shtrikman_moduli(C_GPa),
    ]

    _print_elastic_summary(C_GPa, modulus_rows)
    _append_elastic_outcar(outcar_path, C_GPa, modulus_rows)
    print(f"[done] Elastic constants appended to {outcar_path}.")
    return C_GPa


def _format_modulus_table(modulus_rows):
    """Format polycrystalline modulus rows as a fixed-width table."""
    lines = [
        "  Polycrystalline elastic moduli",
        f"  {'Approximation':<28s}{'K (GPa)':>12s}{'G (GPa)':>12s}"
        f"{'E (GPa)':>12s}{'Poisson ν':>12s}",
        "  " + "-" * 76,
    ]
    for name, bulk_modulus, shear_modulus, young_modulus, poisson_ratio in modulus_rows:
        lines.append(
            f"  {name:<28s}{bulk_modulus:>12.3f}{shear_modulus:>12.3f}"
            f"{young_modulus:>12.3f}{poisson_ratio:>12.4f}"
        )
    return "\n".join(lines)


def _print_elastic_summary(C, modulus_rows):
    header = "  ".join(f"{label:>8}" for label in _ASE_LABELS)
    print("\n Elastic tensor (GPa) — ASE Voigt ordering: xx yy zz yz xz xy")
    print(f" {'':8s}  {header}")
    for i, row_label in enumerate(_ASE_LABELS):
        row = "  ".join(f"{C[i,j]:8.2f}" for j in range(6))
        print(f" {row_label:8s}  {row}")
    print()
    print(_format_modulus_table(modulus_rows))


def _append_elastic_outcar(outcar_path, C_GPa, modulus_rows):
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
        f.write(" POLYCRYSTALLINE ELASTIC MODULI (GPa)\n")
        f.write(_format_modulus_table(modulus_rows) + "\n")
        f.write("\n")
