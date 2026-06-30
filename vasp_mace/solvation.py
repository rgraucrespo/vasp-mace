"""Implicit-solvation correction terms (slab/cluster only).

Two terms, added on top of the MACE calculator:

* **Nonpolar / cavitation** (always on with ``LSOL``)::

      E_nonpolar = TAU * SASA

  where ``SASA`` is the solvent-accessible surface area (Shrake-Rupley, PBC
  aware) and ``TAU`` is an effective surface tension (INCAR ``TAU``,
  meV/Angstrom^2, VASPsol convention).

* **Polar / electrostatic** (Generalized-Born, OBC variant; on when an
  ``eb_k`` dielectric is supplied)::

      E_polar = -1/2 * k_e * (1/eps_in - 1/eps_out)
                * sum_ij q_i q_j / f_GB(r_ij, R_i, R_j)

  with ``eps_in = 1`` (solute), ``eps_out = EB_K`` (solvent, INCAR ``EB_K``),
  EEQ partial charges ``q`` from :func:`vasp_mace.charges.eeq_charges` (the
  shared charge engine), Onufriev-Bashford-Case effective Born radii ``R`` (a
  PBC-aware HCT descreening integral rescaled by the OBC tanh correction), and
  the Still function ``f_GB = sqrt(r^2 + R_i R_j exp(-r^2/(4 R_i R_j)))``.

  The self terms (``i == j``, giving ``q_i^2 / R_i``) are the Born solvation of
  each atom; the cross terms (``i != j``) provide the essential cancellation
  that keeps *neutral*-system solvation physically small. Crucially, the cross
  terms use the **minimum-image** distance (one nearest image per pair), NOT
  the full periodic lattice sum: the latter is a conditionally-convergent,
  cutoff-sensitive Madelung-like sum that is ill-defined here (it gave +6 eV on
  PbS(100), vs the correct ~-0.2 eV from the minimum-image form, against a
  VASPsol reference of -0.08 eV).

This is a density-free surrogate, NOT VASPsol's Poisson-Boltzmann model, and is
restricted to slabs / clusters / molecules; a fully periodic 3D bulk cell (no
vacuum) is rejected.

Forces are central finite differences of the total solvation energy. This is
deliberate: the EEQ charges are recomputed at every displaced geometry, so the
charge-position coupling (dq/dr) is included automatically and no analytic
SASA/GB gradient is needed. Stress is not yet included (returned as zero with a
warning) for ISIF >= 3; see not_for_release/solvation_design.md.
"""

from typing import Any, Optional, Tuple
import warnings

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

# vasp_mace/solvation.py

# --- nonpolar (SASA) parameters ---
# Water probe radius (Angstrom).
_PROBE = 1.4
# Fallback van der Waals radius (Angstrom) for elements missing from
# ase.data.vdw_radii (which is NaN for ~50 elements).
_DEFAULT_VDW = 1.5
# Number of surface sample points per atom (Shrake-Rupley).
_N_POINTS = 960
# Central-difference step for forces (Angstrom).
_FD_STEP = 1.0e-3
# Minimum vacuum gap (Angstrom) for a direction to count as "open" (slab).
_MIN_VACUUM = 4.0

# --- polar (Generalized-Born / OBC) parameters ---
# Coulomb constant in eV * Angstrom / e^2 (1 / 4 pi eps0).
_KE = 14.399645
# Solute interior dielectric.
_EPS_IN = 1.0
# Default solvent dielectric (water), VASPsol EB_K default.
_EB_K_DEFAULT = 78.4
# Dielectric offset (Angstrom) subtracted from atomic radii for Born radii.
_OFFSET = 0.09
# OBC-II parameters (Onufriev, Bashford, Case 2004; AMBER igb=5).
_OBC_ALPHA, _OBC_BETA, _OBC_GAMMA = 1.0, 0.8, 4.85
# Default HCT descreening scale factor for elements not in _HCT_SCALE.
_HCT_DEFAULT = 0.8
# Element-specific HCT scale factors (AMBER mbondi set, common elements).
_HCT_SCALE = {
    1: 0.85,  # H
    6: 0.72,  # C
    7: 0.79,  # N
    8: 0.85,  # O
    9: 0.88,  # F
    15: 0.86,  # P
    16: 0.96,  # S
    17: 0.80,  # Cl
}
# Real-space cutoff (Angstrom) for the Born-radii descreening integral.
_BORN_CUTOFF = 10.0


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` roughly uniform unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0**0.5)
    theta = golden * i
    sin_phi = np.sin(phi)
    return np.stack(
        [np.cos(theta) * sin_phi, np.sin(theta) * sin_phi, np.cos(phi)], axis=1
    )


def _atom_radii(numbers: np.ndarray) -> np.ndarray:
    """Per-atom van der Waals radii (Angstrom) with a fallback for NaN."""
    from ase.data import vdw_radii

    r = vdw_radii[numbers].astype(float).copy()
    r[np.isnan(r)] = _DEFAULT_VDW
    return r


def _hct_scales(numbers: np.ndarray) -> np.ndarray:
    """Per-atom HCT descreening scale factors (default for unlisted elements)."""
    return np.array([_HCT_SCALE.get(int(z), _HCT_DEFAULT) for z in numbers])


def solvent_accessible_surface_area(
    atoms: Any,
    probe: float = _PROBE,
    n_points: int = _N_POINTS,
    sphere: Optional[np.ndarray] = None,
) -> float:
    """Total solvent-accessible surface area (Angstrom^2), PBC aware.

    Uses the Shrake-Rupley algorithm: each atom carries a sphere of expanded
    radius ``r_vdw + probe`` sampled by ``n_points`` points; a point counts as
    exposed if it lies outside every neighbouring atom's expanded sphere.
    Neighbour vectors include periodic images, so the area is correct for
    slabs periodic in plane.
    """
    from ase.neighborlist import neighbor_list

    numbers = atoms.get_atomic_numbers()
    radii = _atom_radii(numbers) + probe
    n = len(atoms)
    if n == 0:
        return 0.0

    unit = sphere if sphere is not None else _fibonacci_sphere(n_points)
    n_pts = len(unit)

    rmax = float(radii.max())
    # Two atoms can occlude each other only within Ri + Rj <= 2 * rmax.
    idx_i, idx_j, vec_ij = neighbor_list("ijD", atoms, 2.0 * rmax)

    total = 0.0
    for i in range(n):
        ri = radii[i]
        sel = idx_i == i
        d_ij = vec_ij[sel]
        r_j = radii[idx_j[sel]]
        # Keep only neighbours whose expanded sphere can reach atom i's surface.
        dist = np.linalg.norm(d_ij, axis=1)
        overlap = dist < (ri + r_j)
        d_ij = d_ij[overlap]
        r_j = r_j[overlap]

        pts = unit * ri  # surface points relative to atom i
        exposed = np.ones(n_pts, dtype=bool)
        for k in range(len(r_j)):
            d2 = np.sum((pts - d_ij[k]) ** 2, axis=1)
            exposed &= d2 > r_j[k] ** 2

        total += (exposed.sum() / n_pts) * 4.0 * np.pi * ri**2

    return float(total)


def born_radii(atoms: Any, radii: Optional[np.ndarray] = None) -> np.ndarray:
    """Onufriev-Bashford-Case effective Born radii (Angstrom), PBC aware.

    Follows the OpenMM reference implementation: a pairwise HCT descreening
    integral per atom, rescaled by the OBC-II ``tanh`` correction.
    """
    from ase.neighborlist import neighbor_list

    numbers = atoms.get_atomic_numbers()
    if radii is None:
        radii = _atom_radii(numbers)
    rho = radii - _OFFSET  # offset radii
    inv_rho = 1.0 / rho
    scale = _hct_scales(numbers)

    integral = np.zeros(len(atoms))
    idx_i, idx_j, vec = neighbor_list("ijD", atoms, _BORN_CUTOFF)
    if len(idx_i):
        r = np.linalg.norm(vec, axis=1)
        oi = rho[idx_i]  # offsetRadiusI
        sr_j = rho[idx_j] * scale[idx_j]  # scaledRadiusJ
        rsj = r + sr_j

        valid = oi < rsj
        big_l = np.maximum(oi, np.abs(r - sr_j))
        ell = 1.0 / big_l
        upp = 1.0 / np.where(valid, rsj, 1.0)
        term = (
            ell
            - upp
            + 0.25 * r * (upp * upp - ell * ell)
            + 0.5 / r * np.log(upp / ell)
            + 0.25 * sr_j * sr_j / r * (ell * ell - upp * upp)
        )
        inside = oi < (sr_j - r)
        term = term + np.where(inside, 2.0 * (inv_rho[idx_i] - ell), 0.0)
        term = np.where(valid, term, 0.0)
        np.add.at(integral, idx_i, term)

    psi = 0.5 * rho * integral
    tanh_sum = np.tanh(_OBC_ALPHA * psi - _OBC_BETA * psi**2 + _OBC_GAMMA * psi**3)
    return 1.0 / (inv_rho - tanh_sum / radii)


def born_polar_energy(
    atoms: Any,
    charges: np.ndarray,
    eb_k: float,
    radii: Optional[np.ndarray] = None,
    eps_in: float = _EPS_IN,
) -> float:
    """Generalized-Born polar solvation energy (eV), OBC radii.

    ``E = -1/2 k_e (1/eps_in - 1/eb_k) sum_ij q_i q_j / f_GB``, with OBC Born
    radii (:func:`born_radii`) and the Still ``f_GB``. ``charges`` are the EEQ
    partial charges and ``eb_k`` the solvent dielectric. Cross terms use the
    minimum-image distance (one nearest image per pair), not the periodic
    lattice sum; see the module docstring.
    """
    born = born_radii(atoms, radii=radii)
    pre = -0.5 * _KE * (1.0 / eps_in - 1.0 / eb_k)

    # Self (Born) terms.
    energy = float(np.sum(charges * charges / born))

    # Cross terms, minimum-image (nearest image only), each unordered pair once;
    # the factor 2 restores the full ordered double sum.
    n = len(atoms)
    if n > 1:
        dist = atoms.get_all_distances(mic=True)
        iu, ju = np.triu_indices(n, k=1)
        r2 = dist[iu, ju] ** 2
        ri_rj = born[iu] * born[ju]
        f_gb = np.sqrt(r2 + ri_rj * np.exp(-r2 / (4.0 * ri_rj)))
        energy += 2.0 * float(np.sum(charges[iu] * charges[ju] / f_gb))

    return pre * energy


def _has_open_direction(atoms: Any, min_vacuum: float = _MIN_VACUUM) -> bool:
    """True if the cell is non-periodic or has a vacuum gap (slab/cluster)."""
    pbc = atoms.get_pbc()
    if not pbc.any():
        return True
    lengths = atoms.cell.lengths()
    frac = atoms.get_scaled_positions(wrap=True)
    for d in range(3):
        if not pbc[d]:
            return True
        span = float(frac[:, d].max() - frac[:, d].min())
        if (1.0 - span) * lengths[d] >= min_vacuum:
            return True
    return False


class SolvationCalculator(Calculator):
    """Implicit-solvation term (nonpolar SASA + optional polar GB), slab only.

    Parameters
    ----------
    tau
        Surface tension in meV/Angstrom^2 (VASPsol ``TAU`` convention) for the
        nonpolar cavitation term ``E = TAU * SASA``.
    eb_k
        Solvent dielectric constant (VASPsol ``EB_K``). When ``None`` only the
        nonpolar term is computed; when set, the polar Generalized-Born term is
        added (requires the ``dftd4`` backend for EEQ charges). ``eb_k = 1``
        disables the polar term (zero dielectric contrast).
    probe
        Solvent probe radius in Angstrom (default 1.4, water).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        tau: float = 0.525,
        eb_k: Optional[float] = None,
        probe: float = _PROBE,
        n_points: int = _N_POINTS,
        fd_step: float = _FD_STEP,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = tau  # meV/Angstrom^2
        self.eb_k = eb_k
        self.probe = probe
        self.fd_step = fd_step
        self._sphere = _fibonacci_sphere(n_points)
        self._stress_warned = False

    def _energy(self, atoms: Any) -> float:
        """Solvation energy in eV for the given geometry."""
        area = solvent_accessible_surface_area(
            atoms, probe=self.probe, sphere=self._sphere
        )
        energy = self.tau * 1.0e-3 * area  # meV -> eV

        if self.eb_k is not None:
            from .charges import eeq_charges

            charges = eeq_charges(atoms)
            energy += born_polar_energy(atoms, charges, self.eb_k)

        return energy

    def _forces_fd(self, atoms: Any) -> np.ndarray:
        h = self.fd_step
        base = atoms.get_positions()
        forces = np.zeros_like(base)
        work = atoms.copy()
        for i in range(len(atoms)):
            for d in range(3):
                pos = base.copy()
                pos[i, d] += h
                work.set_positions(pos)
                e_plus = self._energy(work)
                pos[i, d] = base[i, d] - h
                work.set_positions(pos)
                e_minus = self._energy(work)
                forces[i, d] = -(e_plus - e_minus) / (2.0 * h)
        return forces

    def calculate(
        self,
        atoms: Any = None,
        properties: Tuple[str, ...] = ("energy",),
        system_changes: Any = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if not _has_open_direction(self.atoms):
            raise ValueError(
                "LSOL=.TRUE. (implicit solvation) requires a slab, cluster, or "
                "molecule with an exposed surface. The cell is fully periodic "
                "with no vacuum gap (>= "
                f"{_MIN_VACUUM} Angstrom), so there is no solvent-accessible "
                "surface. Solvation is not supported for dense 3D bulk."
            )

        self.results["energy"] = self._energy(self.atoms)
        self.results["forces"] = self._forces_fd(self.atoms)

        # Stress from the solvation terms is not yet implemented; return zero so
        # ISIF >= 3 cell relaxation still runs (solvation simply does not
        # contribute to the stress). See not_for_release/solvation_design.md.
        if not self._stress_warned:
            warnings.warn(
                "Solvation stress is not implemented; contributing zero "
                "stress (relevant only for ISIF >= 3).",
                stacklevel=2,
            )
            self._stress_warned = True
        self.results["stress"] = np.zeros(6, dtype=float)
