"""Implicit-solvation correction terms (slab/cluster only).

Stage 3a implements the **nonpolar / cavitation** term only:

    E_nonpolar = TAU * SASA

where ``SASA`` is the solvent-accessible surface area (Shrake-Rupley, PBC
aware) and ``TAU`` is an effective surface tension (INCAR tag, meV/Angstrom^2,
VASPsol convention). The polar/electrostatic (Generalized-Born) term that uses
the shared EEQ charges (vasp_mace.charges) is deferred to Stage 3b.

This is a density-free surrogate, NOT VASPsol's Poisson-Boltzmann model. It is
meaningful only where there is an exposed surface, so it is restricted to
slabs / clusters / molecules; a fully periodic 3D bulk cell (no vacuum) is
rejected.

Forces are evaluated by central finite differences of the solvation energy.
This is deliberate: it keeps the (later) GB term's charge-position coupling
(dq/dr) automatic, since the EEQ charges are recomputed at every displaced
geometry. Stress is not yet included (returned as zero with a warning) for
ISIF >= 3; see not_for_release/solvation_design.md.
"""

from typing import Any, Optional, Tuple
import warnings

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

# vasp_mace/solvation.py

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


class SASASolvationCalculator(Calculator):
    """Nonpolar implicit-solvation term ``E = TAU * SASA`` (slab/cluster only).

    Parameters
    ----------
    tau
        Effective surface tension in meV/Angstrom^2 (VASPsol ``TAU`` convention).
    probe
        Solvent probe radius in Angstrom (default 1.4, water).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        tau: float,
        probe: float = _PROBE,
        n_points: int = _N_POINTS,
        fd_step: float = _FD_STEP,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = tau  # meV/Angstrom^2
        self.probe = probe
        self.fd_step = fd_step
        self._sphere = _fibonacci_sphere(n_points)
        self._stress_warned = False

    def _energy(self, atoms: Any) -> float:
        """Solvation energy in eV for the given geometry."""
        area = solvent_accessible_surface_area(
            atoms, probe=self.probe, sphere=self._sphere
        )
        return self.tau * 1.0e-3 * area  # meV -> eV

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

        # Stress from the cavitation term is not yet implemented; return zero so
        # ISIF >= 3 cell relaxation still runs (solvation simply does not
        # contribute to the stress). See not_for_release/solvation_design.md.
        if not self._stress_warned:
            warnings.warn(
                "SASA solvation stress is not implemented; contributing zero "
                "stress (relevant only for ISIF >= 3).",
                stacklevel=2,
            )
            self._stress_warned = True
        self.results["stress"] = np.zeros(6, dtype=float)
