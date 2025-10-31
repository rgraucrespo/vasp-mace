import os, warnings, logging
os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
warnings.filterwarnings("ignore", message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"e3nn\.o3\._wigner")
warnings.filterwarnings("ignore", category=UserWarning, module=r"mace\.calculators\.mace")
for name in ("cuequivariance", "cuequivariance_torch", "e3nn", "mace"):
    logging.getLogger(name).setLevel(logging.ERROR)

import argparse
from .incar import parse_incar
from .io_vasp import (
    read_poscar,
    write_oszicar,
    write_outcar_like,
    write_contcar,
    write_relax_vasprun_xml,
    write_single_vasprun_xml
)

from .mace_loader import load_calc
from .relax import run_relax
import numpy as np
from types import SimpleNamespace

def main():
    ap = argparse.ArgumentParser(description="Minimal VASP-like MACE simulator")
    DEFAULT_MODEL = os.environ.get(
        "MACE_MODEL_PATH",
        os.path.expanduser("~/software/mace/2024-01-07-mace-128-L2_epoch-199.model"),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Path to MACE .model checkpoint (default: {DEFAULT_MODEL} or $MACE_MODEL_PATH)")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","mps"])
    ap.add_argument("--dtype", default="auto", choices=["auto","float32","float64"])
    ap.add_argument("--optimizer", default="BFGS", choices=["BFGS","FIRE"])
    args = ap.parse_args()

    # Read inputs first so NSW decides the mode
    cfg = parse_incar("INCAR")
    atoms = read_poscar("POSCAR")

    # Load calculator
    calc, device, dtype = load_calc(args.model, device=args.device, dtype=args.dtype)
    atoms.calc = calc


    # --- NSW logic: 0 => single-point; >0 => relaxation ---

    if cfg.NSW <= 0:
        # Single-point calculation
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()                # (N, 3) ndarray from ASE
        f = np.asarray(forces, dtype=float)        # be defensive
        fmax = float(np.max(np.linalg.norm(f, axis=1)))

        stress = None
        try:
            stress = atoms.get_stress(voigt=True).tolist()  # 6 comps in eV/Å^3
        except Exception:
            pass

        # Write ShengBTE/Phonopy-compatible vasprun.xml
        write_single_vasprun_xml("vasprun.xml", atoms, forces=f, stress=stress, energy=energy)

        # Tiny OUTCAR + OSZICAR for sanity (one “ionic” record)
        one = SimpleNamespace(n=1, energy=float(energy), dE=0.0, fmax=fmax)
        write_outcar_like("OUTCAR", atoms, [one], stress=stress)
        write_oszicar("OSZICAR", [one])

        print("[done] Single-point written (NSW=0): vasprun.xml, OUTCAR, OSZICAR")
        return

    # Relaxation calculation (NSW >0)
    pressure_GPa = cfg.PSTRESS * 0.1
    print(f"[info] Model={args.model}, device={device}, dtype={dtype}, "
      f"ISIF={cfg.ISIF}, NSW={cfg.NSW}, EDIFFG={cfg.EDIFFG}, "
      f"PSTRESS={cfg.PSTRESS} kBar ({pressure_GPa:.3f} GPa)")
    steps, converged = run_relax(atoms, calc, cfg, optimizer=args.optimizer, pressure_GPa=pressure_GPa)

    stress = None
    try:
        stress = atoms.get_stress(voigt=True).tolist()  # 6 comps in eV/Å^3
    except Exception:
        pass

    write_outcar_like("OUTCAR", atoms, steps, stress=stress)
    write_oszicar("OSZICAR", steps)
    write_contcar("CONTCAR", atoms)
    write_relax_vasprun_xml("vasprun.xml", atoms, steps)

    print("[done] Wrote OSZICAR, OUTCAR, CONTCAR, vasprun.xml")
    if not converged:
        print("[note] Reached NSW without meeting convergence criterion.]")

