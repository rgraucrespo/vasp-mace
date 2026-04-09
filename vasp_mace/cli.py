import os, sys, warnings, logging, time
os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
warnings.filterwarnings("ignore", message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"e3nn\.o3\._wigner")
warnings.filterwarnings("ignore", category=UserWarning, module=r"mace\.calculators\.mace")
for name in ("cuequivariance", "cuequivariance_torch", "e3nn", "mace"):
    logging.getLogger(name).setLevel(logging.ERROR)

import argparse
import numpy as np
from .incar import parse_incar
from .io_vasp import (
    read_poscar,
    write_oszicar,
    write_outcar,
    write_outcar_like,
    write_contcar,
    write_relax_vasprun_xml,
    write_single_vasprun_xml,
)
from .logging_utils import StepRecord
from .mace_loader import load_calc
from .relax import run_relax
from .md import run_md


def main():
    try:
        _run()
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)


def _run():
    ap = argparse.ArgumentParser(description="Minimal VASP-like MACE simulator")
    DEFAULT_MODEL = os.environ.get(
        "MACE_MODEL_PATH",
        os.path.expanduser("~/software/mace/2024-01-07-mace-128-L2_epoch-199.model"),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Path to MACE .model checkpoint (default: {DEFAULT_MODEL} or $MACE_MODEL_PATH)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])
    ap.add_argument("--optimizer", default="BFGS", choices=["BFGS", "FIRE", "LBFGS"])
    args = ap.parse_args()

    # Read inputs first so NSW/IBRION decides the mode
    cfg = parse_incar("INCAR")
    atoms = read_poscar("POSCAR")

    # Load calculator; IVDW > 0 enables empirical dispersion (DFT-D3)
    dispersion = cfg.IVDW > 0
    calc, device, dtype = load_calc(args.model, device=args.device, dtype=args.dtype,
                                    dispersion=dispersion)
    atoms.calc = calc

    # --- IBRION=0: MD mode ---
    if cfg.IBRION == 0:
        print(
            f"[info] Model={args.model}, device={device}, dtype={dtype}, "
            f"MDALGO={cfg.MDALGO}, NSW={cfg.NSW}, TEBEG={cfg.TEBEG} K, "
            f"POTIM={cfg.POTIM} fs, NBLOCK={cfg.NBLOCK}"
        )
        records = run_md(atoms, calc, cfg)
        write_contcar("CONTCAR", atoms)
        print(f"[done] MD complete ({len(records)} steps). Wrote XDATCAR, CONTCAR.")
        return

    # --- NSW=0: single-point ---
    if cfg.NSW <= 0:
        t0_wall = time.time()
        t0_cpu  = time.process_time()

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        f = np.asarray(forces, dtype=float)
        fmax = float(np.max(np.linalg.norm(f, axis=1)))

        stress = None
        try:
            stress = atoms.get_stress(voigt=True).tolist()
        except Exception:
            pass

        elapsed = time.time()         - t0_wall
        cpu_t   = time.process_time() - t0_cpu

        write_single_vasprun_xml("vasprun.xml", atoms, forces=f, stress=stress, energy=energy)

        sv = np.array(stress) if stress is not None else None
        one = StepRecord(
            n=1, energy=float(energy), dE=float(energy), fmax=fmax,
            positions=atoms.get_positions().copy(),
            forces=f.copy(),
            stress=sv,
            cell=np.array(atoms.get_cell()).copy(),
        )
        write_outcar("OUTCAR", atoms, [one], incar_raw=cfg.raw,
                     converged=True, elapsed=elapsed, cpu_time=cpu_t)
        write_oszicar("OSZICAR", [one])

        print("[done] Single-point written (NSW=0): vasprun.xml, OUTCAR, OSZICAR")
        return

    # --- NSW > 0: relaxation ---
    pressure_GPa = cfg.PSTRESS * 0.1
    print(
        f"[info] Model={args.model}, device={device}, dtype={dtype}, "
        f"ISIF={cfg.ISIF}, NSW={cfg.NSW}, EDIFFG={cfg.EDIFFG}, "
        f"PSTRESS={cfg.PSTRESS} kBar ({pressure_GPa:.3f} GPa), "
        f"IVDW={cfg.IVDW}"
    )

    atoms_initial = atoms.copy()  # snapshot before relaxation for vasprun.xml initialpos

    t0_wall = time.time()
    t0_cpu  = time.process_time()

    steps, converged = run_relax(atoms, calc, cfg, optimizer=args.optimizer,
                                 pressure_GPa=pressure_GPa)

    elapsed = time.time()         - t0_wall
    cpu_t   = time.process_time() - t0_cpu

    write_outcar("OUTCAR", atoms, steps, incar_raw=cfg.raw,
                 converged=converged, elapsed=elapsed, cpu_time=cpu_t)
    write_oszicar("OSZICAR", steps)
    write_contcar("CONTCAR", atoms)
    write_relax_vasprun_xml("vasprun.xml", atoms_initial, atoms, steps, incar_raw=cfg.raw)

    print("[done] Wrote OSZICAR, OUTCAR, CONTCAR, vasprun.xml")
    if not converged:
        print("[note] Reached NSW without meeting convergence criterion.")
