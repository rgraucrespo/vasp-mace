import os, sys, warnings, logging, time

os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
warnings.filterwarnings(
    "ignore",
    message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"e3nn\.o3\._wigner")
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"mace\.calculators\.mace"
)
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
    write_outcar_tail,
    write_contcar,
    write_relax_vasprun_xml,
    write_single_vasprun_xml,
)
from .logging_utils import StepRecord
from .mace_loader import load_calc
from .relax import run_relax, EV_A3_TO_KBAR
from .md import run_md


def main() -> None:
    """Run the ``vasp-mace`` command-line interface.

    The function wraps the internal dispatcher, prints a concise ``[error]``
    message for uncaught exceptions, and exits with status code 1 on failure.
    It is exposed as the package console-script entry point.
    """
    try:
        _run()
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)


def _run() -> None:
    ap = argparse.ArgumentParser(description="Minimal VASP-like MACE simulator")
    DEFAULT_MODEL = os.environ.get(
        "MACE_MODEL_PATH",
        os.path.expanduser("~/software/mace/2024-01-07-mace-128-L2_epoch-199.model"),
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to MACE .model checkpoint (default: {DEFAULT_MODEL} or $MACE_MODEL_PATH)",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])
    ap.add_argument("--optimizer", default="BFGS", choices=["BFGS", "FIRE", "LBFGS"])
    args = ap.parse_args()

    cfg = parse_incar("INCAR")

    # ML_LHEAT is meaningful only for MD (IBRION=0). If it's set elsewhere,
    # ignore the keyword but tell the user — silently swallowing it would
    # mask a typo'd INCAR.
    if cfg.ML_LHEAT and cfg.IBRION != 0:
        print(
            f"[warn] ML_LHEAT=.TRUE. is only meaningful for MD (IBRION=0); "
            f"IBRION={cfg.IBRION} detected. Ignoring ML_LHEAT."
        )

    # --- IMAGES > 0: NEB mode -----------------------------------------------
    # Must be checked before read_poscar: NEB has no top-level POSCAR;
    # images live in 00/, 01/, … subdirectories.
    if cfg.IMAGES > 0:
        from .neb import run_neb

        dispersion = cfg.IVDW > 0
        n_total = cfg.IMAGES + 2
        image_steps, converged = run_neb(
            cfg,
            args.model,
            device=args.device,
            dtype=args.dtype,
            dispersion=dispersion,
            optimizer=args.optimizer,
        )
        print(
            f"[done] NEB complete. "
            f"Wrote 00/ … {n_total - 1:02d}/ (CONTCAR, OUTCAR, OSZICAR, vasprun.xml)."
        )
        if not converged:
            print("[note] Reached NSW without meeting NEB convergence criterion.")
        return

    # Non-NEB modes: read top-level POSCAR and load a single calculator
    atoms = read_poscar("POSCAR")

    # Load calculator; IVDW > 0 enables empirical dispersion (DFT-D3)
    dispersion = cfg.IVDW > 0

    # ML_LHEAT requires a float64 heat-flux backend (mace-unfolded has a
    # float32 dtype-mismatch bug). Constructing two MACECalculators with
    # different dtypes in the same process pollutes torch's global default
    # dtype and the integrator's float32 batch ends up touching the
    # float64 unfolded graph mid-run. Avoid that by forcing the main MD
    # calculator to float64 too whenever ML_LHEAT is on.
    main_dtype = args.dtype
    if cfg.ML_LHEAT and cfg.IBRION == 0 and args.dtype != "float64":
        if args.dtype != "auto":
            print(
                f"[note] ML_LHEAT: forcing main calculator to float64 "
                f"(--dtype={args.dtype} ignored). mace-unfolded has a "
                f"float32 dtype bug, and mixing float32/float64 calculators "
                f"in the same process is unreliable."
            )
        main_dtype = "float64"

    calc, device, dtype = load_calc(
        args.model, device=args.device, dtype=main_dtype, dispersion=dispersion
    )
    atoms.calc = calc

    # --- IBRION=5/6: phonon finite differences ---
    if cfg.IBRION in (5, 6):
        if "POTIM" not in cfg.raw:
            cfg.POTIM = 0.015
            print(
                f"[info] POTIM not set; using VASP default 0.015 Å for phonon displacement."
            )
        print(
            f"[info] Model={args.model}, device={device}, dtype={dtype}, "
            f"IBRION={cfg.IBRION}, NFREE={cfg.NFREE}, POTIM={cfg.POTIM} Å"
        )
        if cfg.PSTRESS != 0.0:
            print(
                f"[info] PSTRESS={cfg.PSTRESS} kBar detected in INCAR. Note that PSTRESS does "
                f"not have any effect on elasticity calculations. Any hydrostatic pressure effect "
                f"should be incorporated using PSTRESS during the geometry relaxation step. "
                f"If the structure was pre-relaxed at this pressure its cell will carry the "
                f"corresponding internal stress, and the elastic constants will reflect the "
                f"material response at that compressed/expanded volume."
            )
        t0_wall = time.time()
        t0_cpu = time.process_time()
        from .phonons import run_phonons

        run_phonons(atoms, calc, cfg)
        elapsed = time.time() - t0_wall
        cpu_t = time.process_time() - t0_cpu
        write_outcar_tail("OUTCAR", elapsed, cpu_t)
        return

    # --- IBRION=0: MD mode ---
    if cfg.IBRION == 0:
        if cfg.ML_LHEAT and cfg.ISIF == 3:
            print(
                "[note] ML_LHEAT=.TRUE. combined with ISIF=3 (NPT): the cell "
                "volume drifts during the run, so the volume_A3 recorded in "
                "ML_HEAT.json reflects the initial cell only."
            )

        extra_info = ""
        if cfg.MDALGO == 1:
            extra_info = f", ANDERSEN_PROB={cfg.ANDERSEN_PROB}"
        elif cfg.MDALGO == 2:
            extra_info = f", SMASS={cfg.SMASS}"
        elif cfg.MDALGO == 3:
            gamma_str = " ".join(str(g) for g in cfg.LANGEVIN_GAMMA)
            extra_info = f", LANGEVIN_GAMMA={gamma_str} ps⁻¹"

        print(
            f"[info] Model={args.model}, device={device}, dtype={dtype}, "
            f"MDALGO={cfg.MDALGO}, NSW={cfg.NSW}, TEBEG={cfg.TEBEG} K, "
            f"POTIM={cfg.POTIM} fs, NBLOCK={cfg.NBLOCK}{extra_info}"
        )
        t0_wall = time.time()
        t0_cpu = time.process_time()
        records = run_md(
            atoms, calc, cfg, model_path=args.model, device=device, dtype=dtype
        )
        elapsed = time.time() - t0_wall
        cpu_t = time.process_time() - t0_cpu
        write_contcar("CONTCAR", atoms)
        write_outcar_tail("OUTCAR", elapsed, cpu_t)
        outputs = "XDATCAR, CONTCAR, OUTCAR"
        if cfg.ML_LHEAT:
            outputs += ", ML_HEAT, ML_HEAT.json"
        print(f"[done] MD complete ({len(records)} steps). Wrote {outputs}.")
        return

    # --- NSW=0: single-point ---
    if cfg.NSW <= 0:
        t0_wall = time.time()
        t0_cpu = time.process_time()

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        f = np.asarray(forces, dtype=float)
        fmax = float(np.max(np.linalg.norm(f, axis=1)))

        stress = None
        try:
            stress = atoms.get_stress(voigt=True).tolist()
        except Exception:
            pass

        elapsed = time.time() - t0_wall
        cpu_t = time.process_time() - t0_cpu

        write_single_vasprun_xml(
            "vasprun.xml",
            atoms,
            forces=f,
            stress=stress,
            energy=energy,
            incar_raw=cfg.raw,
        )
        write_contcar("CONTCAR", atoms)

        sv = np.array(stress) if stress is not None else None
        one = StepRecord(
            n=1,
            energy=float(energy),
            dE=float(energy),
            fmax=fmax,
            positions=atoms.get_positions().copy(),
            forces=f.copy(),
            stress=sv,
            cell=np.array(atoms.get_cell()).copy(),
        )
        write_outcar(
            "OUTCAR",
            atoms,
            [one],
            incar_raw=cfg.raw,
            converged=True,
            elapsed=elapsed,
            cpu_time=cpu_t,
        )
        write_oszicar("OSZICAR", [one])

        stress_str = ""
        if stress is not None:
            max_sig = float(np.max(np.abs(stress)))
            stress_str = f", max|σ|={max_sig*EV_A3_TO_KBAR:.3f} kBar"

        print(
            f"[done] Single-point written (NSW=0): E={energy:.6f} eV, Fmax={fmax:.3f} eV/Å{stress_str}"
        )
        return

    # --- NSW > 0: relaxation ---
    pressure_GPa = cfg.PSTRESS * 0.1
    print(
        f"[info] Model={args.model}, device={device}, dtype={dtype}, "
        f"ISIF={cfg.ISIF}, NSW={cfg.NSW}, EDIFFG={cfg.EDIFFG}, "
        f"PSTRESS={cfg.PSTRESS} kBar ({pressure_GPa:.3f} GPa), "
        f"IVDW={cfg.IVDW}"
    )

    atoms_initial = (
        atoms.copy()
    )  # snapshot before relaxation for vasprun.xml initialpos

    t0_wall = time.time()
    t0_cpu = time.process_time()

    steps, converged = run_relax(
        atoms, calc, cfg, optimizer=args.optimizer, pressure_GPa=pressure_GPa
    )

    elapsed = time.time() - t0_wall
    cpu_t = time.process_time() - t0_cpu

    write_outcar(
        "OUTCAR",
        atoms,
        steps,
        incar_raw=cfg.raw,
        converged=converged,
        elapsed=elapsed,
        cpu_time=cpu_t,
    )
    write_oszicar("OSZICAR", steps)
    write_contcar("CONTCAR", atoms)
    write_relax_vasprun_xml(
        "vasprun.xml", atoms_initial, atoms, steps, incar_raw=cfg.raw
    )

    print("[done] Wrote OSZICAR, OUTCAR, CONTCAR, vasprun.xml")
    if not converged:
        print("[note] Reached NSW without meeting convergence criterion.")
