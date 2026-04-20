#!/bin/bash
# NVT (Langevin) → NPT (Langevin) sequential MD for PbTe
# Usage: bash run.sh [--model /path/to/model] [extra vasp-mace flags]
set -e

VASP_MACE="vasp-mace $*"

# ── Stage 1: NVT ─────────────────────────────────────────────────────────────
echo "=== Stage 1: NVT Langevin ==="
cp INCAR_NVT INCAR
$VASP_MACE

mkdir -p nvt_output
cp CONTCAR XDATCAR ase_files/mace.traj ase_files/md.log nvt_output/ 2>/dev/null || true

# ── Stage 2: NPT ─────────────────────────────────────────────────────────────
echo "=== Stage 2: NPT Langevin ==="
cp CONTCAR POSCAR        # start NPT from the NVT final structure
cp INCAR_NPT INCAR
$VASP_MACE

mkdir -p npt_output
cp CONTCAR XDATCAR ase_files/mace.traj ase_files/md.log npt_output/ 2>/dev/null || true

echo "=== Done. NVT outputs in nvt_output/, NPT outputs in npt_output/ ==="
