#!/usr/bin/env bash
# full-CUDA windows run
#
set -euo pipefail

export GFLOW_WAKE_AGE_MAX_REV=5

# ---------------------------------------------------------------------------
export GFLOW_FMM_GPU=2

export GFLOW_NU_EFFECTIVE=1.469e-5

export GFLOW_QCRIT_GRID_N=64
export GFLOW_QCRIT_BOX_MIN="-0.6,0.0,-0.5"
export GFLOW_QCRIT_BOX_MAX="2.3,1.11,0.2"

export GFLOW_STRIP_CSV=1

# ---------------------------------------------------------------------------
export GFLOW_FUSELAGE_HS3D=1

# ---------------------------------------------------------------------------

export GFLOW_AIRFRAME_KUTTA=0
export GFLOW_KUTTA_AUDIT=1

export GFLOW_BODY_UNSTEADY=0


#---------------------------------------------------------------------------
export GFLOW_MESH_AUDIT=1      # watertightness report at startup; keep on permanently
export GFLOW_CLOSE_TE_SEAM=1   # close the blade trailing-edge seam


# ---------------------------------------------------------------------------
BIN=./VPMX
INPUT=VPMx_multirotor.i
OUTDIR=out_cuda_run
DT=0.0001
DEVICE=-1          # -1 = let the backend choose (or GFLOW_CUDA_DEVICE_INDEX)
BACKEND=cuda       # auto | cuda | opencl

mkdir -p "$OUTDIR"
echo "=== Case CUDA run ==="
echo "input=$INPUT out=$OUTDIR dt=$DT device=$DEVICE backend=$BACKEND"
env | grep -E '^GFLOW_' | sort | sed 's/^/  /'
echo

"$BIN" "$INPUT" "$OUTDIR" "$DT" "$DEVICE" "$BACKEND"
