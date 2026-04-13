#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/HyperCT_UPDT/logs}"
SBATCH_GPU_CONSTRAINT="${SBATCH_GPU_CONSTRAINT:-nvidia_h100}"
mkdir -p "$LOG_DIR"

submit_job() {
    local dependency="$1"
    local script_path="$2"
    local sbatch_args=(--parsable --chdir="$PROJECT_DIR/HyperCT_UPDT")

    if [[ -n "${SBATCH_GPU_CONSTRAINT:-}" ]]; then
        sbatch_args+=(--constraint="$SBATCH_GPU_CONSTRAINT")
    fi

    if [[ -n "$dependency" ]]; then
        sbatch_args+=(--dependency="afterok:$dependency")
        sbatch "${sbatch_args[@]}" "$script_path"
    else
        sbatch "${sbatch_args[@]}" "$script_path"
    fi
}

echo "Submitting HyperCT pipeline from $PROJECT_DIR"

HYPERNET_JOB_ID="$(submit_job "" "$SCRIPT_DIR/train_hypernet.sh")"
echo "train_hypernet.sh -> job $HYPERNET_JOB_ID"

PRECOMPUTE_JOB_ID="$(submit_job "$HYPERNET_JOB_ID" "$SCRIPT_DIR/precompute.sh")"
echo "precompute.sh -> job $PRECOMPUTE_JOB_ID (afterok:$HYPERNET_JOB_ID)"

VLM_JOB_ID="$(submit_job "$PRECOMPUTE_JOB_ID" "$SCRIPT_DIR/train_vlm.sh")"
echo "train_vlm.sh -> job $VLM_JOB_ID (afterok:$PRECOMPUTE_JOB_ID)"

cat <<EOF

Submitted pipeline:
  hypernet   $HYPERNET_JOB_ID
  precompute $PRECOMPUTE_JOB_ID
  vlm        $VLM_JOB_ID

Useful commands:
  squeue -u \$USER
  sacct -j ${VLM_JOB_ID} --format=JobID,JobName,Partition,State,Elapsed,MaxRSS
EOF
