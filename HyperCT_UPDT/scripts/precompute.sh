#!/bin/bash
#SBATCH --job-name=hyperct_precompute
#SBATCH --partition=cornell
#SBATCH --account=cornell
#SBATCH --qos=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/hyperct_precompute_%j.out
#SBATCH --error=logs/hyperct_precompute_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/lustre/cornell/$USER/source/Radiology_VLM_AI4ML}"
DATA_ROOT="${DATA_ROOT:-/mnt/lustre/cornell/$USER/data/ct_ratev2}"
DATASET_DIR="${DATASET_DIR:-$DATA_ROOT/data_volumes/dataset}"
INPUT_DATA_DIR="${INPUT_DATA_DIR:-$DATASET_DIR/train_fixed}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/HyperCT_UPDT/precomputed_tokens}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/HyperCT_UPDT/checkpoints/hypernet/best_checkpoint.pth}"

mkdir -p "$PROJECT_DIR/HyperCT_UPDT/logs" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
    module purge || true
    if [[ -n "${HYPERCT_MODULES:-cuda12.4/toolkit/12.4.1}" ]]; then
        for module_name in ${HYPERCT_MODULES:-cuda12.4/toolkit/12.4.1}; do
            module load "$module_name"
        done
    fi
fi

if [[ -n "${HYPERCT_ENV_ACTIVATE:-}" ]]; then
    eval "$HYPERCT_ENV_ACTIVATE"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${HYPERCT_CONDA_ENV:-hyperct_updt}"
fi

python -u HyperCT_UPDT/precompute_tokens.py \
    --data_dir "$INPUT_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --num_slices "${NUM_SLICES:-33}" \
    --slice_height "${SLICE_HEIGHT:-224}" \
    --slice_width "${SLICE_WIDTH:-224}" \
    --cube_pool_levels "${CUBE_POOL_LEVELS:-2}" \
    --encoder_name "${ENCODER_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}" \
    --lora_rank "${LORA_RANK:-16}" \
    --lora_scaling "${LORA_SCALING:-1.0}"
