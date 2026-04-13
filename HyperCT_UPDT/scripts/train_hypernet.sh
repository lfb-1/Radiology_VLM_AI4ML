#!/bin/bash
#SBATCH --job-name=hyperct_train_hypernet
#SBATCH --partition=cornell
#SBATCH --account=cornell
#SBATCH --qos=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/hyperct_train_hypernet_%j.out
#SBATCH --error=logs/hyperct_train_hypernet_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/lustre/cornell/$USER/source/Radiology_VLM_AI4ML}"
DATA_ROOT="${DATA_ROOT:-/mnt/lustre/cornell/$USER/data/ct_ratev2}"
DATASET_DIR="${DATASET_DIR:-$DATA_ROOT/data_volumes/dataset}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$DATASET_DIR/train_fixed}"
VAL_DATA_DIR="${VAL_DATA_DIR:-$DATASET_DIR/valid_fixed}"
TRAIN_JSON="${TRAIN_JSON:-$DATASET_DIR/vqa/train_vqa.json}"
VAL_JSON="${VAL_JSON:-$DATASET_DIR/vqa/valid_vqa.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/HyperCT_UPDT/checkpoints/hypernet}"
CHECKPOINT="${CHECKPOINT:-}"

mkdir -p "$PROJECT_DIR/HyperCT_UPDT/logs" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
    module purge || true
    if [[ -n "${HYPERCT_MODULES:-cuda12.4/toolkit/12.4.1}" ]]; then
        # Example: export HYPERCT_MODULES="cuda/12.4.1"
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

extra_args=()
if [[ -n "$CHECKPOINT" ]]; then
    extra_args+=(--checkpoint "$CHECKPOINT")
fi

python -u HyperCT_UPDT/train_hypernet.py \
    --data_dir "$TRAIN_DATA_DIR" \
    --labels_json "$TRAIN_JSON" \
    --val_labels_json "$VAL_JSON" \
    --val_data_dir "$VAL_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --encoder_name "${ENCODER_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}" \
    --lora_rank "${LORA_RANK:-16}" \
    --lora_scaling "${LORA_SCALING:-1.0}" \
    --num_slices "${NUM_SLICES:-33}" \
    --slice_height "${SLICE_HEIGHT:-224}" \
    --slice_width "${SLICE_WIDTH:-224}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --lr "${LR:-1e-5}" \
    --weight_decay "${WEIGHT_DECAY:-1e-2}" \
    --epochs "${EPOCHS:-20}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --max_batches_per_epoch "${MAX_BATCHES_PER_EPOCH:-5000}" \
    --seed "${SEED:-42}" \
    "${extra_args[@]}"
