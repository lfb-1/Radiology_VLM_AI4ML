#!/bin/bash
#SBATCH --job-name=hyperct_train_vlm
#SBATCH --partition=cornell
#SBATCH --account=cornell
#SBATCH --qos=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/hyperct_train_vlm_%j.out
#SBATCH --error=logs/hyperct_train_vlm_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/lustre/cornell/$USER/source/Radiology_VLM_AI4ML}"
DATA_ROOT="${DATA_ROOT:-/mnt/lustre/cornell/$USER/data/ct_ratev2}"
DATASET_DIR="${DATASET_DIR:-$DATA_ROOT/data_volumes/dataset}"
TOKENS_DIR="${TOKENS_DIR:-$PROJECT_DIR/HyperCT_UPDT/precomputed_tokens}"
TRAIN_JSON="${TRAIN_JSON:-$DATASET_DIR/vqa/train_vqa.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/HyperCT_UPDT/checkpoints/hyperct_vlm}"
QFORMER_CHECKPOINT="${QFORMER_CHECKPOINT:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
BF16_ARGS=(--bf16)

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

if [[ -n "${DISABLE_BF16:-}" ]]; then
    BF16_ARGS=()
fi

extra_args=()
if [[ -n "$QFORMER_CHECKPOINT" ]]; then
    extra_args+=(--qformer_checkpoint "$QFORMER_CHECKPOINT")
fi
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
    extra_args+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$GPUS_PER_NODE}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    HyperCT_UPDT/train_vlm.py \
    --tokens_dir "$TOKENS_DIR" \
    --data_json "$TRAIN_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --llm_name "${LLM_NAME:-meta-llama/Llama-3.1-8B-Instruct}" \
    --llm_hidden_size "${LLM_HIDDEN_SIZE:-4096}" \
    --vision_dim "${VISION_DIM:-768}" \
    --num_queries "${NUM_QUERIES:-64}" \
    --qformer_layers "${QFORMER_LAYERS:-6}" \
    --qformer_heads "${QFORMER_HEADS:-12}" \
    --lora_r "${LLM_LORA_R:-128}" \
    --lora_alpha "${LLM_LORA_ALPHA:-256}" \
    --lora_dropout "${LLM_LORA_DROPOUT:-0.05}" \
    --lr "${VLM_LR:-2e-5}" \
    --epochs "${VLM_EPOCHS:-3}" \
    --batch_size "${VLM_BATCH_SIZE:-1}" \
    --grad_accum "${GRAD_ACCUM:-8}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --num_task_tokens "${NUM_TASK_TOKENS:-3}" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
    "${BF16_ARGS[@]}" \
    "${extra_args[@]}"
