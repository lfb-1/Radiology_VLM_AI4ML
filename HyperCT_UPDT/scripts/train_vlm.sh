#!/bin/bash
#SBATCH --job-name=hyperct_train_vlm
#SBATCH -p sablab-gpu-low
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=hyperct_train_vlm_%j.out
#SBATCH --error=hyperct_train_vlm_%j.err

set -euo pipefail
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate test

# Install dependencies (confirmed working setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops open_clip_torch timm deepspeed ninja
pip install flash-attn
pip install "transformers>=4.34.0" nibabel tqdm
pip install "numpy<2"
pip install --upgrade peft
pip install --upgrade pip wheel
pip install --force-reinstall --no-deps markupsafe==3.0.3

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/HyperCT_UPDT
cd "$PROJECT_DIR"

torchrun --nproc_per_node=4 train_vlm.py \
    --tokens_dir ./precomputed_tokens \
    --data_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/train_vqa.json \
    --output_dir ./checkpoints/hyperct_vlm \
    --llm_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_hidden_size 4096 \
    --vision_dim 768 \
    --num_queries 64 \
    --qformer_layers 6 \
    --qformer_heads 12 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lr 2e-5 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --max_length 2048 \
    --num_task_tokens 3 \
    --bf16 \
    --attn_implementation eager
