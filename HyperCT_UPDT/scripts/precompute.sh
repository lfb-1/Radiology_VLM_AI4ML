#!/bin/bash
#SBATCH --job-name=hyperct_precompute
#SBATCH -p sablab-gpu-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hyperct_precompute_%j.out
#SBATCH --error=hyperct_precompute_%j.err

set -euo pipefail
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate test

# Install dependencies (confirmed working setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops open_clip_torch timm deepspeed ninja
pip install flash-attn
pip install "transformers>=4.56.0" nibabel tqdm
pip install "numpy<2"
pip install --upgrade peft
pip install --upgrade pip wheel
pip install --force-reinstall --no-deps markupsafe==3.0.3

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/HyperCT_UPDT
cd "$PROJECT_DIR"

python precompute_tokens.py \
    --data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/train \
    --output_dir ./precomputed_tokens \
    --num_slices 33 \
    --slice_height 512 \
    --slice_width 512 \
    --cube_pool_levels 2 \
    --encoder_name facebook/dinov2-base \
    --lora_rank 16 \
    --lora_scaling 1.0
