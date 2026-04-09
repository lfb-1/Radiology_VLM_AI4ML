#!/bin/bash
#SBATCH --job-name=hyperct_train_hypernet
#SBATCH -p sablab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=hyperct_train_hypernet_%j.out
#SBATCH --error=hyperct_train_hypernet_%j.err

set -euo pipefail
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate test

# Install dependencies (confirmed working setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops open_clip_torch timm deepspeed ninja
pip install flash-attn --no-build-isolation
pip install "transformers>=4.56.0" nibabel tqdm
pip install "numpy<2"
pip install --upgrade peft
pip install --upgrade pip wheel
pip install --force-reinstall --no-deps markupsafe==3.0.3

PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/HRadiology_VLM_AI4ML/HyperCT_UPDT
cd "$PROJECT_DIR"

python train_hypernet.py \
    --data_dir /midtier/sablab/scratch/isg4006/VLM_Project/data/ct_volumes \
    --labels_json /midtier/sablab/scratch/isg4006/VLM_Project/data/labels_train.json \
    --val_labels_json /midtier/sablab/scratch/isg4006/VLM_Project/data/labels_val.json \
    --output_dir ./checkpoints/hypernet \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --lora_rank 16 \
    --lora_scaling 1.0 \
    --num_slices 33 \
    --slice_height 512 \
    --slice_width 512 \
    --batch_size 8 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --epochs 20 \
    --num_workers 4 \
    --seed 42
