#!/bin/bash
#SBATCH --job-name=vlm_medical
#SBATCH -p sablab-gpu-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=vlm_medical_%j.out
#SBATCH --error=vlm_medical_%j.err

set -euo pipefail
module purge
module load anaconda3
conda init bash
conda activate test

# Use ONLY midtier paths
PROJECT_DIR=/midtier/sablab/scratch/isg4006/VLM_Project/AI4ML-initiative---Medical-VLM-Model
# VENV_PY=/midtier/sablab/scratch/isg4006/VLM_Project/venv/bin/python

cd "$PROJECT_DIR"
python -c "import nltk; print('nltk ok in job env')"


# 1. Pre-encoding CT volumes (run first, once per dataset):
#    python "$PROJECT_DIR/CT-CHAT/llava/serve/encode_script.py" --path /path/to/volume.nii.gz --slope 1.0 --intercept 0.0 --xy_spacing 0.7 --z_spacing 1.5
# 2. Training (run after pre-encoding):
python "$PROJECT_DIR/CT-CHAT/llava/train/train_mem.py"

