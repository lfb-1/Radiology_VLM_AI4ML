#!/bin/bash
#SBATCH --job-name=vlm_medical
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=vlm_medical_%j.out
#SBATCH --error=vlm_medical_%j.err
#SBATCH --cpus-per-task=8

# Load modules or activate conda environment
module load cuda/11.7 || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# Install requirements if needed (uncomment if first run)
# pip install -r requirements.txt

# Run the main pipeline
python main.py
