# HyperCT_UPDT — Full Run Guide

## Prerequisites

```bash
conda activate test
cd /midtier/sablab/scratch/isg4006/VLM_Project/Radiology_VLM_AI4ML/HRadiology_VLM_AI4ML/HyperCT_UPDT
```

---

## Stage 0: Preprocess Volumes (run ONCE, offline)

Converts raw `.nii.gz` → fast `.pt` tensors. **Only needs to run once per dataset.** No GPU required.

```bash
# Train set
python preprocess_volumes.py \
    --data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/train_fixed \
    --labels_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/train_vqa.json \
    --output_dir ./preprocessed_train \
    --num_slices 33 \
    --slice_height 224 \
    --slice_width 224

# Validation set
python preprocess_volumes.py \
    --data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/valid_fixed \
    --labels_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/valid_vqa.json \
    --output_dir ./preprocessed_valid \
    --num_slices 33 \
    --slice_height 224 \
    --slice_width 224
```

**Time:** ~1-2 hours (CPU only)
**Output:** `./preprocessed_train/*.pt` and `./preprocessed_valid/*.pt`

---

## Stage 1: Train HyperNetwork

Trains DINOv3 + LoRAHypernet + CubePooler + TaskClassifier on labeled CT data.

```bash
sbatch scripts/train_hypernet.sh
```

**What it runs:**

```bash
python train_hypernet.py \
    --data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/train_fixed \
    --labels_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/train_vqa.json \
    --val_labels_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/valid_vqa.json \
    --val_data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/valid_fixed \
    --preprocess_dir ./preprocessed_train \
    --output_dir ./checkpoints/hypernet \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --lora_rank 16 --lora_scaling 1.0 \
    --num_slices 33 --slice_height 224 --slice_width 224 \
    --batch_size 2 --cube_pool_levels 2 \
    --lr 1e-5 --weight_decay 1e-2 \
    --epochs 20 --num_workers 4 \
    --max_batches_per_epoch 5000 --seed 42
```

**Resources:** 1 GPU, 128GB RAM, ~48 hours
**Output:** `./checkpoints/hypernet/best_checkpoint.pth`

---

## Stage 2: Precompute Vision Tokens

Runs all volumes through the trained HyperNet for every task, saves `.npz` tokens.

```bash
sbatch scripts/precompute.sh
```

**What it runs:**

```bash
python precompute_tokens.py \
    --data_dir /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/train_fixed \
    --output_dir ./precomputed_tokens \
    --checkpoint ./checkpoints/hypernet/best_checkpoint.pth \
    --num_slices 33 --slice_height 224 --slice_width 224 \
    --cube_pool_levels 2 \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --lora_rank 16 --lora_scaling 1.0
```

**Resources:** 1 GPU, 64GB RAM, ~72 hours
**Output:** `./precomputed_tokens/*.npz` (one per volume)

---

## Stage 3: Train VLM

Trains Q-Former + LoRA Llama 3.1 on VQA conversations using precomputed tokens.

```bash
sbatch scripts/train_vlm.sh
```

**What it runs:**

```bash
torchrun --nproc_per_node=4 train_vlm.py \
    --tokens_dir ./precomputed_tokens \
    --data_json /midtier/sablab/scratch/data/CT-RATEV2/data_volumes/dataset/vqa/train_vqa.json \
    --output_dir ./checkpoints/hyperct_vlm \
    --llm_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_hidden_size 4096 --vision_dim 768 \
    --num_queries 64 --qformer_layers 6 --qformer_heads 12 \
    --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --lr 2e-5 --epochs 3 \
    --batch_size 4 --grad_accum 2 \
    --max_length 2048 --num_task_tokens 3 \
    --bf16 --attn_implementation flash_attention_2
```

**Resources:** 4 GPUs, 128GB RAM, ~48 hours
**Output:** `./checkpoints/hyperct_vlm/` (Q-Former weights + LoRA adapter)

---

## Quick Reference

| Stage | Command | GPU | Time | Input | Output |
|-------|---------|-----|------|-------|--------|
| 0 | `python preprocess_volumes.py` | None | ~1-2h | `.nii.gz` | `.pt` files |
| 1 | `sbatch scripts/train_hypernet.sh` | 1 | ~48h | `.pt` + labels JSON | `best_checkpoint.pth` |
| 2 | `sbatch scripts/precompute.sh` | 1 | ~72h | `.nii.gz` + checkpoint | `.npz` tokens |
| 3 | `sbatch scripts/train_vlm.sh` | 4 | ~48h | `.npz` + VQA JSON | Q-Former + LoRA |

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch logs in real time
tail -f hyperct_train_hypernet_<JOB_ID>.out

# Check GPU usage
srun --jobid=<JOB_ID> nvidia-smi
```

---

## Resuming from Checkpoint

If Stage 1 gets interrupted:

```bash
python train_hypernet.py \
    ... (same args) ... \
    --checkpoint ./checkpoints/hypernet/epoch_<N>/checkpoint.pth
```

---

## Directory Structure After Training

```
HyperCT_UPDT/
├── preprocessed_train/          # Stage 0 output (.pt files)
├── preprocessed_valid/          # Stage 0 output (.pt files)
├── checkpoints/
│   └── hypernet/
│       ├── epoch_1/checkpoint.pth
│       ├── ...
│       ├── best_checkpoint.pth  # Stage 1 output
│       └── final_checkpoint.pth
├── precomputed_tokens/          # Stage 2 output (.npz files)
└── checkpoints/
    └── hyperct_vlm/             # Stage 3 output
        ├── qformer_final.pt
        └── llm_lora/
```
