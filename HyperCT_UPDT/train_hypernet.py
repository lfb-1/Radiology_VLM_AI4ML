"""
HyperNetwork Training Script for HyperCT_UPDT Pipeline

Trains the HyperNetwork + TaskClassifier on labeled CT data so that
task-specific LoRA weights improve downstream classification.

Architecture (following HyperCT reference github.com/lfb-1/HyperCT):
    - DINOv3 backbone: FROZEN (requires_grad=False)
    - HyperNetwork: TRAINABLE — generates LoRA A/B per task per layer
    - TaskClassifier: TRAINABLE — multi-label BCE supervision
    - LoRA injection: via forward hooks (HookBasedLoRAManager)

Training loop per epoch:
    For each CT volume batch:
        1. Sample one task per volume from multi-label ground truth
        2. Generate LoRA weights via HyperNetwork for that task
        3. Apply LoRA via hooks → forward DINOv3 → patch tokens
        4. Global pool → TaskClassifier → logits
        5. BCEWithLogitsLoss against multi-label targets
        6. Backward through: loss → classifier → features → LoRA → hypernet

Data format:
    JSON list of dicts:
        {"id": "scan_001", "image": "scan_001.nii.gz",
         "labels": {"opacity": 1, "nodule": 0, "consolidation": -1, ...}}
    -1 = unknown/abstain, 0 = negative, 1 = positive

Usage:
    python train_hypernet.py \\
        --data_dir /path/to/nifti_files \\
        --labels_json /path/to/labels.json \\
        --output_dir ./checkpoints/hypernet
"""

import os
import json
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RADIOLOGICAL_TASKS
from models.encoder import DINOv3LoRAEncoder
from models.pooling import ensure_length, pad_volume_slices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CTMultiLabelDataset(Dataset):
    """
    Dataset for HyperNetwork training on labeled CT volumes.

    Each item returns:
        - pixel_values: (3, H, W)  single RGB image (3 consecutive slices)
        - labels: (num_tasks,) multi-label vector (-1=abstain, 0/1)
        - valid_mask: (num_tasks,) boolean mask of non-abstain labels
    """

    def __init__(self, data_dir: str, labels_json: str,
                 slice_size: tuple = (512, 512), num_slices: int = 33,
                 hu_min: float = -1000, hu_max: float = 1000):
        with open(labels_json, "r") as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if "image" in r]
        log.info(f"Loaded {len(self.records)} records with images "
                 f"(skipped {len(all_records) - len(self.records)} without)")
        self.data_dir = data_dir
        self.slice_size = slice_size
        self.num_slices = ensure_length(num_slices, divisor=3)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.num_rgb = self.num_slices // 3

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _resolve_nifti_path(data_dir: str, image_ref: str) -> str:
        """Resolve JSON image ref to actual nested path on disk.

        JSON stores flat refs like:  train_fixed/train_13158_c_2.nii.gz
        Actual structure is:         train_fixed/train_13158/train_13158_c/train_13158_c_2.nii.gz

        Pattern: {split}/{patient_id}/{series_id}/{filename}
          patient_id = filename stem minus last two '_'-separated tokens  → train_13158
          series_id  = filename stem minus last '_'-separated token       → train_13158_c
        """
        direct = os.path.join(data_dir, image_ref)
        if os.path.exists(direct):
            return direct
        split_dir = os.path.dirname(image_ref)        # e.g. train_fixed
        fname = os.path.basename(image_ref)            # e.g. train_13158_c_2.nii.gz
        stem = fname.replace(".nii.gz", "")            # train_13158_c_2
        series_id = stem.rsplit("_", 1)[0]             # train_13158_c
        patient_id = series_id.rsplit("_", 1)[0]       # train_13158
        return os.path.join(data_dir, split_dir, patient_id, series_id, fname)

    def _load_volume(self, path: str) -> torch.Tensor:
        """Load NIfTI, HU window, resample → (num_slices, H, W)."""
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        vol = np.clip(vol, self.hu_min, self.hu_max)
        vol = (vol - self.hu_min) / (self.hu_max - self.hu_min)

        dims = vol.shape
        depth_axis = int(np.argmin(dims))
        if depth_axis != 0:
            vol = np.moveaxis(vol, depth_axis, 0)

        D, H, W = vol.shape
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(vol_t, size=(self.num_slices, H, W),
                              mode="trilinear", align_corners=False)
        vol_t = F.interpolate(vol_t.squeeze(0),
                              size=(self.slice_size[0], self.slice_size[1]),
                              mode="bilinear", align_corners=False)
        slices = vol_t.squeeze(0)
        slices = pad_volume_slices(slices, self.num_slices)
        return slices

    @staticmethod
    def _labels_from_conversations(conversations: list) -> torch.Tensor:
        """Derive binary task labels by keyword-matching GPT responses.

        Keywords per task mapped from RADIOLOGICAL_TASKS. A task is:
          1  if the GPT response mentions the condition as present
          0  if the response explicitly says it is absent / not observed
         -1  (abstain) if not mentioned at all
        """
        # Collect all GPT response text
        gpt_text = " ".join(
            turn["value"].lower()
            for turn in conversations
            if turn.get("from") == "gpt"
        )

        # Keyword map: task_name → (positive keywords, negative phrases)
        TASK_KEYWORDS = {
            "opacity":               (["opaci", "opacity", "opacification"], ["no opaci", "without opaci"]),
            "nodule":                (["nodule", "nodular"], ["no nodule", "without nodule"]),
            "consolidation":         (["consolidat"], ["no consolidat"]),
            "atelectasis":           (["atelectas", "atelectatic"], ["no atelectas"]),
            "pleural_effusion":      (["pleural effusion", "pleural fluid"], ["no pleural effusion", "pleural effusion was not"]),
            "cardiomegaly":          (["cardiomegaly", "enlarged heart", "cardiac enlargement"], ["no cardiomegaly", "heart size is normal"]),
            "emphysema":             (["emphysema", "emphysematous"], ["no emphysema"]),
            "fibrosis":              (["fibros", "fibrotic"], ["no fibros"]),
            "bronchiectasis":        (["bronchiectasis", "bronchiectatic"], ["no bronchiectasis"]),
            "lymphadenopathy":       (["lymphadenopathy", "lymph node enlargement", "mediastinal lymph"], ["no lymphadenopathy"]),
            "mass":                  (["mass ", "masses", "mass lesion"], ["no mass"]),
            "pneumothorax":          (["pneumothorax"], ["no pneumothorax", "pneumothorax was not"]),
            "pericardial_effusion":  (["pericardial effusion"], ["no pericardial effusion", "pericardial effusion was not", "pericardial effusion-thickening was not"]),
            "calcification":         (["calcif", "calcific"], ["no calcif"]),
            "medical_material":      (["catheter", "pacemaker", "stent", "prosthes", "implant", "device", "tube"], []),
            "mosaic_attenuation":    (["mosaic attenuation", "mosaic pattern"], ["no mosaic"]),
            "peribronchial_thickening": (["peribronchial thickening", "bronchial wall thickening"], ["no peribronchial"]),
            "hiatal_hernia":         (["hiatal hernia", "hiatus hernia"], ["no hiatal hernia"]),
        }

        labels = torch.full((len(RADIOLOGICAL_TASKS),), -1.0)
        for i, task in enumerate(RADIOLOGICAL_TASKS):
            pos_kws, neg_phrases = TASK_KEYWORDS.get(task, ([], []))
            neg_hit = any(neg in gpt_text for neg in neg_phrases)
            pos_hit = any(kw in gpt_text for kw in pos_kws)
            if neg_hit:
                labels[i] = 0.0
            elif pos_hit:
                labels[i] = 1.0
            # else remains -1 (abstain)
        return labels

    def __getitem__(self, idx):
        rec = self.records[idx]
        nifti_path = self._resolve_nifti_path(self.data_dir, rec["image"])
        if idx < 3:
            log.info(f"[dataset] loading idx={idx}: {nifti_path}")
        slices = self._load_volume(nifti_path)  # (num_slices, H, W)

        # Pick random RGB group (3 consecutive slices) for this sample
        group_idx = random.randint(0, self.num_rgb - 1)
        rgb = slices[group_idx * 3 : group_idx * 3 + 3]  # (3, H, W)

        # Build multi-label vector — prefer explicit labels, fall back to
        # keyword extraction from GPT conversation responses.
        labels = torch.full((len(RADIOLOGICAL_TASKS),), -1.0)
        if "labels" in rec:
            for i, task in enumerate(RADIOLOGICAL_TASKS):
                if task in rec["labels"]:
                    labels[i] = float(rec["labels"][task])
        elif "conversations" in rec:
            labels = self._labels_from_conversations(rec["conversations"])

        valid_mask = labels != -1

        return {
            "pixel_values": rgb,         # (3, H, W)
            "labels": labels,            # (num_tasks,)
            "valid_mask": valid_mask,     # (num_tasks,)
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def sample_task_per_sample(labels: torch.Tensor, valid_mask: torch.Tensor):
    """
    For each sample in the batch, randomly pick one valid task.
    Returns task indices and the corresponding binary label.

    Following HyperCT reference: _sample_tasks_from_multilabel_batch
    """
    B = labels.shape[0]
    task_indices = []
    task_labels = []
    sample_valid = []

    for i in range(B):
        valid = valid_mask[i].nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            task_indices.append(0)
            task_labels.append(0.0)
            sample_valid.append(False)
            continue
        chosen = valid[random.randint(0, len(valid) - 1)].item()
        task_indices.append(chosen)
        task_labels.append(labels[i, chosen].item())
        sample_valid.append(True)

    return (
        torch.tensor(task_indices, dtype=torch.long),
        torch.tensor(task_labels, dtype=torch.float32),
        torch.tensor(sample_valid, dtype=torch.bool),
    )


def train_one_epoch(
    encoder: DINOv3LoRAEncoder,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    max_batches: int = None,
):
    """
    Train one epoch with inline LoRA via forward_with_lora.

    Following HyperCT reference: train_epoch_with_hypernet
    - Base model stays frozen (.eval())
    - HyperNetwork + TaskClassifier are trainable (.train())
    """
    encoder.encoder.eval()          # DINOv3 backbone frozen
    encoder.hypernet.train()        # HyperNetwork trainable
    encoder.classifier.train()      # Classifier trainable

    total_loss = 0.0
    num_batches = 0
    num_valid = 0
    num_skipped = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            log.info(f"  reached max_batches={max_batches}, stopping epoch early")
            break
        if batch_idx == 0:
            log.info("First batch loaded — training started.")
        if batch_idx % 50 == 0:
            log.info(f"  batch {batch_idx} / {len(dataloader)}")
        pv = batch["pixel_values"].to(device)      # (B, 3, H, W)
        labels = batch["labels"].to(device)         # (B, num_tasks)
        vmask = batch["valid_mask"].to(device)      # (B, num_tasks)

        task_ids, task_labels, sample_ok = sample_task_per_sample(labels, vmask)
        if not sample_ok.any():
            num_skipped += 1
            continue

        # Filter to valid samples only
        pv = pv[sample_ok]
        task_ids = task_ids[sample_ok].to(device)
        task_labels = task_labels[sample_ok].to(device)

        optimizer.zero_grad()

        # Group same-task samples for batched forward — instead of B serial
        # calls at batch size 1, run at most 18 task groups with larger batches.
        # F.scaled_dot_product_attention (Flash Attn 2) handles the full batch.
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            pred_logits_list = [None] * pv.shape[0]
            for task in task_ids.unique():
                idx = (task_ids == task).nonzero(as_tuple=True)[0]
                lora_w = encoder.hypernet.generate_full_model_lora(task.unsqueeze(0))
                patch_tokens = encoder.forward_with_lora(pv[idx], lora_w)  # (K, N, D)
                logits = encoder.classify(patch_tokens)                      # (K, num_tasks)
                for j, orig_i in enumerate(idx):
                    pred_logits_list[orig_i.item()] = logits[j, task.item()]

            pred_logits = torch.stack(pred_logits_list)  # (valid_B,)
            loss = criterion(pred_logits, task_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.hypernet.parameters()) + list(encoder.classifier.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        num_valid += pv.shape[0]

        if batch_idx % 10 == 0:
            log.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                     f"Loss {loss.item():.4f} | Valid {pv.shape[0]}")

    avg_loss = total_loss / max(num_batches, 1)
    log.info(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}, "
             f"valid samples: {num_valid}, skipped batches: {num_skipped}")
    return avg_loss


@torch.no_grad()
def evaluate(
    encoder: DINOv3LoRAEncoder,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    """Evaluate on validation set."""
    encoder.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        pv = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        vmask = batch["valid_mask"].to(device)

        task_ids, task_labels, sample_ok = sample_task_per_sample(labels, vmask)
        if not sample_ok.any():
            continue

        pv = pv[sample_ok]
        task_ids = task_ids[sample_ok].to(device)
        task_labels = task_labels[sample_ok].to(device)

        pred_logits_list = [None] * pv.shape[0]
        for task in task_ids.unique():
            idx = (task_ids == task).nonzero(as_tuple=True)[0]
            lora_w = encoder.hypernet.generate_full_model_lora(task.unsqueeze(0))
            patch_tokens = encoder.forward_with_lora(pv[idx], lora_w)
            logits = encoder.classify(patch_tokens)
            for j, orig_i in enumerate(idx):
                pred_logits_list[orig_i.item()] = logits[j, task.item()]

        pred_logits = torch.stack(pred_logits_list)
        loss = criterion(pred_logits, task_labels)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    log.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(encoder, epoch, output_dir, is_best=False):
    """Save HyperNetwork + Classifier checkpoint."""
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "hypernet": encoder.hypernet.state_dict(),
        "classifier": encoder.classifier.state_dict(),
    }
    path = os.path.join(epoch_dir, "checkpoint.pth")
    torch.save(state, path)
    log.info(f"Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(output_dir, "best_checkpoint.pth")
        torch.save(state, best_path)
        log.info(f"Saved best checkpoint: {best_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train HyperCT HyperNetwork")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with .nii.gz files")
    parser.add_argument("--labels_json", type=str, required=True,
                        help="JSON with multi-label annotations")
    parser.add_argument("--val_labels_json", type=str, default=None,
                        help="Optional validation labels JSON")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hypernet")
    parser.add_argument("--encoder_name", type=str,
                        default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_scaling", type=float, default=1.0)
    parser.add_argument("--num_slices", type=int, default=33)
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                        help="Cap batches per epoch for faster iteration")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fastest kernels for fixed 224×224 input
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    log.info(f"Initializing DINOv3 encoder: {args.encoder_name}")
    encoder = DINOv3LoRAEncoder(
        encoder_name=args.encoder_name,
        num_tasks=len(RADIOLOGICAL_TASKS),
        lora_rank=args.lora_rank,
        lora_scaling=args.lora_scaling,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        log.info(f"Resumed from {args.checkpoint}")

    # Freeze backbone explicitly
    encoder.encoder.requires_grad_(False)

    # Log param counts
    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    log.info(f"Total params: {total:,} | Trainable: {trainable:,} "
             f"({100*trainable/total:.1f}%)")

    # Dataset
    slice_size = (args.slice_height, args.slice_width)
    train_ds = CTMultiLabelDataset(
        args.data_dir, args.labels_json, slice_size, args.num_slices,
    )
    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=use_persistent,
    )
    log.info(f"Training set: {len(train_ds)} volumes")

    val_loader = None
    if args.val_labels_json:
        val_ds = CTMultiLabelDataset(
            args.data_dir, args.val_labels_json, slice_size, args.num_slices,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=use_persistent,
        )
        log.info(f"Validation set: {len(val_ds)} volumes")

    # Optimizer: only hypernet + classifier params
    trainable_params = [
        {"params": encoder.hypernet.parameters(), "lr": args.lr},
        {"params": encoder.classifier.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # Training loop
    best_val_loss = float("inf")
    log.info("Starting HyperNetwork training...")

    for epoch in range(1, args.epochs + 1):
        log.info(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            encoder, optimizer, criterion, train_loader, device, epoch, scaler,
            max_batches=args.max_batches_per_epoch,
        )

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(encoder, criterion, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        log.info(f"LR: {current_lr:.2e}")

        # Checkpointing
        is_best = val_loss is not None and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            log.info(f"New best validation loss: {best_val_loss:.4f}")

        save_checkpoint(encoder, epoch, args.output_dir, is_best=is_best)

    # Save final
    final_path = os.path.join(args.output_dir, "final_checkpoint.pth")
    torch.save({
        "encoder": encoder.state_dict(),
        "hypernet": encoder.hypernet.state_dict(),
        "classifier": encoder.classifier.state_dict(),
    }, final_path)
    log.info(f"Saved final checkpoint: {final_path}")
    log.info("HyperNetwork training complete.")


if __name__ == "__main__":
    main()
