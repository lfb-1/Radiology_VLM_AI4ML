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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.pooling import ensure_length, pad_volume_slices, CubePooler
from models.encoder import DINOv3LoRAEncoder
from config import RADIOLOGICAL_TASKS
import os
import json
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CTMultiLabelDataset(Dataset):
    """
    Dataset for HyperNetwork training on labeled CT volumes.

    Each item returns:
        - slices: (num_slices, H, W)  all resampled slices for 3D training
        - labels: (num_tasks,) multi-label vector (-1=abstain, 0/1)
        - valid_mask: (num_tasks,) boolean mask of non-abstain labels
    """

    def __init__(self, data_dir: str, labels_json: str,
                 slice_size: tuple = (224, 224), num_slices: int = 33,
                 hu_min: float = -1000, hu_max: float = 1000,
                 preprocess_dir: str = None):
        with open(labels_json, "r") as f:
            all_records = json.load(f)
        raw_records = [r for r in all_records if "image" in r]
        log.info(f"Loaded {len(raw_records)} records with images "
                 f"(skipped {len(all_records) - len(raw_records)} without)")

        # Deduplicate: multiple VQA records per volume → keep first per image
        seen = {}
        for r in raw_records:
            key = r["image"]
            if key not in seen:
                seen[key] = r
        self.records = list(seen.values())
        log.info(f"Deduplicated to {len(self.records)} unique volumes "
                 f"(was {len(raw_records)} records)")

        self.data_dir = data_dir
        self.preprocess_dir = preprocess_dir
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
        # e.g. train_13158_c_2.nii.gz
        fname = os.path.basename(image_ref)
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

        # Try loading preprocessed .pt file first (much faster: no NIfTI
        # decompression or resampling). Fall back to raw NIfTI if unavailable.
        slices = None
        if self.preprocess_dir is not None:
            pt_name = os.path.basename(rec["image"]).replace(".nii.gz", ".pt")
            pt_path = os.path.join(self.preprocess_dir, pt_name)
            if os.path.exists(pt_path):
                slices = torch.load(pt_path, weights_only=True)

        if slices is None:
            nifti_path = self._resolve_nifti_path(self.data_dir, rec["image"])
            if idx < 3:
                log.info(f"[dataset] loading idx={idx}: {nifti_path}")
            slices = self._load_volume(nifti_path)  # (num_slices, H, W)

        # Return full volume slices so training loop can process multiple
        # RGB groups through DINOv3+LoRA + CubePooler, matching the
        # precompute pipeline (fixes Stage 1↔2 distribution mismatch).

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
            "slices": slices,            # (num_slices, H, W)
            "labels": labels,            # (num_tasks,)
            "valid_mask": valid_mask,     # (num_tasks,)
        }


def collate_fn(batch):
    return {
        "slices": torch.stack([b["slices"] for b in batch]),
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


def sample_task_for_batch(labels: torch.Tensor, valid_mask: torch.Tensor):
    """
    Pick ONE task for the entire batch — the task valid for the most samples.

    This yields exactly 1 LoRA generation + 1 batched forward pass per step,
    instead of up to num_tasks separate passes (2-8x faster per step).
    All tasks are still covered across batches over the epoch.

    Returns:
        chosen_task: int or None (if no valid task)
        batch_task_labels: (K,) float tensor of binary labels for valid samples
        sample_ok: (B,) bool mask of samples valid for the chosen task
    """
    valid_counts = valid_mask.sum(dim=0)  # (num_tasks,)
    if valid_counts.max() == 0:
        return None, None, None

    # Pick randomly among tasks with the most valid samples
    max_count = valid_counts.max().item()
    candidates = (valid_counts == max_count).nonzero(as_tuple=True)[0]
    chosen = candidates[random.randint(0, len(candidates) - 1)].item()

    sample_ok = valid_mask[:, chosen]
    batch_task_labels = labels[sample_ok, chosen]

    return chosen, batch_task_labels, sample_ok


def _build_rgb_groups(volume_slices: torch.Tensor) -> torch.Tensor:
    """
    Convert (num_slices, H, W) volume into (num_rgb, 3, H, W) RGB groups.
    Groups 3 consecutive slices as R, G, B channels.
    """
    num_slices = volume_slices.shape[0]
    num_rgb = num_slices // 3
    groups = []
    for g in range(num_rgb):
        groups.append(volume_slices[g * 3: g * 3 + 3])  # (3, H, W)
    return torch.stack(groups)  # (num_rgb, 3, H, W)


def train_one_epoch(
    encoder: DINOv3LoRAEncoder,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    pooler: CubePooler,
    max_batches: int = None,
):
    """
    Train one epoch with multi-slice 3D-aware training.

    For each batch:
        1. Pick one task for the batch
        2. Generate LoRA weights via HyperNetwork
        3. Process ALL RGB slice groups through DINOv3+LoRA (batched)
        4. CubePooler per sample → compressed 3D tokens
        5. Classify on pooled tokens → BCE loss

    This aligns training with the precompute pipeline, fixing the
    Stage 1↔2 distribution mismatch.
    """
    encoder.encoder.eval()          # DINOv3 backbone frozen
    encoder.hypernet.train()        # HyperNetwork trainable
    encoder.classifier.train()      # Classifier trainable
    pooler.train()                  # CubePooler trainable

    total_loss = 0.0
    num_batches = 0
    num_valid = 0
    num_skipped = 0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            log.info(
                f"  reached max_batches={max_batches}, stopping epoch early")
            break
        if batch_idx == 0:
            log.info("First batch loaded — training started.")
        if batch_idx % 50 == 0:
            log.info(f"  batch {batch_idx} / {len(dataloader)}")

        all_slices = batch["slices"].to(device)    # (B, num_slices, H, W)
        labels = batch["labels"].to(device)         # (B, num_tasks)
        vmask = batch["valid_mask"].to(device)      # (B, num_tasks)

        # Single task per batch for efficiency
        chosen_task, batch_task_labels, sample_ok = sample_task_for_batch(
            labels, vmask)
        if chosen_task is None or not sample_ok.any():
            num_skipped += 1
            continue

        all_slices = all_slices[sample_ok]  # (K, num_slices, H, W)
        batch_task_labels = batch_task_labels.to(device)
        K = all_slices.shape[0]

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            task_id = torch.tensor([chosen_task], device=device)

            # Image conditioning: extract content features from raw pixels
            # so LoRA weights adapt to this specific scan, not just the task.
            per_sample_rgb = []
            for i in range(K):
                per_sample_rgb.append(_build_rgb_groups(all_slices[i]))
            num_rgb = per_sample_rgb[0].shape[0]
            all_rgb = torch.cat(per_sample_rgb, dim=0)  # (K*num_rgb, 3, H, W)

            encoder.hypernet.set_image_conditioning(all_rgb)
            lora_w = encoder.hypernet.generate_full_model_lora(task_id)
            encoder.hypernet.clear_image_conditioning()

            all_tokens = encoder.forward_with_lora(all_rgb, lora_w)
            # all_tokens: (K * num_rgb, N_patches, D)

            # Split back per sample and pool with CubePooler
            pooled_list = []
            for i in range(K):
                start = i * num_rgb
                sample_tokens = all_tokens[start: start + num_rgb]
                # CubePooler expects list of (1, N, D)
                token_list = [sample_tokens[g:g+1] for g in range(num_rgb)]
                pooled = pooler(token_list)  # (1, final_tokens, D)
                pooled_list.append(pooled)

            # (K, final_tokens, D)
            pooled_batch = torch.cat(pooled_list, dim=0)
            logits = encoder.classify(pooled_batch)  # (K, num_tasks)
            pred_logits = logits[:, chosen_task]      # (K,)
            loss = criterion(pred_logits, batch_task_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.hypernet.parameters()) +
            list(encoder.classifier.parameters()) +
            list(pooler.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        num_valid += K

        # Collect for AUC
        all_preds.append(pred_logits.detach().cpu())
        all_targets.append(batch_task_labels.detach().cpu())

        if batch_idx % 10 == 0:
            log.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                     f"Loss {loss.item():.4f} | Valid {K}")

    avg_loss = total_loss / max(num_batches, 1)

    # Compute AUC
    train_auc = None
    if all_preds:
        all_p = torch.cat(all_preds).sigmoid().numpy()
        all_t = torch.cat(all_targets).numpy()
        if len(np.unique(all_t)) > 1:
            train_auc = roc_auc_score(all_t, all_p)
            log.info(f"Epoch {epoch} train AUC: {train_auc:.4f}")
        else:
            log.info(f"Epoch {epoch} train AUC: N/A (single class in targets)")

    log.info(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}, "
             f"valid samples: {num_valid}, skipped batches: {num_skipped}")
    return avg_loss, train_auc


@torch.no_grad()
def evaluate(
    encoder: DINOv3LoRAEncoder,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pooler: CubePooler = None,
):
    """Evaluate on validation set with same multi-slice pipeline as training."""
    encoder.eval()
    pooler.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        all_slices = batch["slices"].to(device)
        labels = batch["labels"].to(device)
        vmask = batch["valid_mask"].to(device)

        chosen_task, batch_task_labels, sample_ok = sample_task_for_batch(
            labels, vmask)
        if chosen_task is None or not sample_ok.any():
            continue

        all_slices = all_slices[sample_ok]
        batch_task_labels = batch_task_labels.to(device)
        K = all_slices.shape[0]

        task_id = torch.tensor([chosen_task], device=device)

        per_sample_rgb = [_build_rgb_groups(all_slices[i]) for i in range(K)]
        num_rgb = per_sample_rgb[0].shape[0]
        all_rgb = torch.cat(per_sample_rgb, dim=0)

        encoder.hypernet.set_image_conditioning(all_rgb)
        lora_w = encoder.hypernet.generate_full_model_lora(task_id)
        encoder.hypernet.clear_image_conditioning()

        all_tokens = encoder.forward_with_lora(all_rgb, lora_w)

        pooled_list = []
        for i in range(K):
            start = i * num_rgb
            sample_tokens = all_tokens[start: start + num_rgb]
            token_list = [sample_tokens[g:g+1] for g in range(num_rgb)]
            pooled = pooler(token_list)
            pooled_list.append(pooled)

        pooled_batch = torch.cat(pooled_list, dim=0)
        logits = encoder.classify(pooled_batch)
        pred_logits = logits[:, chosen_task]
        loss = criterion(pred_logits, batch_task_labels)
        total_loss += loss.item()
        num_batches += 1

        all_preds.append(pred_logits.cpu())
        all_targets.append(batch_task_labels.cpu())

    avg_loss = total_loss / max(num_batches, 1)

    # Compute AUC
    val_auc = None
    if all_preds:
        all_p = torch.cat(all_preds).sigmoid().numpy()
        all_t = torch.cat(all_targets).numpy()
        if len(np.unique(all_t)) > 1:
            val_auc = roc_auc_score(all_t, all_p)
            log.info(f"Validation AUC: {val_auc:.4f}")
        else:
            log.info("Validation AUC: N/A (single class in targets)")

    log.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss, val_auc


def save_checkpoint(encoder, epoch, output_dir, is_best=False, pooler=None):
    """Save HyperNetwork + Classifier + CubePooler checkpoint."""
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "hypernet": encoder.hypernet.state_dict(),
        "classifier": encoder.classifier.state_dict(),
        "pooler": pooler.state_dict() if pooler is not None else {},
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
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Directory with validation .nii.gz files (defaults to --data_dir)")
    parser.add_argument("--output_dir", type=str,
                        default="./checkpoints/hypernet")
    parser.add_argument("--encoder_name", type=str,
                        default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_scaling", type=float, default=1.0)
    parser.add_argument("--num_slices", type=int, default=33)
    parser.add_argument("--slice_height", type=int, default=224)
    parser.add_argument("--slice_width", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size. Lower than before because each sample "
                             "now processes all RGB groups (multi-slice 3D training).")
    parser.add_argument("--cube_pool_levels", type=int, default=2,
                        help="CubePooler 2x2x2 merging levels (must match precompute)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Stop if val loss doesn't improve for N epochs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                        help="Cap batches per epoch for faster iteration")
    parser.add_argument("--preprocess_dir", type=str, default=None,
                        help="Directory with preprocessed .pt volumes (from preprocess_volumes.py)")
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
        ckpt = torch.load(
            args.checkpoint, map_location=device, weights_only=True)
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
        preprocess_dir=args.preprocess_dir,
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
        val_data_dir = args.val_data_dir or args.data_dir
        val_ds = CTMultiLabelDataset(
            val_data_dir, args.val_labels_json, slice_size, args.num_slices,
            preprocess_dir=args.preprocess_dir,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=use_persistent,
        )
        log.info(f"Validation set: {len(val_ds)} volumes")

    # CubePooler — trained alongside hypernet + classifier so the
    # classifier sees CubePooler-aggregated features, matching precompute.
    pooler = CubePooler(dim=768, num_levels=args.cube_pool_levels).to(device)

    if args.checkpoint:
        if "pooler" in ckpt:
            pooler.load_state_dict(ckpt["pooler"])
            log.info("Loaded CubePooler from checkpoint")

    # Optimizer: hypernet + classifier + CubePooler
    trainable_params = [
        {"params": encoder.hypernet.parameters(), "lr": args.lr},
        {"params": encoder.classifier.parameters(), "lr": args.lr},
        {"params": pooler.parameters(), "lr": args.lr},
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
    epochs_without_improvement = 0
    log.info("Starting HyperNetwork training...")

    for epoch in range(1, args.epochs + 1):
        log.info(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_auc = train_one_epoch(
            encoder, optimizer, criterion, train_loader, device, epoch, scaler,
            pooler=pooler,
            max_batches=args.max_batches_per_epoch,
        )

        val_loss, val_auc = None, None
        if val_loader is not None:
            val_loss, val_auc = evaluate(encoder, criterion, val_loader, device,
                                         pooler=pooler)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        log.info(f"LR: {current_lr:.2e}")

        # Checkpointing
        is_best = val_loss is not None and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            log.info(f"New best validation loss: {best_val_loss:.4f}")
        elif val_loss is not None:
            epochs_without_improvement += 1
            log.info(f"No improvement for {epochs_without_improvement} epoch(s)")

        save_checkpoint(encoder, epoch, args.output_dir, is_best=is_best,
                        pooler=pooler)

        # Early stopping
        if val_loss is not None and epochs_without_improvement >= args.early_stop_patience:
            log.info(f"Early stopping triggered after {epoch} epochs "
                     f"(patience={args.early_stop_patience})")
            break

    # Save final
    final_path = os.path.join(args.output_dir, "final_checkpoint.pth")
    torch.save({
        "encoder": encoder.state_dict(),
        "hypernet": encoder.hypernet.state_dict(),
        "classifier": encoder.classifier.state_dict(),
        "pooler": pooler.state_dict(),
    }, final_path)
    log.info(f"Saved final checkpoint: {final_path}")
    log.info("HyperNetwork training complete.")


if __name__ == "__main__":
    main()
