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
        --output_dir ./checkpoint_2
"""

from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
import argparse
import random
import json
import os
from config import RADIOLOGICAL_TASKS
from models.encoder import DINOv3LoRAEncoder
from models.pooling import ensure_length, pad_volume_slices, CubePooler
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


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
                 preprocess_dir: str = None, augment: bool = False):
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
        deduped = list(seen.values())
        log.info(f"Deduplicated to {len(deduped)} unique volumes "
                 f"(was {len(raw_records)} records)")

        # Filter out records with no valid (0/1) labels — these would be
        # skipped during training anyway, wasting I/O and compute.
        self.records = []
        num_no_labels = 0
        for r in deduped:
            labels = torch.full((len(RADIOLOGICAL_TASKS),), -1.0)
            if "labels" in r and isinstance(r["labels"], dict) and r["labels"]:
                for i, task in enumerate(RADIOLOGICAL_TASKS):
                    if task in r["labels"]:
                        labels[i] = float(r["labels"][task])
            # Fall through to conversations if labels dict was empty or missing
            if (labels == -1).all() and "conversations" in r:
                labels = self._labels_from_conversations(r["conversations"])
            if (labels != -1).any():
                self.records.append(r)
            else:
                num_no_labels += 1
        if num_no_labels > 0:
            log.info(f"Filtered out {num_no_labels} volumes with no valid labels "
                     f"(all abstain) — {len(self.records)} usable volumes remain")

        self.data_dir = data_dir
        self.preprocess_dir = preprocess_dir
        self.slice_size = slice_size
        self.num_slices = ensure_length(num_slices, divisor=3)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.num_rgb = self.num_slices // 3
        self.augment = augment

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
        # Collect all GPT/assistant response text
        # Support both "gpt" (legacy) and "assistant" (OpenAI/Llama chat format)
        gpt_text = " ".join(
            turn["value"].lower()
            for turn in conversations
            if turn.get("from") in ("gpt", "assistant")
        )

        # Keyword map: task_name → (positive keywords, negative phrases)
        # IMPORTANT: negative phrases are checked FIRST. If a negative phrase
        # matches, the task is labelled 0 regardless of positive keywords.
        # Positive keywords use word-boundary-safe strings (no trailing space
        # hacks) to avoid false positives like "mass" missing end-of-sentence.
        TASK_KEYWORDS = {
            "opacity":               (["opaci", "opacification"], ["no opaci", "without opaci"]),
            "nodule":                (["nodule", "nodular"], ["no nodule", "without nodule", "no evidence of nodule"]),
            "consolidation":         (["consolidat"], ["no consolidat", "without consolidat"]),
            "atelectasis":           (["atelectas", "atelectatic"], ["no atelectas", "without atelectas"]),
            "pleural_effusion":      (["pleural effusion", "pleural fluid"], ["no pleural effusion", "pleural effusion was not", "no pleural fluid"]),
            "cardiomegaly":          (["cardiomegaly", "enlarged heart", "cardiac enlargement"], ["no cardiomegaly", "heart size is normal", "normal cardiac size"]),
            "emphysema":             (["emphysema", "emphysematous"], ["no emphysema", "without emphysema"]),
            "fibrosis":              (["fibros", "fibrotic"], ["no fibros", "without fibros"]),
            "bronchiectasis":        (["bronchiectasis", "bronchiectatic"], ["no bronchiectasis", "without bronchiectasis"]),
            "lymphadenopathy":       (["lymphadenopathy", "lymph node enlargement", "mediastinal lymph"], ["no lymphadenopathy", "no enlarged lymph"]),
            # "mass" uses multi-word phrases only to avoid matching "no mass" substring
            "mass":                  (["mass lesion", "soft tissue mass", "pulmonary mass", "lung mass", " masses"], ["no mass", "no evidence of mass"]),
            "pneumothorax":          (["pneumothorax"], ["no pneumothorax", "pneumothorax was not", "without pneumothorax"]),
            "pericardial_effusion":  (["pericardial effusion"], ["no pericardial effusion", "pericardial effusion was not", "pericardial effusion-thickening was not"]),
            "calcification":         (["calcif", "calcific"], ["no calcif", "without calcif"]),
            "medical_material":      (["catheter", "pacemaker", "stent", "prosthes", "implant", "medical device", "chest tube"], []),
            "mosaic_attenuation":    (["mosaic attenuation", "mosaic pattern"], ["no mosaic", "without mosaic"]),
            "peribronchial_thickening": (["peribronchial thickening", "bronchial wall thickening"], ["no peribronchial", "without peribronchial"]),
            "hiatal_hernia":         (["hiatal hernia", "hiatus hernia"], ["no hiatal hernia", "without hiatal hernia"]),
        }

        labels = torch.full((len(RADIOLOGICAL_TASKS),), -1.0)
        for i, task in enumerate(RADIOLOGICAL_TASKS):
            pos_kws, neg_phrases = TASK_KEYWORDS.get(task, ([], []))
            # Negation check must come first — a sentence like "no mass lesion"
            # contains both "no mass" (neg) and "mass lesion" (pos). Checking
            # neg first ensures the negative label wins.
            neg_hit = any(neg in gpt_text for neg in neg_phrases)
            if neg_hit:
                labels[i] = 0.0
            elif any(kw in gpt_text for kw in pos_kws):
                labels[i] = 1.0
            # else remains -1 (abstain)
        return labels

    @staticmethod
    def _augment_slices(slices: torch.Tensor) -> torch.Tensor:
        """
        Light CT-appropriate augmentations on (num_slices, H, W) tensor.
        Values are in [0, 1] after HU windowing. Applied only during training.

        - Horizontal flip: anatomically valid for CT (left-right symmetric findings)
        - Intensity jitter: simulates scanner variability (±5% scale, ±3% shift)
        - Gaussian noise: simulates low-dose CT noise (std=0.02)

        No rotation/elastic deformation here — those require resampling the
        already-loaded volume and would break slice ordering for CubePooler.
        """
        # Horizontal flip (p=0.5)
        if random.random() < 0.5:
            slices = torch.flip(slices, dims=[-1])

        # Intensity jitter (p=0.5) — scale then shift, then clamp to [0,1]
        if random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)
            shift = random.uniform(-0.03, 0.03)
            slices = (slices * scale + shift).clamp(0.0, 1.0)

        # Gaussian noise (p=0.3)
        if random.random() < 0.3:
            noise = torch.randn_like(slices) * 0.02
            slices = (slices + noise).clamp(0.0, 1.0)

        return slices

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

        if self.augment:
            slices = self._augment_slices(slices)

        # Return full volume slices so training loop can process multiple
        # RGB groups through DINOv3+LoRA + CubePooler, matching the
        # precompute pipeline (fixes Stage 1↔2 distribution mismatch).

        # Build multi-label vector — prefer explicit labels, fall back to
        # keyword extraction from GPT conversation responses.
        labels = torch.full((len(RADIOLOGICAL_TASKS),), -1.0)
        if "labels" in rec and isinstance(rec["labels"], dict) and rec["labels"]:
            for i, task in enumerate(RADIOLOGICAL_TASKS):
                if task in rec["labels"]:
                    labels[i] = float(rec["labels"][task])
        # Fall through to conversations if labels dict was empty or missing
        if (labels == -1).all() and "conversations" in rec:
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
    Pick ONE task for the entire batch using inverse-frequency weighted sampling.

    Tasks with fewer valid samples are sampled MORE often, ensuring rare tasks
    (e.g. pneumothorax n=20, cardiomegaly n=28) get gradient signal instead of
    being crowded out by common tasks. This is the primary fix for AUC ~0.5 on
    rare tasks.

    Returns:
        chosen_task: int or None (if no valid task)
        batch_task_labels: (K,) float tensor of binary labels for valid samples
        sample_ok: (B,) bool mask of samples valid for the chosen task
    """
    valid_counts = valid_mask.sum(dim=0).float()  # (num_tasks,)
    if valid_counts.max() == 0:
        return None, None, None

    # Inverse-frequency weights: rare tasks get higher probability.
    # Zero out tasks with no valid samples so they are never chosen.
    weights = 1.0 / (valid_counts + 1e-6)
    weights = weights * (valid_counts > 0).float()
    chosen = torch.multinomial(weights, 1).item()

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
    pos_weight: torch.Tensor = None,
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
    # Per-task prediction tracking — mixing tasks into a single AUC is
    # methodologically wrong; compute per-task AUC then macro-average.
    task_preds: dict = defaultdict(list)
    task_targets: dict = defaultdict(list)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            log.info(
                f"  reached max_batches={max_batches}, stopping epoch early")
            break

        if batch_idx % 50 == 0:
            log.info(f"  batch {batch_idx} / {len(dataloader)}")

        all_slices = batch["slices"].to(device)     # (B, num_slices, H, W)
        labels = batch["labels"].to(device)         # (B, num_tasks)
        vmask = batch["valid_mask"].to(device)      # (B, num_tasks)

        if batch_idx == 0:
            log.info("First batch loaded — training started.")
            # Diagnostic: log label stats for first batch to catch data issues early
            valid_per_sample = vmask.sum(dim=1)  # (B,) valid labels per sample
            # (num_tasks,) valid samples per task
            valid_per_task = vmask.sum(dim=0)
            log.info(f"  First batch diagnostics: valid_per_sample={valid_per_sample.tolist()}, "
                     f"total_valid_labels={vmask.sum().item()}, "
                     f"tasks_with_valid={int((valid_per_task > 0).sum().item())}/18")

        # Single task per batch for efficiency
        chosen_task, batch_task_labels, sample_ok = sample_task_for_batch(
            labels, vmask)
        if chosen_task is None or not sample_ok.any():
            num_skipped += 1
            if num_skipped <= 3:
                log.warning(f"  Skipping batch {batch_idx}: chosen_task={chosen_task}, "
                            f"vmask_sum={vmask.sum().item()}, labels_range="
                            f"[{labels.min().item():.1f}, {labels.max().item():.1f}]")
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
            # Use per-task pos_weight (scalar) instead of full pos_weight vector
            # to avoid shape mismatch: pred_logits is (K,) not (num_tasks,)
            if pos_weight is not None:
                pw = pos_weight[chosen_task].expand_as(batch_task_labels)
                loss = F.binary_cross_entropy_with_logits(
                    pred_logits, batch_task_labels, pos_weight=pw)
            else:
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

        # Collect per-task for AUC
        task_preds[chosen_task].append(pred_logits.detach().cpu())
        task_targets[chosen_task].append(batch_task_labels.detach().cpu())

        if batch_idx % 10 == 0:
            log.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                     f"Loss {loss.item():.4f} | Valid {K}")

    avg_loss = total_loss / max(num_batches, 1)

    # Compute per-task AUC then macro-average (correct for paper reporting)
    per_task_auc = {}
    for task_idx in sorted(task_preds.keys()):
        p = torch.cat(task_preds[task_idx]).sigmoid().numpy()
        t = torch.cat(task_targets[task_idx]).numpy()
        if len(np.unique(t)) > 1:
            task_auc = roc_auc_score(t, p)
            per_task_auc[task_idx] = task_auc
            log.info(f"  Epoch {epoch} | {RADIOLOGICAL_TASKS[task_idx]}: "
                     f"AUC={task_auc:.4f} (n={len(t)})")
    train_auc = float(np.mean(list(per_task_auc.values()))
                      ) if per_task_auc else None
    if train_auc is not None:
        log.info(f"Epoch {epoch} train macro-AUC: {train_auc:.4f} "
                 f"({len(per_task_auc)}/{len(RADIOLOGICAL_TASKS)} tasks)")
    else:
        log.info(f"Epoch {epoch} train AUC: N/A")

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
    pos_weight: torch.Tensor = None,
):
    """
    Evaluate on validation set with same multi-slice pipeline as training.

    Unlike training, evaluation iterates over ALL tasks that have valid labels
    in each batch — not just the most-common one. This ensures rare tasks
    (pneumothorax, cardiomegaly) appear in the AUC computation instead of
    being silently skipped, giving an accurate macro-AUC.
    """
    encoder.eval()
    pooler.eval()
    total_loss = 0.0
    num_batches = 0
    task_preds: dict = defaultdict(list)
    task_targets: dict = defaultdict(list)

    for batch in dataloader:
        all_slices = batch["slices"].to(device)
        labels = batch["labels"].to(device)
        vmask = batch["valid_mask"].to(device)

        # Evaluate every task that has at least one valid sample in this batch.
        # This is affordable at eval time (no backward pass).
        valid_counts = vmask.sum(dim=0)  # (num_tasks,)
        tasks_in_batch = (valid_counts > 0).nonzero(as_tuple=True)[0].tolist()
        if not tasks_in_batch:
            continue

        for chosen_task in tasks_in_batch:
            sample_ok = vmask[:, chosen_task]
            if not sample_ok.any():
                continue

            slices_k = all_slices[sample_ok]
            batch_task_labels = labels[sample_ok, chosen_task]
            K = slices_k.shape[0]

            task_id = torch.tensor([chosen_task], device=device)

            per_sample_rgb = [_build_rgb_groups(slices_k[i]) for i in range(K)]
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
            # Use per-task pos_weight to avoid shape mismatch
            if pos_weight is not None:
                pw = pos_weight[chosen_task].expand_as(batch_task_labels)
                loss = F.binary_cross_entropy_with_logits(
                    pred_logits, batch_task_labels, pos_weight=pw)
            else:
                loss = criterion(pred_logits, batch_task_labels)
            total_loss += loss.item()
            num_batches += 1

            task_preds[chosen_task].append(pred_logits.cpu())
            task_targets[chosen_task].append(batch_task_labels.cpu())

    avg_loss = total_loss / max(num_batches, 1)

    # Compute per-task AUC then macro-average
    per_task_auc = {}
    for task_idx in sorted(task_preds.keys()):
        p = torch.cat(task_preds[task_idx]).sigmoid().numpy()
        t = torch.cat(task_targets[task_idx]).numpy()
        if len(np.unique(t)) > 1:
            task_auc = roc_auc_score(t, p)
            per_task_auc[task_idx] = task_auc
            log.info(f"  Val | {RADIOLOGICAL_TASKS[task_idx]}: "
                     f"AUC={task_auc:.4f} (n={len(t)})")
    val_auc = float(np.mean(list(per_task_auc.values()))
                    ) if per_task_auc else None
    if val_auc is not None:
        log.info(f"Validation macro-AUC: {val_auc:.4f} "
                 f"({len(per_task_auc)}/{len(RADIOLOGICAL_TASKS)} tasks)")
    else:
        log.info("Validation AUC: N/A")

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
                        default="./checkpoint_2")
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
        preprocess_dir=args.preprocess_dir, augment=True,
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
    # Warmup for the first ~20% of epochs, then cosine decay to 1% of peak LR.
    # Without warmup the model jumps into a poor local minimum in epoch 1 and
    # val loss diverges on subsequent epochs.
    warmup_epochs = max(1, args.epochs // 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    # --- Compute class frequencies for pos_weight ---
    # This will help balance rare and common tasks, boosting macro-AUC.
    label_counts = torch.zeros(len(RADIOLOGICAL_TASKS), 2)  # [task, 0/1]
    for rec in train_ds.records:
        labels = torch.full((len(RADIOLOGICAL_TASKS),), -1.0)
        if "labels" in rec and isinstance(rec["labels"], dict) and rec["labels"]:
            for i, task in enumerate(RADIOLOGICAL_TASKS):
                if task in rec["labels"]:
                    labels[i] = float(rec["labels"][task])
        if (labels == -1).all() and "conversations" in rec:
            labels = train_ds._labels_from_conversations(rec["conversations"])
        for i, v in enumerate(labels):
            if v == 0:
                label_counts[i, 0] += 1
            elif v == 1:
                label_counts[i, 1] += 1

    # Diagnostic: log total valid labels found
    total_valid = label_counts.sum().item()
    log.info(f"Label statistics: {int(total_valid)} total valid labels "
             f"across {len(train_ds.records)} records")
    if total_valid == 0:
        log.warning("NO VALID LABELS FOUND — all tasks will be skipped! "
                    "Check your JSON format: need 'labels' dict with keys from "
                    "RADIOLOGICAL_TASKS or 'conversations' with from='gpt'/'assistant'.")
    pos_weight = torch.ones(len(RADIOLOGICAL_TASKS))
    for i in range(len(RADIOLOGICAL_TASKS)):
        n_pos = label_counts[i, 1]
        n_neg = label_counts[i, 0]
        if n_pos > 0:
            # Cap at 10 to prevent extreme weights on very rare classes
            # (e.g. pneumothorax 20 pos / 5000 neg → raw weight 250 destabilises training).
            # A cap of 10 still strongly upweights rare positives without causing
            # gradient explosion or pushing logits to ±inf in early epochs.
            pos_weight[i] = min(n_neg / n_pos, 10.0)
        else:
            pos_weight[i] = 1.0  # fallback if no positives
    for i, task in enumerate(RADIOLOGICAL_TASKS):
        log.info(f"  pos_weight[{task}] = {pos_weight[i]:.3f} "
                 f"(neg={int(label_counts[i,0])}, pos={int(label_counts[i,1])})")
    pos_weight = pos_weight.to(device)
    # no pos_weight here; passed per-task in train/eval
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # Training loop
    # Track best by val_auc (primary paper metric) with val_loss as tiebreaker.
    best_val_auc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    log.info("Starting HyperNetwork training...")

    for epoch in range(1, args.epochs + 1):
        log.info(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_auc = train_one_epoch(
            encoder, optimizer, criterion, train_loader, device, epoch, scaler,
            pooler=pooler,
            max_batches=args.max_batches_per_epoch,
            pos_weight=pos_weight,
        )

        val_loss, val_auc = None, None
        if val_loader is not None:
            val_loss, val_auc = evaluate(encoder, criterion, val_loader, device,
                                         pooler=pooler, pos_weight=pos_weight)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        log.info(f"LR: {current_lr:.2e}")

        # Checkpointing — best model by val_auc (primary paper metric).
        # Fall back to val_loss improvement if val_auc is unavailable.
        if val_auc is not None:
            is_best = val_auc > best_val_auc or (
                val_auc == best_val_auc and val_loss is not None and val_loss < best_val_loss
            )
        else:
            is_best = val_loss is not None and val_loss < best_val_loss

        if is_best:
            if val_auc is not None:
                best_val_auc = val_auc
            if val_loss is not None:
                best_val_loss = val_loss
            epochs_without_improvement = 0
            log.info(f"New best — val macro-AUC: {best_val_auc:.4f}, "
                     f"val loss: {best_val_loss:.4f}")
        elif val_loss is not None or val_auc is not None:
            epochs_without_improvement += 1
            log.info(
                f"No improvement for {epochs_without_improvement} epoch(s)")

        save_checkpoint(encoder, epoch, args.output_dir, is_best=is_best,
                        pooler=pooler)

        # Early stopping
        if val_loss is not None and epochs_without_improvement >= args.early_stop_patience:
            log.info(f"Early stopping triggered after {epoch} epochs "
                     f"(patience={args.early_stop_patience})")
            break

    # Save final
    final_path = os.path.join(args.output_dir, "final_checkpoint.pth")
    final_state = {
        "encoder": encoder.state_dict(),
        "hypernet": encoder.hypernet.state_dict(),
        "classifier": encoder.classifier.state_dict(),
        "pooler": pooler.state_dict(),
    }
    torch.save(final_state, final_path)
    log.info(f"Saved final checkpoint: {final_path}")
    log.info("HyperNetwork training complete.")


if __name__ == "__main__":
    main()
