"""
Offline Volume Preprocessing for HyperCT_UPDT

Converts .nii.gz CT volumes to lightweight .pt tensor files containing
resampled slices. Dramatically speeds up training by eliminating
per-sample NIfTI decompression and resampling.

Speedup: ~10x faster data loading during training.
Storage: ~6 MB per .pt file (33 slices × 224 × 224 × float32)
         vs 100-500 MB per .nii.gz (compressed 3D volume)

Usage:
    python preprocess_volumes.py \\
        --data_dir /path/to/nifti_files \\
        --labels_json /path/to/labels.json \\
        --output_dir ./preprocessed \\
        --num_slices 33 \\
        --slice_height 224 \\
        --slice_width 224
"""

from models.pooling import ensure_length, pad_volume_slices
import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def resolve_nifti_path(data_dir: str, image_ref: str) -> str:
    """Resolve JSON image ref to actual nested path on disk."""
    direct = os.path.join(data_dir, image_ref)
    if os.path.exists(direct):
        return direct
    split_dir = os.path.dirname(image_ref)
    fname = os.path.basename(image_ref)
    stem = fname.replace(".nii.gz", "")
    series_id = stem.rsplit("_", 1)[0]
    patient_id = series_id.rsplit("_", 1)[0]
    return os.path.join(data_dir, split_dir, patient_id, series_id, fname)


def load_and_resample(path: str, num_slices: int, slice_size: tuple,
                      hu_min: float = -1000, hu_max: float = 1000) -> torch.Tensor:
    """Load NIfTI, HU window, resample → (num_slices, H, W) float32 tensor."""
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)

    dims = vol.shape
    depth_axis = int(np.argmin(dims))
    if depth_axis != 0:
        vol = np.moveaxis(vol, depth_axis, 0)

    D, H, W = vol.shape
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    vol_t = F.interpolate(vol_t, size=(num_slices, H, W),
                          mode="trilinear", align_corners=False)
    vol_t = F.interpolate(vol_t.squeeze(0),
                          size=(slice_size[0], slice_size[1]),
                          mode="bilinear", align_corners=False)
    slices = vol_t.squeeze(0)
    slices = pad_volume_slices(slices, num_slices)
    return slices


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NIfTI volumes to .pt tensors")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with .nii.gz files")
    parser.add_argument("--labels_json", type=str, required=True,
                        help="JSON with training records (to discover volumes)")
    parser.add_argument("--output_dir", type=str, default="./preprocessed",
                        help="Output directory for .pt files")
    parser.add_argument("--num_slices", type=int, default=33)
    parser.add_argument("--slice_height", type=int, default=224)
    parser.add_argument("--slice_width", type=int, default=224)
    args = parser.parse_args()

    args.num_slices = ensure_length(args.num_slices, divisor=3)
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover unique volumes from labels JSON
    with open(args.labels_json, "r") as f:
        all_records = json.load(f)

    seen = set()
    volumes = []
    for r in all_records:
        if "image" not in r:
            continue
        key = r["image"]
        if key not in seen:
            seen.add(key)
            volumes.append(key)

    log.info(f"Found {len(volumes)} unique volumes to preprocess")

    slice_size = (args.slice_height, args.slice_width)
    skipped = 0
    failed = 0

    for image_ref in tqdm(volumes, desc="Preprocessing"):
        pt_name = os.path.basename(image_ref).replace(".nii.gz", ".pt")
        pt_path = os.path.join(args.output_dir, pt_name)

        if os.path.exists(pt_path):
            skipped += 1
            continue

        nifti_path = resolve_nifti_path(args.data_dir, image_ref)
        if not os.path.exists(nifti_path):
            log.warning(f"Not found: {nifti_path}")
            failed += 1
            continue

        try:
            slices = load_and_resample(nifti_path, args.num_slices, slice_size)
            torch.save(slices, pt_path)
        except Exception as e:
            log.warning(f"Failed {nifti_path}: {e}")
            failed += 1

    log.info(f"Done. Preprocessed: {len(volumes) - skipped - failed}, "
             f"Skipped (exists): {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
