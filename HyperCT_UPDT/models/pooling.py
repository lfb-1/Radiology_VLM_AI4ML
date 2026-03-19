"""Cube Merge Pooling for 3D CT Token Compression

Qwen-style 2×2×2 cube merging: groups tokens into spatial-temporal cubes
and merges each cube to 1 token via learned linear projection.

    Raw: num_rgb_images × N_patches_per_image × D
    After L levels of 2×2×2 merging: reduced by 8^L total

No separate spatial_output or temporal_output parameters needed —
compression is controlled solely by num_levels.

Includes ensure_length utility to pad total slices to be divisible by 3
(since 3 consecutive slices form one RGB image).
"""

import torch
import torch.nn as nn


def ensure_length(num_slices: int, divisor: int = 3) -> int:
    """Ensure num_slices is divisible by divisor. Rounds up if needed."""
    if num_slices % divisor != 0:
        num_slices = ((num_slices // divisor) + 1) * divisor
    return num_slices


def pad_volume_slices(slices: torch.Tensor, target_slices: int) -> torch.Tensor:
    """
    Pad or trim slice tensor to target number of slices.

    Args:
        slices: (num_slices, H, W) or (num_slices, ...)
        target_slices: desired number of slices (divisible by 3)
    Returns:
        padded: (target_slices, ...) tensor
    """
    current = slices.shape[0]
    if current == target_slices:
        return slices
    elif current > target_slices:
        return slices[:target_slices]
    else:
        pad_count = target_slices - current
        padding = slices[-1:].expand(pad_count, *slices.shape[1:])
        return torch.cat([slices, padding], dim=0)


class CubePooler(nn.Module):
    """
    Qwen-style 2×2×2 cube merging for spatial-temporal token compression.

    Groups tokens into 2(height) × 2(width) × 2(depth) cubes and merges
    each cube to a single token via concat + linear projection.

    Applied iteratively for num_levels of compression.
    Each level reduces token count by 8×.

    No spatial_output or temporal_output params — controlled by num_levels only.
    """

    def __init__(self, dim: int = 768, num_levels: int = 2):
        super().__init__()
        self.num_levels = num_levels
        self.merge_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim * 8),
                nn.Linear(dim * 8, dim),
                nn.SiLU(),
            )
            for _ in range(num_levels)
        ])

    def forward(self, all_slice_tokens: list) -> torch.Tensor:
        """
        Args:
            all_slice_tokens: list of (1, N_patches, D) tensors, one per slice
        Returns:
            merged: (1, final_tokens, D)
        """
        D = all_slice_tokens[0].shape[-1]
        N = all_slice_tokens[0].shape[1]
        H = W = int(N ** 0.5)
        assert H * W == N, (
            f"CubePooler expects square patch grids, got H*W={H*W} != N={N}. "
            f"Input image dimensions may not produce square patches."
        )

        # Reshape each slice to 2D grid, trim to H*W
        grids = []
        for tokens in all_slice_tokens:
            t = tokens.squeeze(0)[:H * W]
            grids.append(t.view(H, W, D))

        # Stack into 5D: (1, S, H, W, D)
        vol = torch.stack(grids, dim=0).unsqueeze(0)

        for merge_layer in self.merge_layers:
            B, S, h, w, d = vol.shape

            # Pad to even dimensions by repeating last element
            if S % 2 != 0:
                vol = torch.cat([vol, vol[:, -1:]], dim=1)
                S += 1
            if h % 2 != 0:
                vol = torch.cat([vol, vol[:, :, -1:]], dim=2)
                h += 1
            if w % 2 != 0:
                vol = torch.cat([vol, vol[:, :, :, -1:]], dim=3)
                w += 1

            # Group into 2×2×2 cubes: (B, S/2, 2, h/2, 2, w/2, 2, d)
            vol = vol.view(B, S // 2, 2, h // 2, 2, w // 2, 2, d)
            # Rearrange to: (B, S/2, h/2, w/2, 2, 2, 2, d)
            vol = vol.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            # Flatten cube: (B, S/2, h/2, w/2, 8*d)
            vol = vol.view(B, S // 2, h // 2, w // 2, 8 * d)
            # Merge via linear projection: (B, S/2, h/2, w/2, d)
            vol = merge_layer(vol)

        B, S, H, W, D = vol.shape
        return vol.view(B, S * H * W, D)
