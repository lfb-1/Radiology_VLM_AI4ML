"""
Hook-Based LoRA Manager for DINOv3 with HyperNetwork

Applies dynamic LoRA weights via forward hooks instead of monkey-patching.
Adapted from HyperCT reference (github.com/lfb-1/HyperCT) lora_hooks.py.

Key design:
    - Register forward hooks on all 6 target Linear layers per encoder layer
    - Hooks intercept output and add  scaling * (input @ A^T) @ B^T
    - Activate/deactivate cleanly; no permanent model modification
    - Supports per-layer indexing via DINOv3 transformer layer structure
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)

# Maps DINOv3 module path suffixes to our hypernet target names
_MODULE_NAME_MAP = {
    "attention.q_proj": "q_proj",
    "attention.k_proj": "k_proj",
    "attention.v_proj": "v_proj",
    "attention.o_proj": "o_proj",
    "mlp.up_proj": "up_proj",
    "mlp.down_proj": "down_proj",
}


class HookBasedLoRAManager:
    """
    Manages dynamic LoRA via forward hooks on DINOv3 Linear layers.

    Usage:
        manager = HookBasedLoRAManager(encoder, scaling=1.0)
        manager.register_hooks()
        lora_w = hypernet.generate_full_model_lora(task_id)
        manager.set_lora_weights(lora_w)
        manager.activate()
        out = encoder(pixel_values)   # hooks add LoRA deltas
        manager.deactivate()
        manager.remove_hooks()
    """

    def __init__(self, encoder: nn.Module, scaling: float = 1.0):
        self.encoder = encoder
        self.scaling = scaling
        self._hooks = []
        self._lora_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active = False

    def _parse_layer_and_module(self, full_name: str):
        """
        Parse DINOv3 module path to extract layer index and target name.

        DINOv3 paths look like: *.layer.{i}.attention.q_proj
        """
        parts = full_name.split(".")
        layer_idx = None
        for j, part in enumerate(parts):
            if part == "layer" and j + 1 < len(parts):
                try:
                    layer_idx = int(parts[j + 1])
                    break
                except ValueError:
                    continue

        if layer_idx is None:
            return None, None

        # Check suffix against target module map
        for suffix, target_name in _MODULE_NAME_MAP.items():
            if full_name.endswith(suffix):
                return layer_idx, target_name

        return layer_idx, None

    def _make_hook(self, layer_idx: int, target_name: str):
        """Create a forward hook that adds LoRA delta to module output."""

        def hook_fn(module, args, output):
            if not self._active:
                return output

            if target_name not in self._lora_weights:
                return output

            lora_A = self._lora_weights[target_name]["lora_A"]  # (num_layers, rank, in)
            lora_B = self._lora_weights[target_name]["lora_B"]  # (num_layers, out, rank)

            if layer_idx >= lora_A.shape[0]:
                return output

            A = lora_A[layer_idx]  # (rank, in_feat)
            B = lora_B[layer_idx]  # (out_feat, rank)

            input_tensor = args[0]
            A = A.to(input_tensor.device, dtype=input_tensor.dtype)
            B = B.to(input_tensor.device, dtype=input_tensor.dtype)

            # LoRA delta: (input @ A^T) @ B^T  * scaling
            delta = (input_tensor @ A.T) @ B.T * self.scaling
            return output + delta

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all target Linear layers in DINOv3."""
        self.remove_hooks()
        for name, module in self.encoder.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            layer_idx, target_name = self._parse_layer_and_module(name)
            if layer_idx is not None and target_name is not None:
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx, target_name)
                )
                self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def set_lora_weights(self, lora_weights: Dict[str, Dict[str, torch.Tensor]]):
        """Set LoRA weights for the current forward pass."""
        self._lora_weights = lora_weights

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False
        self._lora_weights = {}


@contextmanager
def dynamic_lora_context(encoder: nn.Module, scaling: float = 1.0):
    """
    Context manager for dynamic LoRA application.

    Usage:
        with dynamic_lora_context(encoder, scaling) as manager:
            lora_w = hypernet.generate_full_model_lora(task_id)
            manager.set_lora_weights(lora_w)
            manager.activate()
            out = encoder_forward(pixel_values)  # hooks add LoRA
            manager.deactivate()
    """
    manager = HookBasedLoRAManager(encoder, scaling)
    manager.register_hooks()
    try:
        yield manager
    finally:
        manager.remove_hooks()
