"""
DINOv3 Vision Encoder with HyperNetwork-Generated Task-Specific LoRA

Uses facebook/dinov3-vitb16-pretrain-lvd1689m from HuggingFace (transformers >= 4.56.0).
DINOv3 architecture (arXiv 2508.10104) with RoPE, layer scale, and drop path.

Architecture:
    3 consecutive CT slices → RGB image → DINOv3 ViT-B → patch tokens
    HyperNetwork(task_one_hot + layer_depth + layer_type) → LoRA weights
    LoRA applied per layer: attention (q_proj, k_proj, v_proj, o_proj)
                          + MLP (up_proj, down_proj)
    TaskClassifier(pooled features) → task predictions

Key design from HyperCT reference (github.com/lfb-1/HyperCT):
    - TaskEncoder: frozen one-hot task embeddings (NOT learnable nn.Embedding)
    - Layer depth encoder: embeds transformer layer index
    - Layer type encoder: embeds target module type (6 modules)
    - MLPResidualBlock: residual connections in weight generator
    - TaskClassifier: classifies from LoRA-adapted features, trains the hypernet

requires_grad_(False) on the backbone IS necessary:
    - Prevents corruption of pretrained weights
    - Only hypernet + LoRA adapters are trainable
    - Saves GPU memory (~86M frozen params)
    - Gradients still flow THROUGH frozen layers for hypernet updates

References:
    - DINOv3: https://arxiv.org/abs/2508.10104
    - HyperCT: https://github.com/lfb-1/HyperCT
    - LoRA: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel
from typing import Dict, List, Optional, Tuple


class TaskEncoder(nn.Module):
    """
    Encodes task identity via frozen one-hot vectors through an MLP.
    One-hot embeddings are NOT learnable — only the MLP projection trains.
    From HyperCT reference: TaskEncoder class.
    """

    def __init__(self, num_tasks: int, encoded_task_emb_size: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.mlp = nn.Sequential(
            nn.Linear(num_tasks, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

    def forward(self, task_idx: torch.Tensor) -> torch.Tensor:
        # Frozen one-hot — no gradients through the one-hot itself
        one_hot = torch.eye(self.num_tasks, device=task_idx.device,
                            dtype=torch.float32)[task_idx]
        return self.mlp(one_hot)


class MLPResidualBlock(nn.Module):
    """Residual MLP block from HyperCT reference."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 pre_layer_norm: bool = True, post_dropout: bool = True):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class LoRAHypernet(nn.Module):
    """
    HyperNetwork following HyperCT reference architecture.

    Generates task-specific LoRA weights conditioned on:
        - Frozen one-hot task embedding (via TaskEncoder)
        - Layer depth embedding (which transformer layer)
        - Layer type embedding (which module: q/k/v/o_proj, up/down_proj)

    Processes through mixer → residual MLPs → per-module output heads.
    Initializes LoRA_B bias to zero so delta_W = B*A starts at 0.

    Follows HyperCT reference: 6 target modules (4 attention + 2 MLP).
    """

    def __init__(
        self,
        target_modules: List[str],
        num_tasks: int = 18,
        num_layers: int = 12,
        lora_rank: int = 16,
        latent_size: int = 128,
        head_in_size: int = 768,
        in_features: Optional[Dict[str, int]] = None,
        out_features: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.target_modules = target_modules
        self.num_layers = num_layers
        self.lora_rank = lora_rank

        if in_features is None:
            self.in_features = {m: 768 for m in target_modules}
        else:
            self.in_features = in_features

        if out_features is None:
            self.out_features = {m: 768 for m in target_modules}
        else:
            self.out_features = out_features

        # Task encoder with frozen one-hot embeddings
        encoded_task_emb_size = latent_size // 2
        self.task_encoder = TaskEncoder(num_tasks, encoded_task_emb_size)

        # Layer depth and type encoders (from HyperCT reference)
        depth_emb_size = latent_size // 4
        type_emb_size = latent_size // 4

        self.layer_depth_encoder = nn.Sequential(
            nn.Embedding(num_layers, depth_emb_size),
            nn.LayerNorm(depth_emb_size),
        )
        self.layer_type_encoder = nn.Sequential(
            nn.Embedding(len(target_modules), type_emb_size),
            nn.LayerNorm(type_emb_size),
        )

        self.module_to_int = {m: i for i, m in enumerate(target_modules)}

        # MLP input = task_emb + depth_emb + type_emb
        mlp_inp_size = encoded_task_emb_size + depth_emb_size + type_emb_size

        # Main processing network (from HyperCT reference)
        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        self.mlp1 = MLPResidualBlock(mlp_inp_size, mlp_inp_size * 4, mlp_inp_size)
        self.mlp2 = MLPResidualBlock(mlp_inp_size, mlp_inp_size * 4, mlp_inp_size)

        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        # Per-module output heads with proper initialization
        self.split_shapes = {}
        heads = {}
        for module in target_modules:
            in_feat = self.in_features[module]
            out_feat = self.out_features[module]
            self.split_shapes[module] = [lora_rank * in_feat, lora_rank * out_feat]
            output_size = self.split_shapes[module][0] + self.split_shapes[module][1]

            layer = nn.Linear(head_in_size, output_size, bias=True)
            # LoRA_A: small random init; LoRA_B: zero init → delta_W starts at 0
            nn.init.normal_(layer.weight, std=0.01)
            with torch.no_grad():
                split_size_A = self.split_shapes[module][0]
                nn.init.normal_(layer.bias[:split_size_A], std=0.01)
                # Zero both weight rows AND bias for B so B_matrices=0 at init
                layer.weight[split_size_A:].zero_()
                layer.bias[split_size_A:].zero_()
            heads[module] = layer

        self.heads = nn.ModuleDict(heads)

    def _embed_layer_depth(self, depth_indices: torch.Tensor) -> torch.Tensor:
        return self.layer_depth_encoder(depth_indices)

    def _embed_layer_type(self, layer_type: str) -> torch.Tensor:
        module_idx = self.module_to_int[layer_type]
        device = next(self.parameters()).device
        module_idx = torch.tensor([module_idx], dtype=torch.long, device=device)
        return self.layer_type_encoder(module_idx)

    def _hypernet_forward(self, layer_indices: torch.Tensor, layer_type: str,
                          encoded_task_emb: torch.Tensor):
        bs = len(layer_indices)
        depth_emb = self._embed_layer_depth(layer_indices)
        layer_type_emb = self._embed_layer_type(layer_type).expand(bs, -1)

        cat_emb = torch.cat([encoded_task_emb, depth_emb, layer_type_emb], dim=-1)

        mlp_inp = self.mixer(cat_emb)
        mlp_out = self.mlp1(mlp_inp)
        mlp_out = self.mlp2(mlp_out)
        head_input = self.mlp3(mlp_out)

        head = self.heads[layer_type]
        head_out = head(head_input)

        return torch.split(head_out, self.split_shapes[layer_type], dim=-1)

    def get_lora_weights(self, layer_indices: torch.Tensor, layer_type: str,
                         task_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LoRA A and B matrices for given layers and task.

        Args:
            layer_indices: (N,) layer indices
            layer_type: target module name (q_proj/k_proj/v_proj/o_proj/up_proj/down_proj)
            task_idx: (1,) or (N,) task index
        Returns:
            A_matrices: (N, rank, in_feat)
            B_matrices: (N, out_feat, rank)
        """
        encoded_task_emb = self.task_encoder(task_idx)
        if encoded_task_emb.dim() == 1:
            encoded_task_emb = encoded_task_emb.unsqueeze(0)

        bs = len(layer_indices)
        if encoded_task_emb.shape[0] != bs:
            encoded_task_emb = encoded_task_emb.expand(bs, -1)

        A_flat, B_flat = self._hypernet_forward(layer_indices, layer_type,
                                                encoded_task_emb)

        in_feat = self.in_features[layer_type]
        out_feat = self.out_features[layer_type]
        A_matrices = A_flat.view(bs, self.lora_rank, in_feat)
        B_matrices = B_flat.view(bs, out_feat, self.lora_rank)

        return A_matrices, B_matrices

    def generate_full_model_lora(self, task_idx: torch.Tensor
                                 ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights for all layers and all target modules.

        Returns:
            Dict[module_name -> {"lora_A": (num_layers, rank, in_feat),
                                 "lora_B": (num_layers, out_feat, rank)}]
        """
        device = next(self.parameters()).device
        layer_indices = torch.arange(self.num_layers, device=device, dtype=torch.long)

        full_lora_weights = {}
        for module_name in self.target_modules:
            A, B = self.get_lora_weights(layer_indices, module_name, task_idx)
            full_lora_weights[module_name] = {"lora_A": A, "lora_B": B}

        return full_lora_weights


class TaskClassifier(nn.Module):
    """
    Classifier that takes globally-pooled LoRA-adapted features from each task
    and produces task predictions. Used to train the HyperNetwork.

    The classifier receives features from a task-specific LoRA pass,
    and gradients flow back through: classifier → features → LoRA → hypernet.
    """

    def __init__(self, input_dim: int = 768, num_tasks: int = 18,
                 hidden_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) globally pooled features from LoRA-adapted encoder
        Returns:
            logits: (B, num_tasks)
        """
        return self.classifier(features)


class DINOv3LoRAEncoder(nn.Module):
    """
    DINOv2 ViT-B encoder with HyperNetwork-generated task-specific LoRA.

    Uses facebook/dinov2-base (public model).
    DINOv2 architecture: RoPE, pre-norm, layer scale, drop path.

    Follows HyperCT reference architecture:
        - Frozen backbone (requires_grad=False prevents pretrained weight
          corruption and saves GPU memory; gradients still flow THROUGH frozen
          layers to update the hypernet)
        - LoRAHypernet with layer depth/type encoders
        - Frozen one-hot task embeddings (via TaskEncoder)
        - 6 LoRA targets: attention q/k/v/o_proj + MLP up/down_proj
        - TaskClassifier for training the hypernet via backprop
    """

    def __init__(self, encoder_name: str = "facebook/dinov2-base",
                 num_tasks: int = 18, lora_rank: int = 16, lora_scaling: float = 1.0,
                 latent_size: int = 128, head_in_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # Freeze backbone — required to prevent pretrained weight corruption
        # and save GPU memory. Only hypernet + classifier are trainable.
        # Gradients still flow through frozen layers for hypernet updates.
        self.encoder.requires_grad_(False)

        # ImageNet normalization (matches HyperCT reference: DINOv3_Encoder.preprocess)
        self.preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.lora_rank = lora_rank
        self.lora_scaling = lora_scaling
        self.scaling = lora_scaling  # Reference uses scaling_factor=1.0 directly

        # Read register tokens from model config
        self.num_register_tokens = getattr(self.encoder.config, 'num_register_tokens', 4)

        # Identify target modules: attention + MLP (6 modules, matching HyperCT reference)
        target_modules = []
        in_features = {}
        out_features = {}

        # Attention modules: q_proj, k_proj, v_proj, o_proj
        sample_attn = self.encoder.model.layer[0].attention
        for module_type in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(sample_attn, module_type)
            target_modules.append(module_type)
            in_features[module_type] = proj.in_features
            out_features[module_type] = proj.out_features

        target_modules.append("o_proj")
        in_features["o_proj"] = sample_attn.o_proj.in_features
        out_features["o_proj"] = sample_attn.o_proj.out_features

        # MLP modules: up_proj, down_proj (HyperCT reference: mlp_up, mlp_down)
        sample_mlp = self.encoder.model.layer[0].mlp
        assert hasattr(sample_mlp, 'up_proj'), f"DINOv3 MLP missing up_proj: {type(sample_mlp)}"
        assert hasattr(sample_mlp, 'down_proj'), f"DINOv3 MLP missing down_proj: {type(sample_mlp)}"
        assert hasattr(sample_mlp, 'act_fn'), f"DINOv3 MLP missing act_fn: {type(sample_mlp)}"
        assert not hasattr(sample_mlp, 'gate_proj'), (
            f"DINOv3 MLP has gate_proj (SwiGLU). The manual MLP decomposition "
            f"(up_proj → act_fn → down_proj) does not handle gating. "
            f"Add gate_proj LoRA support before proceeding.")

        for module_type in ["up_proj", "down_proj"]:
            proj = getattr(sample_mlp, module_type)
            target_modules.append(module_type)
            in_features[module_type] = proj.in_features
            out_features[module_type] = proj.out_features

        self.target_module_names = target_modules
        self.num_encoder_layers = len(self.encoder.model.layer)

        # HyperNetwork with layer depth/type encoders (HyperCT architecture)
        self.hypernet = LoRAHypernet(
            target_modules=target_modules,
            num_tasks=num_tasks,
            num_layers=self.num_encoder_layers,
            lora_rank=lora_rank,
            latent_size=latent_size,
            head_in_size=head_in_size,
            in_features=in_features,
            out_features=out_features,
        )

        # Classifier — takes LoRA-adapted features, trains the hypernet
        encoder_dim = sample_attn.q_proj.out_features
        self.classifier = TaskClassifier(encoder_dim, num_tasks)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims — standard RoPE helper."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to patch tokens only, leaving CLS/register tokens unchanged."""
        num_tokens = q.shape[-2]
        num_patches = cos.shape[-2]
        num_prefix = num_tokens - num_patches

        q_prefix, q_patches = q.split((num_prefix, num_patches), dim=-2)
        k_prefix, k_patches = k.split((num_prefix, num_patches), dim=-2)

        q_patches = (q_patches * cos) + (self._rotate_half(q_patches) * sin)
        k_patches = (k_patches * cos) + (self._rotate_half(k_patches) * sin)

        q = torch.cat((q_prefix, q_patches), dim=-2)
        k = torch.cat((k_prefix, k_patches), dim=-2)
        return q, k

    def forward_with_lora(self, pixel_values: torch.Tensor,
                          lora_weights: Dict[str, Dict[str, torch.Tensor]]
                          ) -> torch.Tensor:
        """
        Forward pass with task-specific LoRA applied per DINOv3 layer.

        DINOv3 layer structure: norm1 → attention(q/k/v/o_proj + RoPE) →
            layer_scale1 → drop_path → norm2 → MLP(up_proj, down_proj) →
            layer_scale2 → drop_path

        LoRA targets (6 modules per layer, matching HyperCT reference):
            attention: q_proj, k_proj, v_proj, o_proj (768→768)
            MLP: up_proj (768→3072), down_proj (3072→768)

        Args:
            pixel_values: (B, 3, H, W) values in [0, 1]
            lora_weights: from LoRAHypernet.generate_full_model_lora()
        Returns:
            patch_tokens: (B, N_patches, D)
        """
        # ImageNet normalization (HyperCT reference: self.preprocess)
        pixel_values = self.preprocess(pixel_values)

        # DINOv3 embeddings: CLS + register_tokens + patch_tokens
        hidden = self.encoder.embeddings(pixel_values)

        # RoPE position embeddings (computed from pixel_values shape)
        cos, sin = self.encoder.rope_embeddings(pixel_values)

        for i, layer in enumerate(self.encoder.model.layer):
            attn = layer.attention
            num_heads = attn.num_heads
            head_dim = attn.head_dim
            Bs, N, D = hidden.shape

            # --- Attention block (pre-norm) ---
            residual = hidden
            hidden_normed = layer.norm1(hidden)

            # Q, K, V with separate per-module LoRA
            q = attn.q_proj(hidden_normed)
            k = attn.k_proj(hidden_normed)
            v = attn.v_proj(hidden_normed)

            for module_name, label in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                if module_name in lora_weights:
                    A = lora_weights[module_name]["lora_A"][i]
                    B_mat = lora_weights[module_name]["lora_B"][i]
                    A = A.to(hidden_normed.device, dtype=hidden_normed.dtype)
                    B_mat = B_mat.to(hidden_normed.device, dtype=hidden_normed.dtype)
                    delta = (hidden_normed @ A.T) @ B_mat.T * self.scaling
                    if label == "q":
                        q = q + delta
                    elif label == "k":
                        k = k + delta
                    else:
                        v = v + delta

            # Multi-head reshape: (B, N, D) → (B, num_heads, N, head_dim)
            q = q.view(Bs, N, num_heads, head_dim).transpose(1, 2)
            k = k.view(Bs, N, num_heads, head_dim).transpose(1, 2)
            v = v.view(Bs, N, num_heads, head_dim).transpose(1, 2)

            # Apply RoPE (patch tokens only, not CLS/register)
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

            # Scaled dot-product attention
            attn_w = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn_w = attn_w.softmax(dim=-1)
            attn_out = (attn_w @ v).transpose(1, 2).reshape(Bs, N, D)

            # Output projection with LoRA
            proj_out = attn.o_proj(attn_out)
            if "o_proj" in lora_weights:
                pA = lora_weights["o_proj"]["lora_A"][i]
                pB = lora_weights["o_proj"]["lora_B"][i]
                pA = pA.to(attn_out.device, dtype=attn_out.dtype)
                pB = pB.to(attn_out.device, dtype=attn_out.dtype)
                proj_out = proj_out + (attn_out @ pA.T) @ pB.T * self.scaling

            # Layer scale + drop path + residual
            hidden = layer.drop_path(layer.layer_scale1(proj_out)) + residual

            # --- MLP block (pre-norm), with LoRA on up_proj and down_proj ---
            residual = hidden
            hidden_normed = layer.norm2(hidden)

            # Decompose MLP forward to apply per-module LoRA (HyperCT reference)
            up_out = layer.mlp.up_proj(hidden_normed)
            if "up_proj" in lora_weights:
                uA = lora_weights["up_proj"]["lora_A"][i]
                uB = lora_weights["up_proj"]["lora_B"][i]
                uA = uA.to(hidden_normed.device, dtype=hidden_normed.dtype)
                uB = uB.to(hidden_normed.device, dtype=hidden_normed.dtype)
                up_out = up_out + (hidden_normed @ uA.T) @ uB.T * self.scaling

            up_out = layer.mlp.act_fn(up_out)

            mlp_out = layer.mlp.down_proj(up_out)
            if "down_proj" in lora_weights:
                dA = lora_weights["down_proj"]["lora_A"][i]
                dB = lora_weights["down_proj"]["lora_B"][i]
                dA = dA.to(up_out.device, dtype=up_out.dtype)
                dB = dB.to(up_out.device, dtype=up_out.dtype)
                mlp_out = mlp_out + (up_out @ dA.T) @ dB.T * self.scaling

            hidden = layer.drop_path(layer.layer_scale2(mlp_out)) + residual

        # Final layernorm
        hidden = self.encoder.norm(hidden)

        # Drop CLS + register tokens, return only patch tokens
        return hidden[:, 1 + self.num_register_tokens:, :]

    def encode_slice(self, pixel_values: torch.Tensor,
                     task_id: torch.Tensor) -> torch.Tensor:
        """Encode a single slice with task-specific LoRA."""
        lora_weights = self.hypernet.generate_full_model_lora(task_id)
        return self.forward_with_lora(pixel_values, lora_weights)

    def classify(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Classify from globally-pooled LoRA-adapted features."""
        pooled = patch_tokens.mean(dim=1)  # (B, D) global average pool
        return self.classifier(pooled)
