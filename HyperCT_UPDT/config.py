"""
HyperCT_UPDT Configuration

Central config for all pipeline stages. All paths and hyperparameters
are overridable via CLI argparse in each script.
"""

from dataclasses import dataclass
from typing import Tuple


RADIOLOGICAL_TASKS = [
    "opacity", "nodule", "consolidation", "atelectasis",
    "pleural_effusion", "cardiomegaly", "emphysema", "fibrosis",
    "bronchiectasis", "lymphadenopathy", "mass", "pneumothorax",
    "pericardial_effusion", "calcification", "medical_material",
    "mosaic_attenuation", "peribronchial_thickening", "hiatal_hernia",
]


@dataclass
class VisionConfig:
    encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    encoder_dim: int = 768
    num_slices: int = 33  # divisible by 3
    slice_size: Tuple[int, int] = (512, 512)  # multiple of DINOv3 patch_size=16
    cube_pool_levels: int = 2  # 2x2x2 cube merging levels
    lora_rank: int = 16
    lora_scaling: float = 1.0  # LoRA output scaling factor (reference default)
    lora_dropout: float = 0.05


@dataclass
class HyperNetConfig:
    num_tasks: int = len(RADIOLOGICAL_TASKS)
    lora_rank: int = 16
    latent_size: int = 128
    head_in_size: int = 768  # matches DINOv3 feature dim (reference default)


@dataclass
class QFormerConfig:
    num_queries: int = 64
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1


@dataclass
class VLMConfig:
    llm_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_hidden_size: int = 4096
    vision_dim: int = 768
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 1  # must be 1 per GPU (HyperCTVLM requires B=1)
    gradient_accumulation_steps: int = 8
