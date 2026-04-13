"""
VLM Training Script for HyperCT_UPDT Pipeline

Trains a Vision-Language Model using precomputed DINOv3 vision tokens:
    1. Load precomputed tokens (.npz) per CT volume
    2. Pass through Q-Former adapter → (num_queries, 4096)
    3. Inject into Llama 3.1 8B at IMAGE_TOKEN positions
    4. Fine-tune LLM with LoRA on VQA pairs

Usage:
    torchrun --nproc_per_node=4 train_vlm.py \
        --tokens_dir ./precomputed_tokens \
        --data_json ./train_data.json \
        --output_dir ./checkpoints \
        --llm_name meta-llama/Llama-3.1-8B-Instruct
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RADIOLOGICAL_TASKS
from models.qformer import QFormerAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class VQADataset(Dataset):
    """
    Dataset loading precomputed vision tokens + VQA conversations.

    Expected data_json format (list of dicts):
        {
            "id": "scan_001",
            "image": "scan_001.npz",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe findings."},
                {"from": "gpt", "value": "The scan shows..."}
            ]
        }
    """

    def __init__(self, data_json: str, tokens_dir: str, tokenizer,
                 max_length: int = 2048, num_task_tokens: int = 3):
        with open(data_json, "r") as f:
            self.data = json.load(f)
        self.tokens_dir = tokens_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_task_tokens = num_task_tokens

        # Llama 3.1 uses <|eot_id|> for end-of-turn, distinct from eos_token_id
        self.eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_token_id is None or self.eot_token_id == tokenizer.unk_token_id:
            self.eot_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load precomputed tokens
        npz_path = os.path.join(self.tokens_dir, item["image"])
        npz_data = np.load(npz_path, allow_pickle=True)
        all_tokens = npz_data["tokens"]       # (num_tasks, T_out, 768)
        predictions = npz_data["predictions"]  # (num_tasks, num_tasks)

        # Select top-k tasks by classifier confidence (diagonal logits)
        # Diagonal[i] = how well task i's LoRA detects task i in this volume
        diag = np.diag(predictions)
        k = min(self.num_task_tokens, len(diag))
        top_indices = np.argsort(diag)[-k:]  # highest-confidence tasks
        vision_tokens = torch.from_numpy(all_tokens[top_indices]).float()
        vision_tokens = vision_tokens.reshape(-1, vision_tokens.shape[-1])

        # Build conversation
        convs = item["conversations"]
        text_parts = []
        for conv in convs:
            role = conv["from"]
            content = conv["value"]
            if role == "human":
                text_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "gpt":
                text_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")

        full_text = "<|begin_of_text|>" + "".join(text_parts)

        # Tokenize — split around IMAGE_TOKEN
        chunks = full_text.split(IMAGE_TOKEN)
        input_ids = []
        image_token_positions = []

        for i, chunk in enumerate(chunks):
            if chunk:
                chunk_ids = self.tokenizer.encode(chunk, add_special_tokens=False)
                input_ids.extend(chunk_ids)
            if i < len(chunks) - 1:
                image_token_positions.append(len(input_ids))
                input_ids.append(IMAGE_TOKEN_INDEX)

        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)

        # Filter positions that survived truncation
        image_token_positions = [p for p in image_token_positions if p < len(input_ids)]

        # Labels: mask everything before assistant responses
        labels = input_ids.clone()
        # Mask image tokens
        labels[input_ids == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # Mask user turns (everything before assistant header)
        assistant_header = self.tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
        )
        header_len = len(assistant_header)

        # Simple masking: mask everything that isn't in an assistant turn.
        # The assistant's <|eot_id|> IS included in the loss so the model
        # learns to produce the end-of-turn signal.
        in_assistant = False
        for i in range(len(labels)):
            if i + header_len <= len(input_ids):
                if input_ids[i:i+header_len].tolist() == assistant_header:
                    in_assistant = True
            if not in_assistant:
                labels[i] = IGNORE_INDEX
            # Llama 3.1: <|eot_id|> marks end-of-turn (NOT eos_token_id).
            # Check AFTER the masking decision so eot_id in assistant
            # turns is supervised (model learns when to stop).
            if input_ids[i].item() == self.eot_token_id:
                in_assistant = False

        return {
            "input_ids": input_ids,
            "labels": labels,
            "vision_tokens": vision_tokens,
            "image_positions": torch.tensor(image_token_positions, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    """Pad sequences and stack vision tokens."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    labels = []
    attention_mask = []
    vision_tokens = []
    image_positions = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(F.pad(b["labels"], (0, pad_len), value=IGNORE_INDEX))
        mask = torch.ones(seq_len, dtype=torch.long)
        attention_mask.append(F.pad(mask, (0, pad_len), value=0))
        vision_tokens.append(b["vision_tokens"])
        image_positions.append(b["image_positions"])

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "vision_tokens": vision_tokens,  # list of variable-size tensors
        "image_positions": image_positions,
    }


class HyperCTVLM(nn.Module):
    """
    Full VLM: Q-Former adapter + LLM.

    Forward:
        1. Q-Former: vision_tokens (B, N, 768) → (B, num_queries, 4096)
        2. Replace IMAGE_TOKEN positions in input embeddings with vision features
        3. LLM forward with modified embeddings
    """

    def __init__(self, llm, qformer: QFormerAdapter):
        super().__init__()
        self.llm = llm
        self.qformer = qformer

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate to LLM so HuggingFace Trainer can enable gradient checkpointing."""
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, input_ids, labels, attention_mask, vision_tokens, image_positions):
        device = input_ids.device
        B = input_ids.shape[0]

        # Vision token injection changes sequence length per sample,
        # so batch_size must be 1 (gradient accumulation handles effective batch).
        assert B == 1, (
            f"HyperCTVLM requires batch_size=1 per GPU (got {B}). "
            "Use gradient_accumulation_steps for larger effective batch."
        )

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(
            input_ids.clamp(min=0)  # clamp IMAGE_TOKEN_INDEX to valid range
        )  # (1, seq_len, hidden_dim)

        if len(image_positions[0]) > 0:
            # Process vision tokens through Q-Former
            v_tokens = vision_tokens[0].unsqueeze(0).to(device)  # (1, N, 768)
            aligned = self.qformer(v_tokens)  # (1, num_queries, 4096)
            aligned = aligned.squeeze(0).to(text_embeds.dtype)  # (num_queries, 4096)
            num_vision = aligned.shape[0]

            # Replace IMAGE_TOKEN placeholder with vision features
            pos = image_positions[0][0].item()
            pre = text_embeds[0, :pos]           # before <image>
            post = text_embeds[0, pos + 1:]      # after <image>
            new_embeds = torch.cat([pre, aligned, post], dim=0).unsqueeze(0)

            # Adjust labels: mask vision token positions
            pre_labels = labels[0, :pos]
            post_labels = labels[0, pos + 1:]
            vision_labels = torch.full(
                (num_vision,), IGNORE_INDEX, device=device, dtype=labels.dtype
            )
            new_labels = torch.cat([pre_labels, vision_labels, post_labels]).unsqueeze(0)

            # Adjust attention mask
            pre_mask = attention_mask[0, :pos]
            post_mask = attention_mask[0, pos + 1:]
            vision_mask = torch.ones(num_vision, device=device, dtype=attention_mask.dtype)
            new_mask = torch.cat([pre_mask, vision_mask, post_mask]).unsqueeze(0)
        else:
            new_embeds = text_embeds
            new_labels = labels
            new_mask = attention_mask

        outputs = self.llm(
            inputs_embeds=new_embeds,
            labels=new_labels,
            attention_mask=new_mask,
        )
        return outputs


def find_linear_names(model):
    """Find all linear layer names for LoRA, excluding vision/projector modules."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            # Skip embeddings and head
            if any(k in name for k in ["embed", "lm_head", "qformer"]):
                continue
            names.add(parts[-1])
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description="Train HyperCT VLM")
    parser.add_argument("--tokens_dir", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vlm")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--llm_hidden_size", type=int, default=4096)
    parser.add_argument("--vision_dim", type=int, default=768)
    parser.add_argument("--num_queries", type=int, default=64)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--qformer_heads", type=int, default=12)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_task_tokens", type=int, default=3,
                        help="Number of top tasks to select based on classifier confidence")
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--qformer_checkpoint", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "flash_attention_2", "sdpa"])
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LLM
    log.info(f"Loading LLM: {args.llm_name}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_implementation,
    )

    # LoRA on LLM
    target_modules = find_linear_names(llm)
    log.info(f"LoRA target modules: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # Q-Former adapter
    qformer = QFormerAdapter(
        vision_dim=args.vision_dim,
        llm_dim=args.llm_hidden_size,
        num_queries=args.num_queries,
        num_layers=args.qformer_layers,
        num_heads=args.qformer_heads,
    )

    if args.qformer_checkpoint:
        state = torch.load(args.qformer_checkpoint, map_location="cpu", weights_only=True)
        qformer.load_state_dict(state)
        log.info(f"Loaded Q-Former checkpoint: {args.qformer_checkpoint}")

    # Dataset
    dataset = VQADataset(args.data_json, args.tokens_dir, tokenizer, args.max_length,
                          num_task_tokens=args.num_task_tokens)
    log.info(f"Dataset: {len(dataset)} samples")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Combine Q-Former + LLM into single model
    model = HyperCTVLM(llm, qformer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    trainer.train()

    # Save Q-Former separately
    qformer_path = os.path.join(args.output_dir, "qformer_final.pt")
    torch.save(qformer.state_dict(), qformer_path)
    log.info(f"Saved Q-Former to {qformer_path}")

    # Save LoRA adapter
    llm.save_pretrained(os.path.join(args.output_dir, "llm_lora"))
    log.info("Training complete.")


if __name__ == "__main__":
    main()
