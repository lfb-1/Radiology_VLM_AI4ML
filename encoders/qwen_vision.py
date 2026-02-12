from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
from PIL import Image
import numpy as np

class QwenVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True).eval()

    def forward(self, x):
        # x: (B, C, D, H, W) - batch of 3D CT volumes
        # Convert each 3D volume to a list of 2D PIL images (axial slices)
        b, c, d, h, w = x.shape
        all_tokens = []
        for i in range(b):
            slices = x[i, 0]  # (D, H, W)
            pil_images = [Image.fromarray((s.cpu().numpy() * 255).astype(np.uint8)).convert("RGB") for s in slices]
            # Use only a subset of slices for efficiency (e.g., every 4th slice)
            pil_images = pil_images[::max(1, d // 16)]
            # Tokenize and encode each slice
            tokens = []
            for img in pil_images:
                inputs = self.tokenizer(images=img, return_tensors="pt")
                with torch.no_grad():
                    out = self.model.vision_tower(img.unsqueeze(0) if hasattr(self.model, 'vision_tower') else img)
                # Use pooled output or flatten as needed
                if hasattr(out, 'last_hidden_state'):
                    tokens.append(out.last_hidden_state.mean(dim=1))
                elif isinstance(out, torch.Tensor):
                    tokens.append(out.mean(dim=(2, 3)))
                else:
                    tokens.append(torch.zeros(1, 32))
            tokens = torch.cat(tokens, dim=0).unsqueeze(0)  # (1, N_slices, C)
            all_tokens.append(tokens)
        return torch.cat(all_tokens, dim=0)  # (B, N_slices, C)
