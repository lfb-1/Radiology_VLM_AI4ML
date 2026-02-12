from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch
from PIL import Image
import numpy as np

class CTCLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("CT-CLIP/CT-CLIP-ViT-B-16")
        self.model = CLIPModel.from_pretrained("CT-CLIP/CT-CLIP-ViT-B-16").eval()

    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        all_tokens = []
        for i in range(b):
            slices = x[i, 0]  # (D, H, W)
            pil_images = [Image.fromarray((s.cpu().numpy() * 255).astype(np.uint8)).convert("RGB") for s in slices]
            pil_images = pil_images[::max(1, d // 16)]
            tokens = []
            for img in pil_images:
                inputs = self.processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    out = self.model.get_image_features(**inputs)
                tokens.append(out)
            tokens = torch.stack(tokens, dim=1)  # (1, N_slices, C)
            all_tokens.append(tokens)
        return torch.cat(all_tokens, dim=0)  # (B, N_slices, C)
