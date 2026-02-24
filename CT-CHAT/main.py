import torch
from llava.model.multimodal_encoder.qwen_vision_encoder import Dinov3VisionEncoder
from llava.model.multimodal_encoder.ct_clip_encoder import CTCLIPEncoder
from llava.model.multimodal_projector.token_compression import TokenCompressor
from llava.model.qformer import QFormer
from llava.model.vqa_head import VQAHead
from evaluations.green_metrics import GreenMetrics

# MONAI-based CT data loader (replace with your data paths)
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImaged, AddChanneld, ScaleIntensityd, ToTensord

import pandas as pd

def get_vqa_dataset(csv_path):
    # CSV should have columns: image_path, question, answer
    return pd.read_csv(csv_path)

def get_ct_dataloader(image_dir):
    # Dummy loader for demonstration; replace with your actual loader logic
    transforms = [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
    # Example: expects a list of dicts with 'image' key
    file_list = [{"image": f"{image_dir}/sample.nii.gz"}]
    ds = Dataset(data=file_list, transform=transforms)
    loader = DataLoader(ds, batch_size=1)
    return loader

DATASET_PATH = "/midtier/sablab/scratch/data/CT-RATEV2"

def main():
    # 1. Load Encoders
    dinov3_encoder = Dinov3VisionEncoder()
    ctclip_encoder = CTCLIPEncoder()

    # 2. Load Token Compressor
    compressor = TokenCompressor(top_k=64)

    # 3. Load Q-Former
    qformer = QFormer(num_queries=32)

    # 4. Load VQA Head
    vqa_head = VQAHead()

    # 5. Green Metrics Tracker
    green_metrics = GreenMetrics()

    # 6. Load CT data and VQA dataset
    data_dir = "./data"  # Update with your data path
    vqa_csv = "./vqa.csv"  # Update with your VQA CSV path
    ct_loader = get_ct_dataloader(data_dir)
    vqa_data = get_vqa_dataset(vqa_csv)

    # 7. Example main loop (process one batch)
    for batch in ct_loader:
        ct_volume = batch["image"]  # (B, C, D, H, W)
        # For demo, use the first question in VQA CSV
        text_query = [vqa_data.iloc[0]["question"]]

        # 8. Encode with both encoders
        qwen_tokens = dinov3_encoder(ct_volume)
        ctclip_tokens = ctclip_encoder(ct_volume)

        # 9. Token Fusion (concatenate)
        fused_tokens = torch.cat([qwen_tokens, ctclip_tokens], dim=1)

        # 10. Token Compression
        compressed_tokens = compressor(fused_tokens)

        # 11. Q-Former Alignment
        aligned_tokens = qformer(compressed_tokens, text_query)

        # 12. VQA Head
        answer = vqa_head(aligned_tokens, text_query)
        print(f"VQA Answer: {answer}")

        # 13. Green Metrics
        green_metrics.track()
        break  # Remove break to process all batches

if __name__ == "__main__":
    main()
