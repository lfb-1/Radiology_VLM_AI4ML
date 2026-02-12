# Medical Imaging VLM Project

This project implements a Visual Language Model (VLM) for 3D medical imaging using dual encoders (Qwen 2.5 Vision and CT-CLIP/CT-RATE ViT), token compression, Q-Former alignment, a VQA head, and green metrics tracking.

## Main Components
- Qwen 2.5 Vision Encoder (multi-resolution support)
- CT-CLIP/CT-RATE Encoder (ViT backbone, medical domain)
- Token Fusion Layer (concat/cross-attention)
- 3D-aware Token Compression (top-k by attention/saliency)
- Q-Former (aligns image tokens with text queries)
- VQA Head (Visual Question Answering)
- Green Metrics (FLOPs, energy, CO2, latency)

## Frameworks & Libraries
- PyTorch
- HuggingFace Transformers
- MONAI (for medical imaging)
- CodeCarbon (for green metrics)

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your 3D CT data and VQA dataset (e.g., VQA-RAD, PathVQA)
3. Run the main pipeline: `python main.py`

## Directory Structure
- `encoders/` - Vision encoder modules
- `compression/` - Token compression modules
- `qformer/` - Q-Former implementation
- `vqa/` - VQA head and evaluation
- `metrics/` - Green metrics utilities
- `main.py` - Pipeline entry point
- `requirements.txt` - Python dependencies

## Notes
- Pretrained weights for Qwen 2.5 Vision and CT-CLIP/CT-RATE are required (see code comments for download links or instructions).
- This project is modular and can be extended for other medical imaging modalities.
