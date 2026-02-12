import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import monai


import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImaged, AddChanneld, ScaleIntensityd, ToTensord

def get_ctrate_file_list(image_dir, csv_path):
    # CT-RATE CSV should have columns: image_id, question, answer
    df = pd.read_csv(csv_path)
    file_list = []
    for idx, row in df.iterrows():
        image_id = row['image_id']
        image_path = os.path.join(image_dir, f"{image_id}.nii.gz")
        if os.path.exists(image_path):
            file_list.append({"image": image_path, "question": row['question'], "answer": row['answer']})
    return file_list

def get_ctrate_dataloader(image_dir, csv_path, batch_size=1):
    file_list = get_ctrate_file_list(image_dir, csv_path)
    transforms = [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
    ds = Dataset(data=file_list, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, file_list

# Example usage:
# dataset = CT3DVQADataset('data/ct', 'data/vqa.csv')
# loader = DataLoader(dataset, batch_size=2, shuffle=True)
