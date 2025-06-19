import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import ast
import random
import time
from tqdm import tqdm
import sys
from tqdm import tqdm

# Add Unet directory to path if needed
sys.path.append("./Unet")
from unet import UNet


def organ_mapper():
    label_dict = {
        0: "background",
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "inferior vena cava",
        10: "portal vein & splenic vein",
        11: "pancreas",
        12: "right adrenal gland",
        13: "left adrenal gland",
    }
    key_to_organ = label_dict
    organ_to_key = {v.lower(): k for k, v in label_dict.items()}
    return key_to_organ, organ_to_key


def clip_and_rescale(ct_array, min_hu=-175, max_hu=250):
    clipped = np.clip(ct_array, min_hu, max_hu)
    return (clipped + abs(min_hu)) / (max_hu + abs(min_hu))


def to_tensor(image, mask):
    img_t = torch.from_numpy(image).float().unsqueeze(0)  # or permute channels
    msk_t = torch.from_numpy(mask).float().unsqueeze(0)
    return img_t, msk_t


# Custom dataset for BTCV slices
class BTCVSliceDataset(Dataset):
    def __init__(self, dataset, organ_name, transform=None, binary=True):
        """
        Args:
            dataset: List of dictionaries with img_path, mask_path, organ, slices_idx
            organ_key: Integer key for the organ to segment
            transform: Optional transforms
            binary: If True, create binary masks (organ vs background)
        """
        self.dataset = dataset
        self.organ_name = organ_name
        self.transform = transform
        self.binary = binary

        # Flatten the dataset to access slices directly
        self.slices = []
        for _, item in dataset.iterrows():
            if item["organ"].lower() == organ_name.lower():
                for slice_idx in range(item["begin_slice"], item["end_slice"] + 1):
                    self.slices.append(
                        {
                            "img_path": item["img_path"],
                            "mask_path": item["mask_path"],
                            "slice_idx": slice_idx,
                        }
                    )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_info = self.slices[idx]

        # Load 3D volume
        img = nib.load(slice_info["img_path"])
        mask = nib.load(slice_info["mask_path"])

        # Get the specific slice
        img_array = img.dataobj[:, :, slice_info["slice_idx"]].astype(
            np.float32
        )  # (H, W)
        mask_array = mask.dataobj[:, :, slice_info["slice_idx"]].astype(
            np.float32
        )  # (H, W)

        # Normalize image to [0, 1]
        img_slice = clip_and_rescale(img_array)
        mask_slice = mask_array
        # Create binary mask (organ vs background)
        if self.binary:
            key_to_organ, organ_to_key = organ_mapper()
            organ_key = organ_to_key[self.organ_name.lower()]
            mask_slice = (mask_slice == organ_key).astype(np.float32)

        # Add channel dimension
        # img_slice = img_slice[np.newaxis, ...]  # (1, H, W)
        # mask_slice = mask_slice[np.newaxis, ...]  # (1, H, W)

        # Convert to torch tensors
        img_tensor, mask_tensor = to_tensor(img_slice, mask_slice)

        return img_tensor, mask_tensor
