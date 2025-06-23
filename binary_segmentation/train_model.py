import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from tqdm import tqdm
import sys
from tqdm import tqdm
from functools import lru_cache
from datetime import datetime
import time

# Add Unet directory to path if needed
sys.path.append("../Unet")
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


def to_tensor(volume, mask):
    vol_t = torch.from_numpy(volume).float().unsqueeze(0)  # or permute channels
    msk_t = torch.from_numpy(mask).float().unsqueeze(0)
    return vol_t, msk_t


def save_progress_plots(history, epoch, num_epochs, organ_name, output_dir):
    """
    Saves the training and validation plots to a PNG file.
    """
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Progress for {organ_name} - Epoch {epoch + 1}/{num_epochs}")

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Dice Score
    plt.subplot(1, 2, 2)
    # Convert tensor to CPU
    train_dice_scores_cpu = [
        d.cpu().item() if hasattr(d, "cpu") else d for d in history["train_dice_scores"]
    ]
    val_dice_scores_cpu = [
        d.cpu().item() if hasattr(d, "cpu") else d for d in history["val_dice_scores"]
    ]
    plt.plot(train_dice_scores_cpu, label="Training Dice Score")
    plt.plot(val_dice_scores_cpu, label="Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(output_dir / f"training_progress_{organ_name}.png")

    plt.close()


# Custom dataset for BTCV slices
class BTCVSliceDataset(Dataset):
    def __init__(self, dataset, organ_name, transform=None, binary=True):
        """
        Args:
            dataset: List of dictionaries with volume_path, mask_path, organ, slices_idx
            organ_key: Integer key for the organ to segment
            transform: Optional transforms
            binary: If True, create binary masks (organ vs background)
        """
        self.dataset = dataset
        self.organ_name = organ_name
        self.transform = transform
        self.binary = binary

        self.volumes = {}
        self.masks = {}
        # Get unique vlomues/masks paths
        unique_volumes_paths = dataset["img_path"].unique()
        unique_masks_paths = dataset["mask_path"].unique()
        for path in tqdm(unique_volumes_paths, desc="Loading volumes"):
            self.volumes[path] = nib.load(path).get_fdata(dtype=np.float32)

        for path in tqdm(unique_masks_paths, desc="Loading masks"):
            self.masks[path] = nib.load(path).get_fdata(dtype=np.float32)
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

    # @lru_cache(maxsize=30)
    # def _get_volume(self, path):
    #     return nib.load(path).get_fdata(dtype=np.float32)

    def __getitem__(self, idx):
        slice_info = self.slices[idx]
        volume_path = slice_info["img_path"]
        mask_path = slice_info["mask_path"]

        # Load 3D volume
        volume = self.volumes[volume_path]
        mask = self.masks[mask_path]

        # Get the specific slice
        vol_array = volume[:, :, slice_info["slice_idx"]].astype(np.float32)  # (H, W)
        mask_array = mask[:, :, slice_info["slice_idx"]].astype(np.float32)  # (H, W)

        # Normalize image to [0, 1]
        vol_slice = clip_and_rescale(vol_array)
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
        vol_tensor, mask_tensor = to_tensor(vol_slice, mask_slice)

        return vol_tensor, mask_tensor


# Training function
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=25,
    patience=5,
    device="cuda",
    output_dir="./results",
):
    """Train the UNet model for semantic segmentation"""

    # Initialize metrics tracking
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice_scores": [],
        "val_dice_scores": [],
    }
    output_dir = (
        Path(output_dir) / f"{organ_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()
    print("Starting model training...")

    print(f"Early stopping enabled with patience of {patience} epochs.")
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_dice = 0.0

            # Iterate over data
            for inputs, masks in tqdm(dataloader):
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                    # Calculate loss
                    loss = criterion(outputs, masks)

                    # Calculate Dice score
                    dice = dice_coefficient(preds, masks)

                    # Backward + optimize only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_dice += dice * inputs.size(0)

            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}")

            # Save metrics
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_dice_scores"].append(epoch_dice.cpu())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_dice_scores"].append(epoch_dice.cpu())
                # Save best model
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(
                        model.state_dict(), output_dir / f"best_unet_{organ_name}.pth"
                    )
                    print(f"New best model saved with val loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(
                        f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}"
                    )
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered. No improvement for {patience} epochs."
                    )
                    break

            # Print progress
            print(f"\nSaving progress plots at epoch {epoch + 1}...")
            save_progress_plots(history, epoch, num_epochs, organ_name, output_dir)

    total_end_time = time.time()
    print("Training finished.")
    total_duration_seconds = total_end_time - total_start_time
    # Format it into hours, minutes, and seconds for readability
    hours, rem = divmod(total_duration_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Training Time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")

    # Save final model
    torch.save(model.state_dict(), output_dir / f"final_unet_{organ_name}.pth")
    return model, history, total_duration_seconds


# Dice coefficient for evaluation
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """Calculate Dice coefficient"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


# Main execution
if __name__ == "__main__":
    # Parameters
    organ_list = ["liver"]  # Change to the organ you want to segment
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./results"

    # Get organ key from organ name
    key_to_organ, organ_to_key = organ_mapper()

    # Load dataset
    # Make sure to parse the string representation of lists if loaded from CSV
    # if isinstance(train_df, pd.DataFrame) and "organs" in train_df.columns:
    #     train_df["organs"] = train_df["organs"].apply(ast.literal_eval)
    data_path = "/home/ismail/projet_PFE/Hands-on-nnUNet/nnUNetFrame/dataset/RawData"
    train_set = pd.read_csv(os.path.join(data_path, "split", "train_dataset.csv"))
    test_set = pd.read_csv(os.path.join(data_path, "split", "test_dataset.csv"))
    # Create datasets
    for organ_name in organ_list:
        if organ_name.lower() not in organ_to_key:
            print(f"Organ '{organ_name}' not found in the dataset. Skipping...")
            continue
        train_dataset = BTCVSliceDataset(train_set, organ_name, binary=True)
        val_dataset = BTCVSliceDataset(test_set, organ_name, binary=True)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # Initialize model
        model = UNet(in_channels=1, out_channels=1).to(device)

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        model, history, total_duration_seconds = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            device=device,
            output_dir=output_dir,
        )

        print("Saving training history to JSON file...")
        # On doit convertir les tenseurs CPU en simples nombres pour le JSON
        history_for_json = {
            key: (
                [val.item() for val in value]
                if isinstance(value[0], torch.Tensor)
                else value
            )
            for key, value in history.items()
        }

        with open(output_dir / f"training_history_{organ_name}.json", "w") as f:
            json.dump(history_for_json, f, indent=4)

        print("History saved successfully.")

        hours, rem = divmod(total_duration_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.suptitle(
            f"Training History for {organ_name}\nTotal Training Time: {time_str}"
        )
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["train_dice_scores"], label="Training Dice Score")
        plt.plot(history["val_dice_scores"], label="Validation Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.legend()

        # Adjust layout to prevent the main title from overlapping with subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(output_dir / f"training_history_{organ_name}.png")
        plt.show()
