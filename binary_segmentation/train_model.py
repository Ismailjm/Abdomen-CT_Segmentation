# Import necessary libraries
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
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Add Unet directory to path if needed
sys.path.append("../Unet")
from unet import UNet  # type: ignore


def organ_mapper():
    """
    Maps organ names to integer keys and vice versa.
    Returns:
        key_to_organ (dict): Mapping from integer keys to organ names.
        organ_to_key (dict): Mapping from organ names to integer keys.
    """
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
    """
    Clip and rescale CT array to [0, 1] range.
    Args:
        ct_array (np.ndarray): Input CT array.
        min_hu (int): Minimum Hounsfield unit to clip.
        max_hu (int): Maximum Hounsfield unit to clip.
    Returns:
        np.ndarray: Rescaled array in [0, 1] range.
    """
    clipped = np.clip(ct_array, min_hu, max_hu)
    return (clipped + abs(min_hu)) / (max_hu + abs(min_hu))


def to_tensor(volume, mask):
    """
    Convert numpy arrays to PyTorch tensors and add batch dimension.
    Args:
        volume (np.ndarray): Input volume array.
        mask (np.ndarray): Input mask array.
    Returns:
        vol_t (torch.Tensor): Volume tensor with shape (1, C, H, W).
        msk_t (torch.Tensor): Mask tensor with shape (1, C, H, W).
    """
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
    def __init__(
        self, dataset, organ_name, by_patient=False, organ_threshold=0, binary=True
    ):
        """
        Args:
            dataset: List of dictionaries with volume_path, mask_path, organ, slices_idx
            organ_key: Integer key for the organ to segment
            transform: Optional transforms
            binary: If True, create binary masks (organ vs background)
        """
        self.dataset = dataset
        self.organ_name = organ_name
        self.by_patient = by_patient
        self.organ_threshold = organ_threshold
        self.binary = binary

        self.volumes = {}
        self.masks = {}
        _, organ_to_key = organ_mapper()
        self.organ_key = organ_to_key[organ_name.lower()]

        # Get unique volmues/masks paths
        unique_volumes_paths = dataset["img_path"].unique()
        unique_masks_paths = dataset["mask_path"].unique()
        for path in tqdm(unique_volumes_paths, desc="Loading volumes"):
            self.volumes[path] = nib.load(path).get_fdata(dtype=np.float32)

        # Load masks and create binary masks for the specified organ
        if self.binary:
            for path in tqdm(unique_masks_paths, desc="Loading masks"):
                self.masks[path] = (
                    (nib.load(path).get_fdata(dtype=np.float32)) == self.organ_key
                ).astype(np.float32)

        total_slices_found = 0
        slices_kept = 0
        self.slices = []
        for _, item in dataset.iterrows():
            if item["organ"].lower() == organ_name.lower():
                masks_data = self.masks[item["mask_path"]]
                max_mask_slice = self._extract_max_slice_simple(item)
                # middle_slice_mask_array = masks_data[:, :, (item["begin_slice"]+item["end_slice"]) // 2]
                for slice_idx in range(item["begin_slice"], item["end_slice"] + 1):
                    total_slices_found += 1

                    mask_array = masks_data[:, :, slice_idx]

                    organ_perc = mask_array.sum() / max_mask_slice.sum()

                    # Only add the slice if it meets the threshold
                    if organ_perc >= self.organ_threshold:
                        slices_kept += 1
                        self.slices.append(
                            {
                                "img_path": item["img_path"],
                                "mask_path": item["mask_path"],
                                "slice_idx": slice_idx,
                            }
                        )
        print(f"Total slices found: {total_slices_found}")
        print(f"Total slices kept: {slices_kept}")

    def _extract_max_slice_simple(self, item):
        """
        Finds the slice with the maximum number of foreground pixels for a single volume.
        """
        mask_volume = self.masks[item["mask_path"]]

        best_slice_index = -1
        max_pixel_sum = -1

        # Loop through all the relevant slices for this volume
        for slice_idx in range(item["begin_slice"], item["end_slice"] + 1):
            # Get the slice from the pre-loaded volume
            current_slice = mask_volume[:, :, slice_idx]
            current_pixel_sum = current_slice.sum()

            # If the current slice is better than the best one seen so far, update it
            if current_pixel_sum > max_pixel_sum:
                max_pixel_sum = current_pixel_sum
                best_slice_index = slice_idx

        # print(
        #     f"Found best slice at index {best_slice_index} with {max_pixel_sum} pixels."
        # )
        # Return the best slice found
        if best_slice_index != -1:
            return mask_volume[:, :, best_slice_index]
        else:
            # Return an empty slice or handle the case where all slices are empty
            return np.zeros_like(mask_volume[:, :, 0])

    def __len__(self):
        if self.by_patient:
            # If by patient, return the number of unique patients
            return len(self.masks)
        else:
            return len(self.slices)

    # @lru_cache(maxsize=30)
    # def _get_volume(self, path):
    #     return nib.load(path).get_fdata(dtype=np.float32)

    def __getitem__(self, idx):
        if not self.by_patient:
            # If not by patient, we can directly access the slice
            # print(f"Getting item {idx} for organ {self.organ_name}")
            slice_info = self.slices[idx]
            # print(f"Slice id {slice_info['slice_idx']} for organ {self.organ_name}")
            volume_path = slice_info["img_path"]
            mask_path = slice_info["mask_path"]

            # Load 3D volume
            volume = self.volumes[volume_path]
            mask = self.masks[mask_path]

            # Get the specific slice
            vol_array = volume[:, :, slice_info["slice_idx"]].astype(
                np.float32
            )  # (H, W)
            mask_array = mask[:, :, slice_info["slice_idx"]].astype(
                np.float32
            )  # (H, W)

            # Normalize image to [0, 1]
            vol_slice = clip_and_rescale(vol_array)
            mask_slice = mask_array
            # Create binary mask (organ vs background)0
            # if self.binary:
            #     key_to_organ, organ_to_key = organ_mapper()
            #     organ_key = organ_to_key[self.organ_name.lower()]
            #     mask_slice = (mask_slice == organ_key).astype(np.float32)

            # Convert to torch tensors
            vol_tensor, mask_tensor = to_tensor(vol_slice, mask_slice)

            return vol_tensor, mask_tensor

        else:
            # Access to volumes and masks by idx and return 3d as tensors
            mask = list(self.masks.values())[idx]
            volume = list(self.volumes.values())[idx]
            volume = clip_and_rescale(volume)
            vol_tensor, mask_tensor = to_tensor(volume, mask)

            return vol_tensor, mask_tensor


def eval_model_by_patient(model, val_loader, device, all_patient_metrics):

    with torch.no_grad():
        for volume, mask in tqdm(val_loader, desc="Evaluating by patient"):
            predicted_slices = []
            # print(
            #     f"Processing patient with volume shape: {volume.shape} and mask shape: {mask.shape}"
            # )
            mask_3d = mask.squeeze()  # Remove batch dimension if present
            # print(f"Volume shape: {volume.shape}, Mask shape: {mask_3d.shape}")
            for i in range(volume.shape[4]):
                vol_slice = volume[:, :, :, :, i].to(device)

                output = model(vol_slice)
                probs = torch.sigmoid(output)
                preds_2d = (probs > 0.5).float()

                predicted_slices.append(preds_2d.squeeze().cpu())

            # Stack the 2D predictions to form a 3D mask
            predicted_volume = torch.stack(predicted_slices, dim=2)
            ground_truth_volume = mask_3d.cpu()
            # Calculate metrics
            dice_3d = dice_coefficient(predicted_volume, ground_truth_volume)

            y_true_flat = ground_truth_volume.flatten().numpy()
            y_pred_flat = predicted_volume.flatten().numpy()
            precision_3d = precision_score(y_true_flat, y_pred_flat, zero_division=0)
            recall_3d = recall_score(y_true_flat, y_pred_flat, zero_division=0)
            f1_3d = f1_score(y_true_flat, y_pred_flat, zero_division=0)

            # Append this patient's metrics to our lists
            all_patient_metrics["dice"].append(dice_3d.item())
            all_patient_metrics["precision"].append(precision_3d)
            all_patient_metrics["recall"].append(recall_3d)
            all_patient_metrics["f1"].append(f1_3d)

    # Calculate the average metrics across all patients
    avg_metrics = {key: np.mean(values) for key, values in all_patient_metrics.items()}

    print("\n--- Final 3D Evaluation Metrics (Averaged over all patients) ---")
    for key, value in avg_metrics.items():
        print(f"Average {key.capitalize()}: {value:.4f}")

    return avg_metrics, all_patient_metrics


# Training function
def train_model(
    model,
    train_loader,
    val_loader,
    patient_val_loader,
    organ_name,
    criterion,
    optimizer,
    by_patient=False,
    num_epochs=25,
    patience=5,
    min_delta=0.01,
    device="cuda",
    output_dir="./results",
):
    """Train the UNet model for semantic segmentation"""

    # Initialize metrics tracking
    best_val_loss = float("inf")
    min_delta = min_delta  # Minimum change in loss to qualify as an improvement
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice_scores": [],
        "val_dice_scores": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }
    all_patient_metrics = {
        "dice": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    phases = ["train", "val"]  # if not by_patient else ["train"]

    total_start_time = time.time()
    print("-" * 50)
    print(f"Starting model training on {organ_name}...")
    print(f"Size of training set: {train_loader.dataset.__len__()}")
    print(f"Size of validation set: {val_loader.dataset.__len__()}")

    print(f"Early stopping enabled with patience of {patience} epochs.")
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has training and validation phase
        for phase in phases:
            if phase == "train":
                model.train()
                dataloader = train_loader
            elif phase == "val":
                model.eval()
                dataloader = val_loader
                all_val_preds = []
                all_val_masks = []

            running_loss = 0.0
            running_dice = 0.0

            # if phase == "val_patient":
            #     model.eval()
            #     dataloader = val_loader
            #     print(f"Evaluating by patient for {organ_name}...")
            #     metrics, all_patient_metrics = eval_model_by_patient(
            #         model, dataloader, device, all_patient_metrics
            #     )

            #     metrics_json = {
            #         key: (
            #             [val.item() for val in value]
            #             if isinstance(value[0], torch.Tensor)
            #             else value
            #         )
            #         for key, value in all_patient_metrics.items()
            #     }

            #     with open(
            #         output_dir / f"eval_by_patient_metrics_{organ_name}.json", "w"
            #     ) as f:
            #         json.dump(metrics_json, f, indent=4)
            # Iterate over data

            for volumes, masks in tqdm(dataloader, desc=f"{phase.capitalize()} phase"):
                volumes = volumes.to(device)
                masks = masks.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(volumes)
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
                running_loss += loss.item() * volumes.size(0)
                running_dice += dice * volumes.size(0)

                if phase == "val":
                    all_val_preds.append(preds.view(-1).cpu())
                    all_val_masks.append(masks.view(-1).cpu())

            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}")

            # Save metrics
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_dice_scores"].append(epoch_dice.cpu())
            elif phase == "val":
                history["val_loss"].append(epoch_loss)
                history["val_dice_scores"].append(epoch_dice.cpu())
                all_preds_tensor = torch.cat(all_val_preds)
                all_masks_tensor = torch.cat(all_val_masks)

                # Convert to NumPy arrays for scikit-learn
                y_true = all_masks_tensor.numpy()
                y_pred = all_preds_tensor.numpy()

                # Calculate precision, recall, and F1 score
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # Store metricin history
                history["val_precision"].append(precision)
                history["val_recall"].append(recall)
                history["val_f1"].append(f1)

                print(
                    f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )
            # Save progress plots
            print(f"\nSaving progress plots at epoch {epoch + 1}...")
            save_progress_plots(history, epoch, num_epochs, organ_name, output_dir)

        # Save best model
        if epoch_loss < best_val_loss - min_delta:
            best_val_loss = epoch_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), output_dir / f"best_unet_{organ_name}.pth")
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}"
            )
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    if by_patient:
        model.eval()
        dataloader = patient_val_loader
        print(f"Evaluating by patient for {organ_name}...")
        metrics, all_patient_metrics = eval_model_by_patient(
            model, dataloader, device, all_patient_metrics
        )

        metrics_json = {
            key: (
                [val.item() for val in value]
                if isinstance(value[0], torch.Tensor)
                else value
            )
            for key, value in all_patient_metrics.items()
        }

        with open(output_dir / f"eval_by_patient_metrics_{organ_name}.json", "w") as f:
            json.dump(metrics_json, f, indent=4)

    total_end_time = time.time()
    print("Training finished.")
    total_duration_seconds = total_end_time - total_start_time
    # Format it into hours, minutes, and seconds for readability
    hours, rem = divmod(total_duration_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Training Time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")

    # Save final model
    torch.save(model.state_dict(), output_dir / f"final_unet_{organ_name}.pth")
    return history, all_patient_metrics, total_duration_seconds


# Dice coefficient for evaluation
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """Calculate Dice coefficient"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def train_loop(
    organ_list,
    data_path,
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda",
    by_patient=True,
    organ_threshold=0.05,
    min_delta=0.01,
):
    """
    Main training loop for the UNet model.
    Args:
        organ_list: List of organs to train on
        data_path: Path to the dataset
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        device: Device to use for training (cuda or cpu)
        by_patient: Whether to evaluate by patient or not
        organ_threshold: Threshold for organ presence in slices
    """
    # Get organ key from organ name
    key_to_organ, organ_to_key = organ_mapper()

    # Load CSVs
    train_set = pd.read_csv(os.path.join(data_path, "split", "train_dataset.csv"))
    val_set = pd.read_csv(os.path.join(data_path, "split", "test_dataset.csv"))
    runs_dir = Path("./results") / f"runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Train the model for each organ
    for organ_name in organ_list:
        output_dir = Path(runs_dir) / f"{organ_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if organ_name.lower() not in organ_to_key:
            print(f"Organ '{organ_name}' not found in the dataset. Skipping...")
            continue

        # Create Dataloaders
        train_dataset = BTCVSliceDataset(
            train_set, organ_name, organ_threshold=organ_threshold, binary=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_dataset = BTCVSliceDataset(
            val_set, organ_name, by_patient=False, binary=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        if by_patient:
            # If by_patient, we need to create a DataLoader that groups by patient
            patient_val_dataset = BTCVSliceDataset(
                val_set, organ_name, by_patient=True, binary=True
            )
            patient_val_loader = DataLoader(
                patient_val_dataset, batch_size=1, shuffle=False, num_workers=4
            )

        # Initialize model
        model = UNet(in_channels=1, out_channels=1).to(device)

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        history, all_patient_metrics, total_duration_seconds = train_model(
            model,
            train_loader,
            val_loader,
            patient_val_loader,
            organ_name,
            criterion,
            optimizer,
            by_patient=by_patient,
            num_epochs=num_epochs,
            min_delta=min_delta,
            device=device,
            output_dir=output_dir,
        )

        print("Saving training history to JSON file...")
        # On doit convertir les tenseurs CPU en simples nombres pour le JSON
        history_for_json = {
            key: (
                [val.item() for val in value]
                if value and isinstance(value[0], torch.Tensor)
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
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(
            all_patient_metrics["dice"],
            label="Dice Score",
            color="blue",
            marker="o",
        )
        plt.plot(
            all_patient_metrics["precision"],
            label="Precision",
            color="green",
            marker="o",
        )
        plt.plot(
            all_patient_metrics["recall"],
            label="Recall",
            color="orange",
            marker="o",
        )
        plt.plot(
            all_patient_metrics["f1"],
            label="F1 Score",
            color="red",
            marker="o",
        )
        plt.xlabel("Patient Index")
        plt.ylabel("Score")
        plt.legend()

        # Adjust layout to prevent the main title from overlapping with subplots
        plt.title(f"Patient-wise Evaluation Metrics for {organ_name}")
        plt.show()
        plt.savefig(output_dir / f"Metrics_per_patient_{organ_name}.png")
        print(f"Training completed for {organ_name}. Results saved in {output_dir}.")


# Main execution
if __name__ == "__main__":
    # Parameters
    organ_list = [
        "spleen",
        "liver",
        "right kidney",
        "left kidney",
    ]
    data_path = "/home/ismail/projet_PFE/Hands-on-nnUNet/nnUNetFrame/dataset/RawData"
    organ_threshold = [0.05, 0.1, 0.15]  # Threshold for organ presence in slices

    # Run training loop
    for threshold in organ_threshold:
        print(f"Training with organ threshold: {threshold}")
        train_loop(
            organ_list,
            data_path,
            organ_threshold=threshold,
            min_delta=0.01,
        )
