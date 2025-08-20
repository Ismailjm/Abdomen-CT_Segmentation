# Consolidated imports (deduplicated)
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

import SimpleITK as sitk
import nibabel as nib

# Add Unet directory to path and import model
sys.path.append("../Unet")
from unet import UNet  # type: ignore


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to {seed_value}")


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


class VolumeCache:
    """
    A simple cache to load and hold all 3D image volumes in memory once.
    """

    def __init__(self, dataset_df):
        self.volumes = {}
        self._load_volumes(dataset_df)

    def _load_volumes(self, dataset_df):
        unique_volumes_paths = dataset_df["img_path"].unique()
        for path in tqdm(
            unique_volumes_paths, desc="Loading all image volumes into cache"
        ):
            if path not in self.volumes:
                self.volumes[path] = nib.load(path).get_fdata(dtype=np.float32)

    def get_volume(self, path):
        return self.volumes.get(path)


# Custom dataset for BTCV slices
class BTCVSliceDataset(Dataset):
    def __init__(
        self,
        dataset,
        organ_name,
        volume_cache,
        by_patient=False,
        organ_threshold=0,
        binary=True,
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

        self.volumes = (
            volume_cache.volumes
        )  # volumes dictionary points to the data in the cache
        self.masks = {}
        _, organ_to_key = organ_mapper()
        self.organ_key = organ_to_key[organ_name.lower()]

        # Get unique volmues/masks paths
        # unique_volumes_paths = dataset["img_path"].unique()
        # for path in tqdm(unique_volumes_paths, desc="Loading volumes"):
        #     self.volumes[path] = nib.load(path).get_fdata(dtype=np.float32)

        unique_masks_paths = dataset["mask_path"].unique()
        # Load masks and create binary masks for the specified organ
        if self.binary:
            for path in tqdm(unique_masks_paths, desc="Loading masks"):
                self.masks[path] = (
                    (nib.load(path).get_fdata(dtype=np.float32)) == self.organ_key
                ).astype(np.float32)

        if by_patient is False:
            total_slices_found = 0
            slices_kept = 0
            self.slices = []
            total_organ_pixels_in_kept_slices = 0
            for _, item in dataset.iterrows():
                if item["organ"].lower() == organ_name.lower():
                    masks_data = self.masks[item["mask_path"]]
                    max_mask_slice = self._extract_max_slice_simple(item)
                    # middle_slice_mask_array = masks_data[:, :, (item["begin_slice"]+item["end_slice"]) // 2]
                    for slice_idx in range(item["begin_slice"], item["end_slice"] + 1):
                        total_slices_found += 1

                        mask_array = masks_data[:, :, slice_idx]
                        # print(mask_array.shape)

                        organ_perc = mask_array.sum() / max_mask_slice.sum()

                        # Only add the slice if it meets the threshold
                        if organ_perc >= self.organ_threshold:
                            slices_kept += 1
                            total_organ_pixels_in_kept_slices += mask_array.sum()
                            self.slices.append(
                                {
                                    "img_path": item["img_path"],
                                    "mask_path": item["mask_path"],
                                    "slice_idx": slice_idx,
                                }
                            )
            if slices_kept > 0:
                total_pixels_in_kept_slices = slices_kept * 512 * 512
                # print(
                #     f"Total organ pixels in kept slices: {total_organ_pixels_in_kept_slices}"
                # )
                # print(f"Total pixels in kept slices: {total_pixels_in_kept_slices}")
                self.avg_white_percent = (
                    total_organ_pixels_in_kept_slices / total_pixels_in_kept_slices
                )

            else:
                self.avg_white_percent = 0
            print(
                f"Average white percentage for {self.organ_name}: {self.avg_white_percent:.4f}"
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

    def sample_k_from_dataset(self, k_shots, seed=None, batch_size=None):
        if seed is not None:
            random.seed(seed)
        slices_indices = random.sample(range(len(self)), k_shots)
        subset = [self.slices[i] for i in slices_indices]
        vol_tensors = []
        mask_tensors = []
        for slice_info in subset:
            volume_path = slice_info["img_path"]
            mask_path = slice_info["mask_path"]

            # Load 3D volume
            volume = self.volumes[volume_path]
            mask = self.masks[mask_path]

            # Get the specific slice
            vol_array = volume[:, :, slice_info["slice_idx"]].astype(np.float32)
            vol_array = clip_and_rescale(vol_array)
            mask_array = mask[:, :, slice_info["slice_idx"]].astype(np.float32)
            # Create binary mask (organ vs background)
            if self.binary:
                key_to_organ, organ_to_key = organ_mapper()
                organ_key = organ_to_key[self.organ_name.lower()]
                mask_array = (mask_array == organ_key).astype(np.float32)
            # Convert to torch tensors
            vol_tensor, mask_tensor = to_tensor(vol_array, mask_array)
            vol_tensors.append(vol_tensor)
            mask_tensors.append(mask_tensor)
        vol_tensors = torch.stack(vol_tensors)
        mask_tensors = torch.stack(mask_tensors)
        return vol_tensors, mask_tensors


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to {seed_value}")


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """Calculate Dice coefficient"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def forward_pass(model, data_loader, criterion, phase, device):
    total_batch_loss = 0.0
    total_batch_dice = 0.0
    num_batches = 0

    # Training Step
    if phase == "train":
        model.train()
    else:
        model.eval()
        total_batch_precision = 0.0
        total_batch_recall = 0.0

    with torch.set_grad_enabled(phase == "train"):
        for batch_images, batch_masks in data_loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            predictions = model(batch_images)
            if criterion == "focal_loss":
                train_loss = sigmoid_focal_loss(
                    predictions,
                    batch_masks,
                    reduction="mean",
                )
            elif criterion == "BCE":
                loss_fn = nn.BCEWithLogitsLoss()
                train_loss = loss_fn(predictions, batch_masks)
            else:
                raise ValueError("Unsupported criterion. Use 'focal_loss' or 'BCE'.")
            if phase == "train":
                total_batch_loss += train_loss
            else:
                total_batch_loss += train_loss.item()
            num_batches += 1
            dice_score = dice_coefficient(torch.sigmoid(predictions), batch_masks)
            total_batch_dice += dice_score.item()
            if phase != "train":
                probs = torch.sigmoid(predictions)
                binary_preds = (probs > 0.5).float()

                # Flatten the tensors for metric calculations
                y_true = batch_masks.cpu().numpy().flatten()
                y_pred = binary_preds.cpu().numpy().flatten()

                precision = precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,  # Handle zero division
                )
                recall = recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )

                total_batch_precision += precision
                total_batch_recall += recall

        if phase == "train":
            return total_batch_loss / num_batches, total_batch_dice / num_batches
        else:
            return (
                total_batch_loss / num_batches,
                total_batch_dice / num_batches,
                total_batch_precision / num_batches,
                total_batch_recall / num_batches,
            )


def test_model(
    train_dataset,
    val_dataloader,
    test_dataloader,
    args,
    output_dir,
    state_dict,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Test MAML model on the BTCV dataset.
    Args:
        dataset: Instance of BTCVDataset
        task: Organ to validate on
        output_dir: Directory to save validation results
        state_dict: Checkpoint state dictionary containing model weights
        criterion: Loss function for validation
        num_epochs: Number of validation epochs
        k_shots: Number of support examples per task
        k_queries: Number of query examples per task
        lr: Learning rate
        device: Device to run the validation on (CPU or GPU)
        checkpoint_interval: Interval to save model checkpoints
    """

    output_dir = Path(output_dir)
    model = UNet(in_channels=args.in_channels, out_channels=1).to(device)
    model_weights = torch.load(state_dict, map_location=device)
    if "model_state_dict" in model_weights:
        model.load_state_dict(model_weights["model_state_dict"])
    else:
        model.load_state_dict(model_weights)

    checkpoint_epoch = model_weights["epoch"] + 1
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    losses = {
        "train": [],
        "val": [],
        "test": [],
    }
    dice_scores = {"train": [], "val": [], "test": []}
    metrics = defaultdict(lambda: {"precision": [], "recall": []})

    train_images, train_masks = train_dataset.sample_k_from_dataset(
        k_shots=args.k_shots,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    # # Move data to device
    train_images = train_images.to(device)
    train_masks = train_masks.to(device)
    for epoch in tqdm(range(args.num_epochs), desc="Meta-Test Progress"):

        model.train()
        optimizer.zero_grad()
        predictions = model(train_images)
        if args.criterion == "focal_loss":
            train_loss = sigmoid_focal_loss(
                predictions,
                train_masks,
                reduction="mean",
            )
        elif args.criterion == "BCE":
            loss_fn = nn.BCEWithLogitsLoss()
            train_loss = loss_fn(predictions, train_masks)
        else:
            raise ValueError("Unsupported criterion. Use 'focal_loss' or 'BCE'.")

        dice_score = dice_coefficient(torch.sigmoid(predictions), train_masks)
        losses["train"].append(train_loss.item())
        dice_scores["train"].append(dice_score.item())
        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Dice: {dice_score.item():.4f}"
        )
        # Backward pass and optimization
        model.zero_grad()
        train_loss.backward()
        optimizer.step()

        ## Validation Loop
        avg_val_loss, avg_val_dice, avg_val_precision, avg_val_recall = forward_pass(
            model,
            val_dataloader,
            args.criterion,
            phase="val",
            device=device,
        )
        losses["val"].append(avg_val_loss)
        dice_scores["val"].append(avg_val_dice)
        metrics["val"]["precision"].append(avg_val_precision)
        metrics["val"]["recall"].append(avg_val_recall)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )
        print("= " * 50)

        # Save validation results

        if (args.checkpoint_interval is not None) and (
            (epoch + 1) % args.checkpoint_interval == 0
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_dir / f"val_model_at_epoch{epoch+1}.pth",
            )

        # plot validation and dice loss
        plt.figure(figsize=(12, 5))
        plt.suptitle(
            f"Validation History for {args.task}, Epochs : {args.num_epochs}",
        )
        plt.subplot(1, 2, 1)
        plt.plot(losses["train"], label="Train Loss", color="blue")
        plt.plot(losses["val"], label="Val Loss", color="green")
        plt.plot(dice_scores["train"], label="Train Dice", color="blue", linestyle="--")
        plt.plot(dice_scores["val"], label="Val Dice", color="green", linestyle="--")
        plt.title("Losses and Dice Scores")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        # plt.ylim(0, 1)  # Set y-axis range
        plt.legend()

        # metrics plots
        plt.subplot(1, 2, 2)
        plt.plot(metrics["val"]["precision"], label="Precision", color="yellow")
        plt.plot(metrics["val"]["recall"], label="Recall", color="red")
        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        # plt.ylim(0, 1)  # Set y-axis range from 0 to 1
        plt.legend()
        plt.savefig(output_dir / f"history_plot.png")
        plt.close()

    # Save the final model
    torch.save(
        {
            "epoch": args.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        },
        output_dir / "final_val_model.pth",
    )

    # Test the model on the test set
    avg_test_loss, avg_test_dice, avg_test_precision, avg_test_recall = forward_pass(
        model,
        test_dataloader,
        args.criterion,
        phase="test",
        device=device,
    )
    losses["test"].append(avg_test_loss)
    dice_scores["test"].append(avg_test_dice)
    metrics["test"]["precision"].append(avg_test_precision)
    metrics["test"]["recall"].append(avg_test_recall)

    print(
        f"Epoch {epoch + 1}/{args.num_epochs}, Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}"
    )
    print("= " * 50)

    # Save the history of validation and dice loss
    with open(output_dir / "losses_and_metrics.json", "w") as f:
        json.dump(
            {
                "support": {
                    "loss": losses["train"],
                    "dice_score": dice_scores["train"],
                },
                "query": {
                    "val": {
                        "loss": losses["val"],
                        "dice_score": dice_scores["val"],
                        "metrics": metrics["val"],
                    },
                    "test": {
                        "loss": losses["test"],
                        "dice_score": dice_scores["test"],
                        "metrics": metrics["test"],
                    },
                },
            },
            f,
            indent=4,
        )

    # Save the validation settings for classic model
    with open(output_dir / "validation_settings.json", "w") as f:
        json.dump(
            {
                "task": args.task,
                "dataset size": {
                    "train": len(train_dataset),
                    "val": len(val_dataloader.dataset),
                    "test": len(test_dataloader.dataset),
                },
                "epochs": args.num_epochs,
                "pretrained_model_path": state_dict,
                "pretrained_model_epoch": checkpoint_epoch,
                "k_shots": args.k_shots,
                "k_queries": args.k_queries,
                "lr": args.lr,
                "loss_function": args.criterion,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
            },
            f,
            indent=4,
        )


def test_loop(args, device):
    """
    Main loop for testing the model.
    Args:
        args: Parsed command line arguments.
        device: Device to run the model on (CPU or GPU).
    """
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
    # Load the dataset
    directory = args.model_path
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_runs_dir = (
        Path(directory)
        / f"test/runs_k-shots-{args.k_shots}_epochs-{args.num_epochs}_lr-{args.lr}_seed-{args.seed}_batch_size-{args.batch_size}_{current_time}"
    )
    dataset = pd.read_csv(os.path.join(args.data_path, "split", "test_dataset.csv"))

    unique_patients = list(dataset["patient_id"].unique())
    random.shuffle(unique_patients)

    train_set = dataset[dataset["patient_id"] == unique_patients[0]]
    val_set = dataset[dataset["patient_id"] == unique_patients[1]]
    test_set = dataset[dataset["patient_id"] == unique_patients[2]]

    train_volume_cache = VolumeCache(train_set)
    val_volume_cache = VolumeCache(val_set)
    test_volume_cache = VolumeCache(test_set)

    train_dataset = BTCVSliceDataset(
        train_set,
        args.task,
        train_volume_cache,
        organ_threshold=args.threshold,
        binary=True,
    )

    val_dataset = BTCVSliceDataset(
        val_set,
        args.task,
        val_volume_cache,
        organ_threshold=args.threshold,
        by_patient=False,
        binary=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_dataset = BTCVSliceDataset(
        test_set,
        args.task,
        test_volume_cache,
        organ_threshold=args.threshold,
        by_patient=False,
        binary=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    for subfolder in os.listdir(directory):
        if args.task == subfolder:
            continue
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            best_model_path = os.path.join(subfolder_path, f"best_unet_{subfolder}.pth")
            final_model_path = os.path.join(
                subfolder_path, f"final_unet_{subfolder}.pth"
            )

        for file in [best_model_path, final_model_path]:

            checkpoint_type = os.path.basename(file).split("_")[0]
            runs_dir = root_runs_dir / subfolder / f"checkpoint_type-{checkpoint_type}"
            runs_dir.mkdir(parents=True, exist_ok=True)

            # Create dataset and task list

            print("==" * 50)
            print(
                f"Testing model at {file} on task {args.task} with test size {args.test_size}"
            )
            test_model(
                train_dataset,
                val_loader,
                test_loader,
                args,
                in_channels=args.in_channels,
                output_dir=runs_dir,
                state_dict=file,
                device=device,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meta-test MAML model on BTCV dataset."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/ieljamiy/RawData",
        help="Path to the meta test CSV file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["maml", "fomaml", "classic"],
        help="Type of model to use for meta-testing.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to validate on.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pretrained model weights.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for the BTCV dataset.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of validation epochs.",
    )
    parser.add_argument(
        "--k_shots",
        type=int,
        default=1,
        help="Number of support examples per task.",
    )
    parser.add_argument(
        "--k_queries",
        type=int,
        default=-1,
        help="Number of query examples per task (-1 means all available queries).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help="Number of input channels for the model (e.g., 3 for RGB images).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="BCE",
        choices=["focal_loss", "BCE"],
        help="Loss function to use for training and validation.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.4,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=None,
        help="Interval for saving model checkpoints during validation.",
    )
    args = parser.parse_args()

    # Define parameters

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pretrained_model_path = "/home/ieljamiy/code/experiments/meta/shapes/fomaml/train/runs_k-shots_5_epochs_500_episodes_5_FOMAML_BCE_42/best_model.pth"  # Path to best model weights

    test_loop(args, device=device)
