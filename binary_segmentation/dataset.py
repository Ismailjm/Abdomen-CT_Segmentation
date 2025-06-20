import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import os
import numpy as np
import random
from pathlib import Path


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


def transform_to_csv(data_root):
    """
    Transform the BTCV dataset to a csv file.
    Args:
        data_root (str): path to the BTCV dataset.

    Returns:
        annotations (list): list of dictionaries containing the images and their masks with a list of organs available in each scan.

    """

    annotations = []
    # images_dir = os.path.join(data_root, "imagesTr")
    # masks_dir = os.path.join(data_root, "labelsTr")
    # imgs = [images_dir + "/" + f for f in sorted(os.listdir(images_dir))]
    # masks = [masks_dir + "/" + f for f in sorted(os.listdir(masks_dir))]
    img_dir = os.path.join(data_path, "imagesTr")
    masks_dir = os.path.join(data_path, "labelsTr")
    imgs = Path(img_dir).glob("*.nii.gz")
    masks = Path(masks_dir).glob("*.nii.gz")

    for img_path, mask_path in zip(sorted(imgs), sorted(masks)):
        mask_nifti = nib.load(mask_path)
        mask_data = mask_nifti.get_fdata()
        organs = [int(organ) for organ in np.unique(mask_data).tolist()]
        annotations.append(
            {"img_path": str(img_path), "mask_path": str(mask_path), "organs": organs}
        )

    annotations_df = pd.DataFrame(annotations)
    # Save to CSV
    annotations_df.to_csv(os.path.join(data_root, "dataset.csv"), index=False, sep=",")

    return annotations


def extract_slices(in_set, organs):
    out_set = []
    key_to_organ, organ_to_key = organ_mapper()
    for row in in_set:
        for organ in organs:
            organ_key = organ_to_key.get(organ)
            if organ_key is None:
                print(f"Organ {organ} not found in organ mapping. Skipping.")
                continue

            img_path = row["img_path"]
            mask_path = row["mask_path"]
            subject_organ = row["organs"]
            if organ_key not in subject_organ:
                continue

            mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask)  # Shape: (Depth, Height, Width)

            slices_with_organ = np.any(mask_arr == organ_key, axis=(1, 2))
            slices_indices = np.where(slices_with_organ)[0].tolist()
            out_set.append(
                {
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "organ": organ,
                    "begin_slice": slices_indices[0],
                    "end_slice": slices_indices[-1],
                }
            )
    return out_set


def create_dataset(annotations, organs, train_size, seed=42):
    """
    Load a specific number of slices from the BTCV dataset.
    Args:
        n_slices (int): Number of slices to load.

    Returns:
        slices_df (list): List of loaded slices.
    """
    random.seed(seed)
    # split the dataset to train and test sets
    annotations_random = random.sample(annotations, len(annotations))
    train_df = random.sample(annotations_random, train_size)
    test_df = [ann for ann in annotations if ann not in train_df]

    # create train & test flattened sets
    train_set = extract_slices(train_df, organs)
    test_set = extract_slices(test_df, organs)

    if not os.path.exists(os.path.join(data_path, "split")):
        os.makedirs(os.path.join(data_path, "split"))

    train_df = pd.DataFrame(train_set)
    train_df.to_csv(
        os.path.join(data_path, "split", "train_dataset.csv"), index=False, sep=","
    )
    test_df = pd.DataFrame(test_set)
    test_df.to_csv(
        os.path.join(data_path, "split", "test_dataset.csv"), index=False, sep=","
    )

    return train_set, test_set


if __name__ == "__main__":

    data_path = "/home/ismail/projet_PFE/Hands-on-nnUNet/nnUNetFrame/dataset/RawData"
    organs_list = []

    annnotations = transform_to_csv(data_path)
    train_set, test_set = create_dataset(
        annnotations, organs_list, train_size=18, seed=42
    )
