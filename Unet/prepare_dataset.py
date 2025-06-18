import nibabel as nib

data_path = "/home/ismail/projet_PFE/Hands-on-nnUNet/nnUNetFrame/dataset/RawData"

import os
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import defaultdict


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
    images_dir = os.path.join(data_root, "imagesTr")
    masks_dir = os.path.join(data_root, "labelsTr")
    imgs = [images_dir + "/" + f for f in sorted(os.listdir(images_dir))]
    masks = [masks_dir + "/" + f for f in sorted(os.listdir(masks_dir))]

    for img_path, mask_path in zip(imgs, masks):
        mask_nifti = nib.load(mask_path)
        mask_data = mask_nifti.get_fdata()
        organs = [int(organ) for organ in np.unique(mask_data).tolist()]
        annotations.append(
            {"img_path": img_path, "mask_path": mask_path, "organs": organs}
        )

    annotations_df = pd.DataFrame(annotations)
    # Save to CSV
    annotations_df.to_csv(os.path.join(data_root, "dataset.csv"), index=False, sep=",")

    return annotations