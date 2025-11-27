# dataset.py

import os
import glob
from typing import Tuple, Dict, Any, List

import torch
from torch.utils.data import DataLoader
from monai.data import CacheDataset, Dataset, partition_dataset
from monai.utils import set_determinism
from monai.transforms import Compose

from transforms import get_train_transforms, get_val_transforms


def _get_spleen_file_lists(
    root_dir: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Build lists of dicts for MONAI Dataset from MSD Task09_Spleen.

    Expected layout (what you already have):
        root_dir/
          imagesTr/
          labelsTr/

    Returns:
        train_files, val_files
    """
    images_tr = sorted(glob.glob(os.path.join(root_dir, "imagesTr", "*.nii.gz")))
    labels_tr = sorted(glob.glob(os.path.join(root_dir, "labelsTr", "*.nii.gz")))

    if len(images_tr) == 0 or len(labels_tr) == 0:
        raise RuntimeError(
            f"No training images/labels found in {root_dir}. "
            f"Check that Task09_Spleen/imagesTr and labelsTr exist."
        )

    if len(images_tr) != len(labels_tr):
        raise RuntimeError(
            f"Number of images ({len(images_tr)}) and labels ({len(labels_tr)}) "
            "do not match."
        )

    data_dicts = [
        {"image": img, "label": seg}
        for img, seg in zip(images_tr, labels_tr)
    ]

    # Simple train/val split: 80% train, 20% val
    train_files, val_files = partition_dataset(
        data=data_dicts,
        ratios=[0.8, 0.2],
        shuffle=True,
        seed=0,
    )

    return train_files, val_files


def get_dataloaders(
    msd_root: str = "data/MSD/Task09_Spleen",
    batch_size: int = 2,
    num_workers: int = 4,
    use_cache: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MONAI train and validation DataLoaders for Task09_Spleen.

    Args:
        msd_root: path to Task09_Spleen (folder containing imagesTr, labelsTr)
        batch_size: batch size for loaders
        num_workers: DataLoader workers
        use_cache: if True, use CacheDataset, else standard Dataset

    Returns:
        train_loader, val_loader
    """
    set_determinism(seed=0)

    train_files, val_files = _get_spleen_file_lists(msd_root)

    train_transforms: Compose = get_train_transforms(keys=("image", "label"))
    val_transforms: Compose = get_val_transforms(keys=("image", "label"))

    DatasetClass = CacheDataset if use_cache else Dataset

    train_ds = DatasetClass(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0 if use_cache else 0.0,
        num_workers=num_workers if use_cache else 0,
    )
    val_ds = DatasetClass(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0 if use_cache else 0.0,
        num_workers=num_workers if use_cache else 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader = get_dataloaders()

    batch = next(iter(train_loader))
    image = batch["image"]  # [B, C, H, W, D]
    label = batch["label"]

    print("Train batch image shape:", image.shape)
    print("Train batch label shape:", label.shape)
