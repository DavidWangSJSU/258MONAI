# transforms.py

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from monai.config import KeysCollection


def get_train_transforms(keys: KeysCollection = ("image", "label")):
    """
    Returns MONAI Compose transform for training.
    Assumes 3D CT spleen dataset in NIfTI format.
    """
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            # Resample to a common voxel spacing (from MONAI spleen tutorial)
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            # Standard orientation
            Orientationd(keys=keys, axcodes="RAS"),
            # Intensity normalization in a reasonable HU range for CT
            ScaleIntensityRanged(
                keys="image",
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Remove empty background around the spleen
            CropForegroundd(keys=keys, source_key="image"),
            # Random patch sampling with positive/negative ratio based on the label
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # Simple augmentations
            RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
            RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
            RandFlipd(keys=keys, spatial_axis=[2], prob=0.5),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            EnsureTyped(keys=keys),
        ]
    )


def get_val_transforms(keys: KeysCollection = ("image", "label")):
    """
    Returns MONAI Compose transform for validation / testing.
    No random augmentations.
    """
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=keys, axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=keys, source_key="image"),
            EnsureTyped(keys=keys),
        ]
    )
