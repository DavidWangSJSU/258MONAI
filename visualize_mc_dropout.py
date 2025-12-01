# visualize_mc_dropout.py
#
# Visualize CT slice, mean prediction, and uncertainty (variance)
# from mc_outputs/val_case0_mc_dropout.pt

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from monai.data import MetaTensor
import torch.serialization as serialization

# allow MONAI MetaTensor to be unpickled safely
serialization.add_safe_globals([MetaTensor])

OUT_FILE = "mc_outputs/val_case0_mc_dropout.pt"


def main():
    if not os.path.exists(OUT_FILE):
        raise FileNotFoundError(
            f"{OUT_FILE} not found. Run infer_mc_dropout.py first to create it."
        )

    # In PyTorch 2.6+, weights_only defaults to True.
    # We are loading a dict of tensors, not pure weights, so set weights_only=False.
    data = torch.load(OUT_FILE, map_location="cpu", weights_only=False)

    mean_pred = data["mean_pred"]  # [B, 1, H, W, D]
    var_pred = data["var_pred"]    # [B, 1, H, W, D]
    label = data["label"]          # [B, 1, H, W, D]

    # convert to numpy and squeeze batch/channel dims -> [H, W, D]
    mean_np = mean_pred.squeeze().numpy()
    var_np = var_pred.squeeze().numpy()
    label_np = label.squeeze().numpy()

    # choose a slice index roughly in the middle
    depth = mean_np.shape[-1]
    slice_idx = depth // 2

    mean_slice = mean_np[:, :, slice_idx]
    var_slice = var_np[:, :, slice_idx]
    label_slice = label_np[:, :, slice_idx]

    # basic plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) Mean prediction (probabilities)
    im0 = axes[0].imshow(mean_slice, cmap="gray")
    axes[0].set_title(f"Mean prediction (slice {slice_idx})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Ground-truth mask
    im1 = axes[1].imshow(label_slice, cmap="gray")
    axes[1].set_title("Ground truth mask")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Uncertainty (variance)
    im2 = axes[2].imshow(var_slice, cmap="hot")
    axes[2].set_title("Uncertainty (variance)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # optionally save to PNG for your report
    os.makedirs("figures", exist_ok=True)
    out_png = os.path.join("figures", f"case0_slice{slice_idx}.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
