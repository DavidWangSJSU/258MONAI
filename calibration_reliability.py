# calibration_reliability.py
#
# Compute calibration (ECE) and plot a reliability diagram
# for the baseline UNet on the validation set.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from dataset import get_dataloaders


def _extract_batch(batch_data, device):
    """
    Same helper as in train_baseline.py / infer_mc_dropout.py.
    """
    if isinstance(batch_data, dict):
        data = batch_data
    elif isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 and isinstance(
        batch_data[0], dict
    ):
        data = batch_data[0]
    else:
        raise TypeError(f"Unexpected batch_data type: {type(batch_data)}")

    inputs = data["image"].to(device)
    labels = data["label"].to(device)
    return inputs, labels


def compute_reliability(probs, labels, n_bins: int = 10):
    """
    Compute bin-wise accuracy and confidence plus ECE.
    probs, labels: 1D numpy arrays of same length.
    """
    assert probs.shape == labels.shape

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges, right=True) - 1  # 0..n_bins-1

    accuracies = []
    confidences = []
    counts = []

    ece = 0.0
    N = len(probs)

    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            accuracies.append(0.0)
            confidences.append(0.0)
            counts.append(0)
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask]

        conf = bin_probs.mean()
        acc = (bin_labels == 1).mean()  # voxel-wise accuracy in this bin
        count = mask.sum()

        accuracies.append(acc)
        confidences.append(conf)
        counts.append(count)

        ece += (count / N) * abs(acc - conf)

    return (
        np.array(accuracies),
        np.array(confidences),
        np.array(counts),
        ece,
        bin_edges,
    )


def build_model(device, dropout: float = 0.2):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=dropout,
    ).to(device)
    return model


def main():
    set_determinism(0)

    msd_root = "data/MSD/Task09_Spleen"
    model_dir = "models"
    best_model_path = os.path.join(model_dir, "best_metric_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. Run train_baseline.py first."
        )

    # Data: use batch_size=1 and full-volume inference
    print("Loading validation data...")
    _, val_loader = get_dataloaders(
        msd_root=msd_root,
        batch_size=1,
        num_workers=4,
        use_cache=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(device=device, dropout=0.2)
    # state_dict only, so weights_only=True is fine, but default is OK too
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    roi_size = (96, 96, 96)

    all_probs = []
    all_labels = []

    print("Running deterministic inference on validation set for calibration...")
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            inputs, labels = _extract_batch(batch_data, device)

            logits = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten
            probs_np = probs.cpu().numpy().ravel()
            labels_np = labels.cpu().numpy().ravel()

            all_probs.append(probs_np)
            all_labels.append(labels_np)

            print(f"  Processed case {idx}, {probs_np.size} voxels")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Total voxels: {all_probs.size}")

    # Compute reliability metrics
    acc, conf, counts, ece, bin_edges = compute_reliability(all_probs, all_labels, n_bins=10)

    print("Bin-wise stats:")
    for i in range(len(acc)):
        print(
            f"  Bin {i:2d}: edge [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}), "
            f"count={counts[i]}, conf={conf[i]:.3f}, acc={acc[i]:.3f}"
        )

    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    # Plot reliability diagram
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    plt.plot(bin_centers, acc, marker="o", label="Model calibration")

    plt.bar(
        bin_centers,
        acc,
        width=0.08,
        alpha=0.3,
        edgecolor="black",
        label="Accuracy per bin",
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    os.makedirs("figures", exist_ok=True)
    out_png = os.path.join("figures", "reliability_diagram.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved reliability diagram to {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
