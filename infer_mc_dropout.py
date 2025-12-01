# infer_mc_dropout.py
#
# Monte Carlo Dropout inference on the validation set
# to estimate voxel-wise uncertainty.

import os
import torch

from monai.networks.nets import UNet
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from dataset import get_dataloaders


def _extract_batch(batch_data, device):
    """
    Same helper as in train_baseline.py:
    handle both dict and list-of-dicts formats.
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


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> float:
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)
    return dice.item()


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


def mc_dropout_inference(
    model,
    val_loader,
    device,
    num_samples: int = 10,
    roi_size=(96, 96, 96),
    sw_batch_size: int = 1,
):
    """
    For each validation case, run multiple stochastic forward passes
    with dropout enabled and compute mean + variance predictions.

    Returns:
        list of dicts, each with:
            "mean_pred": tensor [B, 1, H, W, D]
            "var_pred":  tensor [B, 1, H, W, D]
            "label":     tensor [B, 1, H, W, D]
    """
    model.train()  # IMPORTANT: keep dropout active

    post_prob = Compose([Activations(sigmoid=True)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    results = []

    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            inputs, labels = _extract_batch(batch_data, device)

            mc_probs = []

            for s in range(num_samples):
                logits = sliding_window_inference(
                    inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )
                probs = post_prob(logits)  # [B,1,H,W,D]
                mc_probs.append(probs)

            # [num_samples, B, 1, H, W, D]
            mc_probs = torch.stack(mc_probs, dim=0)

            mean_pred = mc_probs.mean(dim=0)  # [B,1,H,W,D]
            var_pred = mc_probs.var(dim=0)    # [B,1,H,W,D]

            # Dice of mean prediction
            mean_bin = (mean_pred > 0.5).float()
            dice = dice_score(mean_bin, labels)

            # For info, also show average uncertainty
            mean_var = var_pred.mean().item()

            print(
                f"Case {idx}: Dice (mean prediction) = {dice:.4f}, "
                f"mean variance = {mean_var:.6f}"
            )

            results.append(
                {
                    "mean_pred": mean_pred.cpu(),
                    "var_pred": var_pred.cpu(),
                    "label": labels.cpu(),
                }
            )

    return results


def main():
    set_determinism(seed=0)

    msd_root = "data/MSD/Task09_Spleen"
    model_dir = "models"
    best_model_path = os.path.join(model_dir, "best_metric_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. "
            "Run train_baseline.py first."
        )

    # Data
    print("Loading validation data...")
    _, val_loader = get_dataloaders(
        msd_root=msd_root,
        batch_size=1,       # we'll use sliding-window inference for full volumes
        num_workers=4,
        use_cache=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(device=device, dropout=0.2)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded weights from {best_model_path}")

    # MC Dropout inference
    results = mc_dropout_inference(
        model=model,
        val_loader=val_loader,
        device=device,
        num_samples=10,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
    )

    # Save the first case for later visualization
    out_dir = "mc_outputs"
    os.makedirs(out_dir, exist_ok=True)
    if len(results) > 0:
        out_path = os.path.join(out_dir, "val_case0_mc_dropout.pt")
        torch.save(results[0], out_path)
        print(f"Saved first case MC results to {out_path}")


if __name__ == "__main__":
    main()
