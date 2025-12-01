# train_baseline.py
#
# Baseline 3D UNet training on MSD Task09_Spleen using MONAI.
# Uses the DataLoaders defined in dataset.py

import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter  # optional; remove if not using TB

from monai.networks.nets import UNet
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

from dataset import get_dataloaders


def _extract_batch(batch_data, device):
    """
    Handle both cases:
    - batch_data is a dict: {"image": ..., "label": ...}
    - batch_data is a list/tuple of dicts: [ {"image": ..., "label": ...}, ... ]
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
    """
    Compute Dice score for binary predictions and labels.
    pred and target should be tensors of same shape, typically [B, 1, H, W, D].
    """
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)
    return dice.item()


def main():
    # -----------------------
    # Hyperparameters
    # -----------------------
    msd_root = "data/MSD/Task09_Spleen"
    batch_size = 2
    num_workers = 4
    max_epochs = 2  # small for testing; increase (e.g., 50) for real training
    learning_rate = 1e-4
    val_interval = 1  # validate every N epochs
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    roi_size = (96, 96, 96)  # patch size for sliding-window inference

    set_determinism(seed=0)

    # -----------------------
    # Data
    # -----------------------
    print("Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        msd_root=msd_root,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cache=True,
    )

    # -----------------------
    # Model / Loss / Optimizer
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # MONAI UNet (3D)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,  # binary spleen vs background
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.2,  # enable dropout for later MC Dropout inference
    ).to(device)

    # Simple binary segmentation loss (logits + BCE)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Optional: TensorBoard logger
    writer = SummaryWriter(log_dir=os.path.join(model_dir, "runs"))

    best_metric = -1.0
    best_metric_epoch = -1
    best_model_path = os.path.join(model_dir, "best_metric_model.pth")

    # -----------------------
    # Training loop
    # -----------------------
    print("Starting training...")
    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        print("-" * 10)
        print(f"Epoch {epoch}/{max_epochs}")

        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = _extract_batch(batch_data, device)

            optimizer.zero_grad()
            outputs = model(inputs)  # logits
            loss = loss_function(outputs, labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"  step {step:3d} / loss = {loss.item():.4f}")

        epoch_loss /= max(step, 1)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f} (time: {elapsed:.1f}s)")
        writer.add_scalar("train/loss", epoch_loss, epoch)

        # -----------------------
        # Validation (with sliding-window inference)
        # -----------------------
        if epoch % val_interval == 0:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = _extract_batch(val_data, device)

                    # sliding-window inference handles arbitrary volume sizes safely
                    val_logits = sliding_window_inference(
                        val_inputs,
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                    )
                    probs = torch.sigmoid(val_logits)
                    preds = (probs > 0.5).float()

                    dice = dice_score(preds, val_labels)
                    val_dices.append(dice)

            metric = sum(val_dices) / len(val_dices) if len(val_dices) > 0 else 0.0

            print(f"Validation Dice: {metric:.4f}")
            writer.add_scalar("val/dice", metric, epoch)

            # Save best model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"  New best metric: {best_metric:.4f} at epoch {best_metric_epoch}. "
                    f"Saved model to {best_model_path}"
                )

    print(
        f"Training completed. Best validation Dice: {best_metric:.4f} at epoch {best_metric_epoch}"
    )
    writer.close()


if __name__ == "__main__":
    main()
