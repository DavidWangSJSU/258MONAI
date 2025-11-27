# train_baseline.py
#
# Baseline 3D UNet training on MSD Task09_Spleen using MONAI.
# Uses the DataLoaders defined in dataset.py

import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter  # optional; you can remove if not using TB

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from monai.utils import set_determinism

from dataset import get_dataloaders


def main():
    # -----------------------
    # Hyperparameters
    # -----------------------
    msd_root = "data/MSD/Task09_Spleen"
    batch_size = 2
    num_workers = 4
    max_epochs = 50
    learning_rate = 1e-4
    val_interval = 1  # validate every N epochs
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

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

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,  # binary spleen vs background
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    ).to(device)

    loss_function = DiceCELoss(
        sigmoid=True,
        to_onehot_y=False,
        squared_pred=True,
        smooth_nr=0.0,
        smooth_dr=1e-5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    # Optional: TensorBoard logger (you can comment this out if you don’t use it)
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
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"  step {step:3d} / loss = {loss.item():.4f}")

        epoch_loss /= step
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f} (time: {elapsed:.1f}s)")
        writer.add_scalar("train/loss", epoch_loss, epoch)

        # -----------------------
        # Validation
        # -----------------------
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    val_outputs = model(val_images)

                    # decollate into list of [C,H,W,D] tensors, apply post transforms
                    val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs_list, y=val_labels_list)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

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
