import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from transformers import (
    VideoMAEForVideoClassification,
)
from dataset import get_train_dataloader, get_test_dataloader
from config import (
    GPU_ENV_CONFIG,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    ID_TO_CAUSE,
    CAUSE_TO_CATEGORY,
)
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from pathlib import Path


def setup_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. This script requires an NVIDIA GPU.")

    print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Optimize CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    return torch.device("cuda")


class DirectCauseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_CONFIG["model_name"],
            num_labels=MODEL_CONFIG["num_causes"],
            ignore_mismatched_sizes=True,
            image_size=MODEL_CONFIG["frame_size"],
            num_frames=MODEL_CONFIG["num_frames"],
        )

    def forward(self, pixel_values):
        if pixel_values.dim() == 4:  # (T, C, H, W)
            pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension
        elif pixel_values.dim() == 5:  # (B, T, C, H, W)
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {pixel_values.dim()}D")
        return self.video_model(pixel_values).logits


def train_model(device: torch.device) -> DirectCauseClassifier:
    if not Path(MODEL_CONFIG["train_root"]).exists():
        raise ValueError(f"Training directory not found: {MODEL_CONFIG['train_root']}")
    if not Path(MODEL_CONFIG["train_csv"]).exists():
        raise ValueError(f"Training CSV not found: {MODEL_CONFIG['train_csv']}")

    model = DirectCauseClassifier().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    train_loader = get_train_dataloader(
        MODEL_CONFIG["train_root"], MODEL_CONFIG["train_csv"]
    )
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAINING_CONFIG["learning_rate"],
        epochs=TRAINING_CONFIG["num_epochs"],
        steps_per_epoch=len(train_loader),
    )
    scaler = GradScaler("cuda") if TRAINING_CONFIG["mixed_precision"] else None
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            cause_ids = batch["cause_id"].to(device, non_blocking=True)

            with autocast("cuda", enabled=TRAINING_CONFIG["mixed_precision"]):
                optimizer.zero_grad(set_to_none=True)
                logits = model(pixel_values)
                loss = F.cross_entropy(logits, cause_ids)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), TRAINING_CONFIG["clip_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), TRAINING_CONFIG["clip_grad_norm"]
                    )
                    optimizer.step()

            scheduler.step()
            epoch_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_CONFIG["patience"]:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model


def predict(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            video_names = batch["video_name"]

            logits = model(pixel_values)
            pred_cause_ids = torch.argmax(logits, dim=1)

            for name, cause_id in zip(video_names, pred_cause_ids):
                cause = ID_TO_CAUSE[cause_id.item()]
                category = CAUSE_TO_CATEGORY[cause]
                predictions.append(
                    {
                        "video_name": name,
                        "predicted_category": category,
                        "predicted_cause": cause,
                    }
                )

    return predictions


def main():
    # Apply NVIDIA environment variables
    os.environ.update(GPU_ENV_CONFIG)

    # Setup device
    device = setup_gpu()
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)

    print("Starting training pipeline...")

    model = train_model(device=device)
    print("\nGenerating predictions for test set...")
    test_loader = get_test_dataloader()
    predictions = predict(model, test_loader, device)

    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(MODEL_CONFIG["output_path"], index=False)
    print(f"\nPredictions saved to: {MODEL_CONFIG['output_path']}")


if __name__ == "__main__":
    main()
