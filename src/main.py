import os
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
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


def setup_gpu() -> torch.device:
    """Configure GPU settings and return device.

    Raises:
        RuntimeError: If no GPU is available
    """
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
    """Video classifier for accident cause prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.video_model = self._init_video_model()

    def _init_video_model(self) -> VideoMAEForVideoClassification:
        """Initialize the video classification model."""
        return VideoMAEForVideoClassification.from_pretrained(
            MODEL_CONFIG.model_name,
            num_labels=MODEL_CONFIG.num_causes,
            ignore_mismatched_sizes=True,
            image_size=MODEL_CONFIG.frame_size,
            num_frames=MODEL_CONFIG.num_frames,
        )

    def forward(self, frames_batch: torch.Tensor) -> torch.Tensor:
        """Process frames and return classification logits.
        
        Args:
            frames_batch: shape (B, num_clips, T, H, W, C)
        Returns:
            logits: shape (B, num_labels)
        """
        B, num_clips, T, H, W, C = frames_batch.shape
        
        # Reshape to process all clips at once
        frames = frames_batch.view(B * num_clips, T, H, W, C)
        
        # Rearrange from (B*num_clips, T, H, W, C) to (B*num_clips, T, C, H, W)
        frames = frames.permute(0, 1, 4, 2, 3)
        
        # Normalize to [0, 1]
        frames = frames.float() / 255.0
        
        # Get predictions for all clips
        logits = self.video_model(pixel_values=frames).logits
        
        # Reshape back to (B, num_clips, num_labels)
        logits = logits.view(B, num_clips, -1)
        
        # Average predictions across clips
        return logits.mean(dim=1)  # Shape: (B, num_labels)


class ModelTrainer:
    """Handles model training and prediction."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = self._initialize_model()
        self.scaler = GradScaler() if TRAINING_CONFIG.mixed_precision else None

    def _initialize_model(self) -> nn.Module:
        """Initialize and configure the model."""
        model = DirectCauseClassifier()
        model = model.to(self.device)  # Move entire model to device once
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        return model

    def _validate_paths(self) -> None:
        """Validate required paths exist."""
        if not MODEL_CONFIG.train_root.exists():
            raise ValueError(f"Training directory not found: {MODEL_CONFIG.train_root}")
        if not MODEL_CONFIG.train_csv.exists():
            raise ValueError(f"Training CSV not found: {MODEL_CONFIG.train_csv}")

    def train(self) -> None:
        """Train the model."""
        self._validate_paths()

        train_loader = get_train_dataloader(
            MODEL_CONFIG.train_root, MODEL_CONFIG.train_csv
        )
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(train_loader, optimizer)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(TRAINING_CONFIG.num_epochs):
            avg_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_model()
            else:
                patience_counter += 1
                if patience_counter >= TRAINING_CONFIG.patience:
                    print("Early stopping triggered")
                    break

        self._load_best_model()

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
    ) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            loss = self._train_step(batch, optimizer)
            scheduler.step()
            epoch_losses.append(loss)

            if batch_idx % 10 == 0:
                self._log_progress(batch_idx, len(train_loader), loss, epoch)

        return sum(epoch_losses) / len(epoch_losses)

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Perform single training step with label smoothing."""
        pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
        cause_ids = batch["cause_id"].to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=TRAINING_CONFIG.mixed_precision):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(pixel_values)
            # Add label smoothing to loss
            loss = F.cross_entropy(
                logits, 
                cause_ids,
                label_smoothing=TRAINING_CONFIG.label_smoothing
            )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                self._clip_gradients()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._clip_gradients()
                optimizer.step()

        return loss.item()

    def predict(self, test_loader: torch.utils.data.DataLoader) -> List[Dict[str, Any]]:
        """Generate predictions for test data."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                batch_predictions = self._predict_batch(batch)
                predictions.extend(batch_predictions)

        return predictions

    def _predict_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Generate predictions for a single batch."""
        pixel_values = batch["pixel_values"].to(self.device)
        video_names = batch["video_name"]

        logits = self.model(pixel_values)
        pred_cause_ids = torch.argmax(logits, dim=1)

        return [
            {
                "video_name": name,
                "predicted_category": CAUSE_TO_CATEGORY[ID_TO_CAUSE[cause_id.item()]],
                "predicted_cause": ID_TO_CAUSE[cause_id.item()],
            }
            for name, cause_id in zip(video_names, pred_cause_ids)
        ]

    def _setup_optimizer(self) -> AdamW:
        """Initialize the optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG.learning_rate,
            weight_decay=TRAINING_CONFIG.weight_decay,
        )

    def _setup_scheduler(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.OneCycleLR:
        """Initialize the learning rate scheduler."""
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=TRAINING_CONFIG.learning_rate,
            epochs=TRAINING_CONFIG.num_epochs,
            steps_per_epoch=len(train_loader),
        )

    def _clip_gradients(self) -> None:
        """Clip gradients to prevent exploding gradients."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), TRAINING_CONFIG.clip_grad_norm
        )

    def _save_model(self) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), "best_model.pth")

    def _load_best_model(self) -> None:
        """Load best model state dict."""
        self.model.load_state_dict(torch.load("best_model.pth"))

    @staticmethod
    def _log_progress(
        batch_idx: int, total_batches: int, loss: float, epoch: int
    ) -> None:
        """Log training progress."""
        print(
            f"Epoch {epoch+1}/{TRAINING_CONFIG.num_epochs} | "
            f"Batch {batch_idx}/{total_batches} | "
            f"Loss: {loss:.4f}"
        )


def main() -> None:
    """Main execution function."""
    # Apply NVIDIA environment variables
    os.environ.update(GPU_ENV_CONFIG.env_dict)

    # Setup device
    device = setup_gpu()
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)

    print("Starting training pipeline...")
    trainer = ModelTrainer(device)
    trainer.train()

    print("\nGenerating predictions for test set...")
    test_loader = get_test_dataloader()
    predictions = trainer.predict(test_loader)

    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(MODEL_CONFIG.output_path, index=False)
    print(f"\nPredictions saved to: {MODEL_CONFIG.output_path}")


if __name__ == "__main__":
    main()
