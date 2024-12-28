import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from config import MODEL_CONFIG, TRAINING_CONFIG, CAUSE_TO_ID


def sample_frame_indices(
    clip_len: int, frame_sample_rate: int, seg_len: int
) -> np.ndarray:
    """
    Sample code from HuggingFace Tutorial.
    Sample a given number of frame indices from the video.
    Args:
        clip_len: Total number of frames to sample
        frame_sample_rate: Sample every n-th frame
        seg_len: Maximum allowed index of sample's last frame
    Returns:
        indices: Array of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    
    # Handle case where video is too short
    if seg_len <= converted_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    return indices


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        csv_file: Optional[str | Path] = None,
        is_test: bool = False,
    ):
        """Initialize VideoDataset.

        Args:
            root_dir: Directory containing video files
            csv_file: Path to CSV containing annotations (required for training)
            is_test: Whether this is a test dataset
        """
        self.root_dir = Path(root_dir)
        self.is_test = is_test
        self.num_frames = MODEL_CONFIG.num_frames
        self.frame_size = MODEL_CONFIG.frame_size
        self.frame_sample_rate = 1

        # Preload all video paths
        if not is_test:
            if csv_file is None:
                raise ValueError("csv_file is required for training dataset")
            self.annotations = pd.read_csv(csv_file)
            # Convert paths to Path objects
            self.video_paths = [
                Path(self.root_dir) / name for name in self.annotations["video_name"]
            ]
        else:
            self.video_paths = list(self.root_dir.glob("*.mp4"))

    def _load_video(self, video_path: str | Path) -> torch.Tensor:
        """Load and sample frames from video."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                indices = sample_frame_indices(
                    clip_len=self.num_frames,
                    frame_sample_rate=self.frame_sample_rate,
                    seg_len=stream.frames,
                )

                frames = []
                for i, frame in enumerate(container.decode(video=0)):
                    if i in indices:
                        img = frame.to_ndarray(format="rgb24")
                        # Ensure consistent size
                        img = cv2.resize(img, (self.frame_size, self.frame_size))
                        frames.append(img)

                # Handle case where we couldn't get enough frames
                while len(frames) < self.num_frames:
                    # Duplicate the last frame if we don't have enough
                    frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3)))

                # Ensure we only take the exact number of frames we need
                frames = frames[:self.num_frames]
                
                return torch.from_numpy(np.stack(frames))

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)  # Now returns tensor

        result = {
            "pixel_values": frames,  # Already a tensor of shape (T, H, W, C)
            "video_name": video_path.name,
        }

        if not self.is_test:
            row = self.annotations.iloc[idx]
            result["cause_id"] = torch.tensor(
                CAUSE_TO_ID[row["causes"]], dtype=torch.long
            )

        return result

    def __len__(self) -> int:
        return len(self.video_paths)


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized tensors."""
    # Separate the different items in the batch
    pixel_values = [item['pixel_values'] for item in batch]
    video_names = [item['video_name'] for item in batch]
    
    # Stack pixel values (they should all be the same size after preprocessing)
    pixel_values = torch.stack(pixel_values)
    
    result = {
        'pixel_values': pixel_values,
        'video_name': video_names,
    }
    
    # Handle cause_id if it exists (for training)
    if 'cause_id' in batch[0]:
        cause_ids = torch.stack([item['cause_id'] for item in batch])
        result['cause_id'] = cause_ids
        
    return result


def create_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool = True, **kwargs
) -> DataLoader:
    """Create a DataLoader with standard configuration."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=custom_collate_fn,
        **kwargs,
    )


def get_train_dataloader(train_root: str | Path, train_csv: str | Path) -> DataLoader:
    """Get DataLoader for training data."""
    dataset = VideoDataset(
        root_dir=train_root,
        csv_file=train_csv,
        is_test=False,
    )
    return create_dataloader(
        dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=True
    )


def get_test_dataloader() -> DataLoader:
    """Get DataLoader for test data."""
    dataset = VideoDataset(
        root_dir=MODEL_CONFIG.test_root,
        is_test=True,
    )
    return create_dataloader(
        dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False
    )
