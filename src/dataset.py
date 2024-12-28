import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
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
        self.num_frames = MODEL_CONFIG["num_frames"]
        self.frame_size = MODEL_CONFIG["frame_size"]
        self.frame_sample_rate = 1

        # Preload all video paths
        if not is_test:
            if csv_file is None:
                raise ValueError("csv_file is required for training dataset")
            self.annotations = pd.read_csv(csv_file)
            self.video_paths = [
                os.path.join(root_dir, name) for name in self.annotations["video_name"]
            ]
        else:
            self.video_paths = list(self.root_dir.glob("*.mp4"))

    def _load_video(self, video_path: Path) -> List[np.ndarray]:
        """Load and sample frames from video.

        Args:
            video_path: Path to video file

        Returns:
            List of sampled frames as numpy arrays

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video loading fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]

                # Sample frame indices
                indices = sample_frame_indices(
                    clip_len=self.num_frames,
                    frame_sample_rate=self.frame_sample_rate,
                    seg_len=stream.frames,
                )

                frames = []
                start_index, end_index = indices[0], indices[-1]

                # Read frames at sampled indices
                for i, frame in enumerate(container.decode(video=0)):
                    if i > end_index:
                        break
                    if i >= start_index and i in indices:
                        frames.append(frame.to_ndarray(format="rgb24"))

                return frames

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)

        result = {
            "pixel_values": frames,
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


def create_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool = True, **kwargs
) -> DataLoader:
    """Create a DataLoader with standard configuration.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments passed to DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"],
        prefetch_factor=2,
        persistent_workers=True,
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
        dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True
    )


def get_test_dataloader() -> DataLoader:
    """Get DataLoader for test data."""
    dataset = VideoDataset(
        root_dir=MODEL_CONFIG["test_root"],
        is_test=True,
    )
    return create_dataloader(
        dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=False
    )
