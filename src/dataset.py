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
    clip_len: int,
    frame_sample_rate: int,
    seg_len: int,
    clip_idx: int = 0,
    num_clips: int = 1,
    clip_overlap: float = 0.0
) -> np.ndarray:
    """
    Inspired by HuggingFace VideoMAE Tutorial.
    Sample frame indices for a clip, supporting multiple clips with overlap.
    
    Args:
        clip_len: Number of frames to sample
        frame_sample_rate: Sample every n-th frame
        seg_len: Total number of frames in video
        clip_idx: Which clip to sample (0 to num_clips-1)
        num_clips: Total number of clips to divide video into
        clip_overlap: Overlap between clips (0.0 to 1.0)
    """
    converted_len = int(clip_len * frame_sample_rate)
    
    if seg_len <= converted_len:
        # Video too short - sample frames uniformly
        return np.linspace(0, seg_len - 1, num=clip_len).astype(np.int64)
        
    # Calculate overlap in frames
    overlap_frames = int(converted_len * clip_overlap)
    
    # Calculate start and end for each clip
    effective_duration = converted_len - overlap_frames
    start_idx = clip_idx * effective_duration
    end_idx = start_idx + converted_len
    
    # Ensure we don't exceed video length
    end_idx = min(end_idx, seg_len)
    start_idx = max(0, end_idx - converted_len)
    
    indices = np.linspace(start_idx, end_idx - 1, num=clip_len)
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    return indices


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        csv_file: Optional[str | Path] = None,
        is_test: bool = False,
    ):
        """Initialize VideoDataset."""
        self.root_dir = Path(root_dir)
        self.is_test = is_test
        self.num_frames = MODEL_CONFIG.num_frames
        self.frame_size = MODEL_CONFIG.frame_size
        self.num_clips = MODEL_CONFIG.num_clips
        self.clip_overlap = MODEL_CONFIG.clip_overlap
        
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
        """Load and sample frames from video for multiple clips."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                all_frames = [
                    cv2.resize(frame.to_ndarray(format="rgb24"), (self.frame_size, self.frame_size))
                    for frame in container.decode(video=0)
                ]
                
                clips = []
                for clip_idx in range(self.num_clips):
                    indices = sample_frame_indices(
                        clip_len=self.num_frames,
                        frame_sample_rate=1,
                        seg_len=len(all_frames),
                        clip_idx=clip_idx,
                        num_clips=self.num_clips,
                        clip_overlap=self.clip_overlap
                    )
                    
                    # Sample frames for this clip
                    clip_frames = [all_frames[i] for i in indices]
                    
                    # Handle case where we couldn't get enough frames
                    while len(clip_frames) < self.num_frames:
                        clip_frames.append(
                            clip_frames[-1] if clip_frames 
                            else np.zeros((self.frame_size, self.frame_size, 3))
                        )
                    
                    clips.append(np.stack(clip_frames[:self.num_frames]))
                
                # Stack all clips together
                return torch.from_numpy(np.stack(clips))  # Shape: (num_clips, T, H, W, C)

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)  

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
    """Custom collate function to handle multiple clips per video."""
    pixel_values = [item['pixel_values'] for item in batch]  # Each item is (num_clips, T, H, W, C)
    video_names = [item['video_name'] for item in batch]
    
    # Stack along batch dimension
    pixel_values = torch.stack(pixel_values)  # Shape: (B, num_clips, T, H, W, C)
    
    result = {
        'pixel_values': pixel_values,
        'video_name': video_names,
    }
    
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
