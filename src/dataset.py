import os
from functools import lru_cache
from typing import List, Dict, Any

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from config import MODEL_CONFIG, TRAINING_CONFIG, CAUSE_TO_ID


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        csv_file: str = None,
        is_test: bool = False,
    ):
        self.root_dir = root_dir
        self.is_test = is_test
        self.num_frames = MODEL_CONFIG["num_frames"]
        self.frame_size = MODEL_CONFIG["frame_size"]

        # Preload all video paths
        if not is_test:
            self.annotations = pd.read_csv(csv_file)
            self.video_paths = [
                os.path.join(root_dir, name) for name in self.annotations["video_name"]
            ]
        else:
            self.video_files = [f for f in os.listdir(root_dir) if f.endswith(".mp4")]
            self.video_paths = [os.path.join(root_dir, f) for f in self.video_files]

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(
            frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_LINEAR
        )

    @lru_cache(maxsize=32)  # Cache recently loaded videos
    def _load_video(self, video_path: str) -> List[np.ndarray]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            container = av.open(video_path)
            stream = container.streams.video[0]

            # Calculate frame indices for uniform sampling
            total_frames = stream.frames
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []

            for idx in indices:
                container.seek(int(idx), stream=stream)
                frame = next(container.decode(video=0))
                frame = frame.to_ndarray(format="rgb24")
                frame = self._resize_frame(frame)
                frames.append(frame)

            container.close()
            return frames

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_test:
            video_path = self.video_paths[idx]
            video_name = os.path.basename(video_path)
            frames = self._load_video(video_path)

            frames = [
                np.transpose(np.array(frame), (2, 0, 1)) for frame in frames
            ]  # Change from HWC to CHW

            frames = np.stack(frames)  # Shape: (T, C, H, W)
            pixel_values = torch.from_numpy(frames).float()  # Shape: (T, C, H, W)

            return {
                "pixel_values": pixel_values,
                "video_name": video_name,
            }
        else:
            row = self.annotations.iloc[idx]
            video_path = self.video_paths[idx]
            frames = self._load_video(video_path)

            frames = [
                np.transpose(np.array(frame), (2, 0, 1)) for frame in frames
            ]  # Change from HWC to CHW

            frames = np.stack(frames)  # Shape: (T, C, H, W)
            pixel_values = torch.from_numpy(frames).float()  # Shape: (T, C, H, W)

            return {
                "pixel_values": pixel_values,
                "cause_id": torch.tensor(CAUSE_TO_ID[row["causes"]], dtype=torch.long),
                "video_name": row["video_name"],
            }

    def __len__(self) -> int:
        return len(self.video_paths)


def get_train_dataloader(train_root: str, train_csv: str) -> DataLoader:
    dataset = VideoDataset(
        root_dir=train_root,
        csv_file=train_csv,
        is_test=False,
    )

    return DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"],
        prefetch_factor=2,
        persistent_workers=True,
    )


def get_test_dataloader() -> DataLoader:
    dataset = VideoDataset(
        root_dir=MODEL_CONFIG["test_root"],
        is_test=True,
    )

    return DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"],
        prefetch_factor=2,
        persistent_workers=True,
    )
