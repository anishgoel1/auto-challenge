from typing import Dict
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GPUConfig:
    visible_devices: str = "0,1"
    tokenizers_parallelism: bool = True
    cuda_launch_blocking: bool = False
    cuda_arch_list: str = "7.0;7.5;8.0;8.6"

    @property
    def env_dict(self) -> Dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": self.visible_devices,
            "TOKENIZERS_PARALLELISM": str(self.tokenizers_parallelism).lower(),
            "CUDA_LAUNCH_BLOCKING": str(int(self.cuda_launch_blocking)),
            "TORCH_CUDA_ARCH_LIST": self.cuda_arch_list,
        }


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 5e-5
    patience: int = 5
    weight_decay: float = 0.02
    clip_grad_norm: float = 5.0
    num_workers: int = 8
    pin_memory: bool = True
    mixed_precision: bool = True
    amp_dtype: str = "float16"
    label_smoothing: float = 0.1


@dataclass
class ModelConfig:
    train_root: Path = Path("train")
    train_csv: Path = Path("train/train_videos.csv")
    test_root: Path = Path("test")
    output_path: Path = Path("predictions.csv")
    num_causes: int = 0  # Will be set after TEMPLATES processing
    frame_size: int = 224
    num_frames: int = 32
    model_name: str = "MCG-NJU/videomae-large-finetuned-kinetics"
    num_clips: int = 4
    clip_overlap: float = 0.2  


# Initialize configs
GPU_ENV_CONFIG = GPUConfig()
TRAINING_CONFIG = TrainingConfig()
MODEL_CONFIG = ModelConfig()

# Accident cause templates
# This was generated by looking at the mode label for each cause
# This is a valid approach because there is negligible label variance for each cause
TEMPLATES = {
    1: [
        "Motorcyclist fails to observe surrounding traffic during maneuvers",
        "Ego-car does not notice the motorcycle when turning",
        "Motorcycle drives too fast with a short braking distance",
        "Motorcycle is out of control",
    ],
    2: ["Cyclist drives on the motorway for a long time"],
    3: [
        "Ego-car does not notice the cyclists when turning",
    ],
    4: [
        "The truck do not give way to normal driving vehicles when turning or changing lanes",
    ],
    5: [
        "the truck does not notice the coming vehicles when crossing the road",
    ],
    6: [
        "Vehicle exceeds safe speed limit resulting in insufficient stopping distance",
        "Vehicle loses operational stability",
        "Heavy vehicle operator misses oncoming traffic during direction changes",
        "Heavy vehicle performs sudden lane deviation to avoid collision",
        "Leading vehicle performs unexpected rapid deceleration",
        "Ego-car does not notice the truck when turning or changing lanes",
        "Heavy vehicle experiences loss of stability",
        "Ego-car driver fail to estimate the accurate distance when turning",
        "The truck does not notice other vehicles when reversing",
        "Driver's attention diverted from road conditions",
        "Ego-car evades other vehicles, pedestrians or objects and changes lanes emergently",
    ],
    7: [
        "Vehicles exceed safe speed limit resulting in insufficient stopping distance",
        "Vehicles overlook approaching traffic while traversing the roadway",
        "Vehicles disregard traffic signal stop indication",
        "Vehicles fail to yield right of way during lane transitions or turns",
        "Driver's view is obstructed or unclear, preventing timely reaction",
        "Ego-car runs red light",
        "Vehicles make illegal U-turns at pedestrian crossing or no-passing zone",
        "vehicles drive too fast with short braking distance",
    ],
    9: [
        "Motorcycle does not notice the coming vehicles when crossing the road",
        "Two-wheeler violates traffic signal",
    ],
    10: [
        "Pedestrian does not notice the coming vehicles when crossing the street.",
    ],
}

ALL_CAUSES = set()
for causes in TEMPLATES.values():
    ALL_CAUSES.update(causes)

CAUSE_TO_ID = {cause: idx for idx, cause in enumerate(sorted(ALL_CAUSES))}
ID_TO_CAUSE = {idx: cause for cause, idx in CAUSE_TO_ID.items()}
CAUSE_TO_CATEGORY = {
    cause: cat for cat, causes in TEMPLATES.items() for cause in causes
}

# Update num_causes after processing templates
MODEL_CONFIG.num_causes = len(ALL_CAUSES)
