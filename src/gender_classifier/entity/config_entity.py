from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_input_shape: list
    params_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    trained_model_path_in_use: Path
    base_model_path: Path
    training_data: Path
    validation_data: Path
    params_steps_per_epoch:int
    params_validation_steps:int
    params_epochs: int
    params_batch_size: int
    params_image_size: list