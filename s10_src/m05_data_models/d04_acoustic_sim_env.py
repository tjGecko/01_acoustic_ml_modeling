from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
import yaml

class PyroomacousticsCfg(BaseModel):
    dimensions: Tuple[float, float, float] = (50.0, 50.0, 30.0)
    fs_hz: int = 16000
    max_order: int = 0
    abs_wall: float = 1.0
    abs_floor: float = 0.35
    abs_ceiling: float = 1.0
    air_absorption: bool = True

class MicConfig(BaseModel):
    xml_path: Path
    position: Tuple[float, float, float]
    mic_array_type: str
    mic_array_info: str

class GridConfig(BaseModel):
    radius: float = 25.0
    az_start: int = 0
    az_end: int = 355
    el_start: int = 0
    el_end: int = 90
    step: int = 5

class CaptureMetadata(BaseModel):
    wav_files_json: Path
    output_dir: Path
    author: str
    author_info: str
    capture_type: str
    capture_approach: str
    capture_info: str

class AcousticSimulationConfig(BaseModel):
    pyroomacoustics: PyroomacousticsCfg
    mics: MicConfig
    grid: GridConfig
    capture_metadata: CaptureMetadata

    @classmethod
    def from_yaml(cls, file_path: Path) -> 'AcousticSimulationConfig':
        """Load configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, file_path: Path):
        """Save configuration to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    @validator('mics')
    def validate_mic_config(cls, v):
        if not v.xml_path.exists():
            raise ValueError(f"Microphone XML file not found: {v.xml_path}")
        return v

    @validator('capture_metadata')
    def create_output_dir(cls, v):
        v.output_dir.mkdir(parents=True, exist_ok=True)
        return v
