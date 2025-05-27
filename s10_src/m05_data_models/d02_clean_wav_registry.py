"""
Data model for the clean WAV file registry.

This module defines the data structure for tracking clean WAV files,
including their paths, drone types, and metadata about the registry creation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class DroneType(str, Enum):
    """Supported drone types in the registry."""
    BEBOP = "bebop"
    MEMBO = "membo"


class CleanWavEntry(BaseModel):
    """Represents a single entry in the clean WAV registry."""
    file_path: Path = Field(..., description="Absolute path to the WAV file")
    drone_type: DroneType = Field(..., description="Type of drone in the recording")
    file_size: int = Field(..., description="Size of the file in bytes")
    modified_time: float = Field(..., description="Last modified timestamp of the file")
    snr_db: float = Field(default=20.0, description="Signal-to-noise ratio in decibels")


class RegistryHeader(BaseModel):
    """Metadata about the registry creation."""
    created_by: str = Field(..., description="Script that generated this registry")
    created_at: datetime = Field(default_factory=datetime.now, description="When the registry was created")
    description: str = Field(default="Clean WAV file registry", description="Purpose of this registry")
    filter_terms: List[str] = Field(..., description="Terms used to filter the WAV files")
    root_dir: Path = Field(..., description="Root directory that the file paths are relative to")


class CleanWavRegistry(BaseModel):
    """Container for clean WAV file registry with metadata."""
    header: RegistryHeader
    entries: List[CleanWavEntry] = Field(default_factory=list, description="List of WAV file entries")

    @property
    def count_by_drone_type(self) -> dict[DroneType, int]:
        """Count entries by drone type."""
        from collections import defaultdict
        counts = defaultdict(int)
        for entry in self.entries:
            counts[entry.drone_type] += 1
        return dict(counts)

    def add_entry(self, file_path: Path, drone_type: DroneType) -> None:
        """Add a new entry to the registry."""
        if not file_path.exists():
            raise FileNotFoundError(f"WAV file not found: {file_path}")
            
        stat = file_path.stat()
        self.entries.append(CleanWavEntry(
            file_path=file_path,
            drone_type=drone_type,
            file_size=stat.st_size,
            modified_time=stat.st_mtime
        ))

    @classmethod
    def create(
        cls,
        created_by: str,
        filter_terms: List[str],
        root_dir: Path,
        description: str = "Clean WAV file registry"
    ) -> 'CleanWavRegistry':
        """Create a new registry with the given metadata."""
        return cls(
            header=RegistryHeader(
                created_by=created_by,
                filter_terms=filter_terms,
                root_dir=root_dir,
                description=description
            )
        )

    def save_to_file(self, file_path: Path) -> None:
        """Save the registry to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.model_dump_json(indent=2, exclude_none=True))

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'CleanWavRegistry':
        """Load a registry from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())
