# Ideal WAV File Extraction Process

## Overview
This document explains the process of extracting and cataloging ideal WAV files for drone audio analysis. The system is designed to:

1. Scan a directory for WAV files
2. Filter files based on drone type (Bebop or Membo)
3. Create a structured registry with file metadata
4. Save the registry for future reference

## System Components

### 1. Data Model (`d02_clean_wav_registry.py`)

The core data structure is defined using Pydantic models:

```python
class DroneType(str, Enum):
    BEBOP = "bebop"
    MEMBO = "membo"

class CleanWavEntry(BaseModel):
    file_path: Path
    drone_type: DroneType
    file_size: int
    modified_time: float
    snr_db: float = 20.0  # Default SNR value

class CleanWavRegistry(BaseModel):
    header: RegistryHeader
    entries: List[CleanWavEntry] = []
```

### 2. Main Script (`t01_create_ideal_data_subset.py`)

The main script performs these steps:

1. **Configuration Loading**
   - Loads settings from `training_data.yaml`
   - Defines absolute paths for input/output

2. **File Discovery**
   - Recursively scans the source directory for WAV files
   - Filters files containing 'bebop' or 'membo' in their names

3. **Registry Creation**
   - Creates a registry entry for each file
   - Automatically detects drone type from filename
   - Captures file metadata (size, modification time)

4. **Output**
   - Saves the registry as JSON
   - Prints a summary of found files

## How to Use

### Prerequisites
- Python 3.10+
- Dependencies in `requirements.txt`
- Access to the source WAV files directory

### Running the Script

```bash
# From the project root
python -m s10_src.m90_manual_tests.m01_gen_synthetic_data.t01_create_ideal_data_subset
```

### Expected Output

```
Loading configuration from: /path/to/training_data.yaml
Scanning for WAV files in: /path/to/source/directory
Filtering for files containing: bebop, membo

Found 1332 matching WAV files:
- MEMBO: 666 files
- BEBOP: 666 files

Saved registry to: /path/to/clean_wav_registry.json

Sample files:
MEMBO samples:
- Membo_1_047-membo_003_.wav
- extra_membo_D2_2012.wav
...
```

## Registry Structure

The generated JSON registry includes:

- **Header**: Metadata about the registry creation
  - Creation timestamp
  - Script that generated it
  - Filter terms used
  - Root directory

- **Entries**: List of WAV files with:
  - Absolute file path
  - Drone type (BEBOP/MEMBO)
  - File size (bytes)
  - Last modified timestamp
  - SNR value (default: 20.0 dB)

## Error Handling

The script includes comprehensive error handling for:
- Missing configuration files
- Invalid file paths
- Permission issues
- Malformed YAML/JSON

## Maintenance

### Adding New Drone Types
1. Add to the `DroneType` enum in `d02_clean_wav_registry.py`
2. Update the `determine_drone_type()` function to handle the new type

### Modifying Metadata
- Edit the `CleanWavEntry` class to add/remove fields
- The registry will automatically include new fields for all entries

## Troubleshooting

### Common Issues
1. **File Not Found**
   - Verify the path in `training_data.yaml`
   - Check filesystem permissions

2. **No Files Found**
   - Confirm the source directory contains WAV files
   - Check the filter terms in the script

3. **Permission Denied**
   - Ensure write permissions for the output directory
   - Check file ownership

## Future Improvements

1. Add support for custom metadata
2. Implement incremental updates
3. Add file validation (e.g., check WAV headers)
4. Support for additional audio formats

## Dependencies

- Python 3.10+
- Pydantic (for data validation)
- PyYAML (for config parsing)

---
*Last Updated: May 27, 2025*