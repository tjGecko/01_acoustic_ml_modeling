# Gimbal Angle Project Overview

## Project Purpose
This project focuses on processing and analyzing microphone array data, specifically for a 16-microphone UMA (Uniform Microphone Array) configuration. The system is designed to handle spatial audio processing, with a particular emphasis on gimbal angle calculations based on microphone array data.

## Project Structure

### Directory Naming Conventions
- **00_docs/**: Project documentation
  - `v05_Gimbal_Angle/`: Versioned documentation for the gimbal angle project
- **05_config/**: Configuration files, including microphone array definitions
- **s10_src/**: Source code (note the 's' prefix for valid Python module naming)
  - `m05_data_models/`: Data models and domain objects
  - `m90_manual_tests/`: Test scripts and validation code

### Python Package Structure
The project uses a custom naming convention for Python modules:
- `s10_src/`: The 's' prefix ensures valid Python module naming (avoids starting with numbers)
- Each subdirectory contains an `__init__.py` file to make it a proper Python package
- Module names follow a pattern of `m##_descriptive_name` where `##` indicates the module's order/priority

## Key Components

### 1. Microphone Array Configuration (`m05_data_models/d01_physical_mic_array.py`)
- Defines the physical layout of the microphone array
- Uses Pydantic for data validation and settings management
- Supports loading configuration from XML files
- Provides 3D spatial information for each microphone

### 2. Test Scripts (`m90_manual_tests/`)
- Contains manual test cases for validating functionality
- Example: `t01_read_mic_array_xml.py` demonstrates loading and working with microphone array data

## Development Environment

### Prerequisites
- Python 3.10 or higher
- Virtual environment (recommended)
- Dependencies listed in `requirements.txt`

### Setup
```bash
# Clone the repository
# Navigate to project root
cd /path/to/r03_Gimbal_Angle_Root

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# From project root
python -m s10_src.m90_manual_tests.t01_read_mic_array_xml
```

## Naming Conventions

### Files and Directories
- Use lowercase with underscores for file and directory names
- Prefix numerical directories with appropriate padding (e.g., `00_docs`, `05_config`)
- Source code directories use 's' prefix for valid Python module names (e.g., `s10_src`)

### Python Modules
- Module names start with 'm' followed by 2-digit number and description (e.g., `m05_data_models`)
- Test files start with 't' followed by 2-digit number and description (e.g., `t01_read_mic_array_xml.py`)
- Classes use PascalCase
- Functions and variables use snake_case

## Best Practices

### Code Organization
- Keep related functionality in dedicated modules
- Use `__init__.py` files to define package structure
- Document public APIs with docstrings
- Follow type hints for better code maintainability

### Version Control
- Use feature branches for development
- Follow conventional commits for commit messages
- Keep commits focused and atomic

## Future Work
- Add automated testing
- Implement gimbal angle calculation logic
- Add documentation for API usage
- Include example configurations for different microphone arrays