"""
Manual Tests Package

This package contains manual test scripts for the Gimbal Angle project.
These tests are meant to be run interactively and may require manual verification.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path for easier imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__all__ = [
    # List test modules here as they are added
    't01_read_mic_array_xml',
]
