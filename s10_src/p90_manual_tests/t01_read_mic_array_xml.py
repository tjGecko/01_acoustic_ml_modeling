"""
Test script for reading and validating the microphone array configuration.

This script demonstrates how to load and work with the microphone array configuration
from an XML file. It shows basic operations like loading the configuration and
accessing microphone properties.

Prerequisites:
1. Python 3.10 or higher
2. Virtual environment with required packages installed

Setup and Execution:
1. Navigate to the project root directory:
   ```bash
   cd /path/to/r03_Gimbal_Angle_Root
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # Linux/Mac
   .\\venv\\Scripts\\activate  # Windows
   ```

3. Install required packages (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   # From the project root directory
   python -m s10_src.p90_manual_tests.t01_read_mic_array_xml
   ```

   Alternatively, you can run it directly:
   ```bash
   # From the project root directory
   python s10_src/p90_manual_tests/t01_read_mic_array_xml.py
   ```

Expected Output:
- Loaded microphone array information
- Number of microphones in the array
- Coordinates of the first few microphones
- Example of retrieving a specific microphone by name
"""
from pathlib import Path
from xml.etree import ElementTree as ET

from s10_src.p05_data_models.d01_physical_mic_array import MicrophoneArray


# Import the data models from the project's data models module
# from p05_data_models.d01_physical_mic_array import MicrophoneArray


def main():
    """Main function to demonstrate loading and using the microphone array configuration."""
    # Path to the XML configuration file
    xml_path = (
        Path(__file__).parent.parent.parent / "05_config" / "uma_16_mic_array.xml"
    )
    
    try:
        # Load the microphone array configuration
        mic_array = MicrophoneArray.from_xml_file(xml_path)
        
        # Print basic information
        print(f"Loaded microphone array: {mic_array.name}")
        print(f"Number of microphones: {len(mic_array.microphones)}")
        
        # Print first 3 microphones as an example
        print("\nFirst 3 microphones:")
        for mic in mic_array.microphones[:3]:
            print(f"  {mic.name}: x={mic.x:.3f}, y={mic.y:.3f}, z={mic.z:.3f}")
        
        # Example of using the get_microphone_by_name method
        example_mic = mic_array.get_microphone_by_name("Point 1")
        if example_mic:
            print(f"\nFound microphone by name 'Point 1': {example_mic}")
            
    except FileNotFoundError:
        print(f"Error: File not found at {xml_path}")
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
