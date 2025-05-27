"""
Test script for reading and validating the microphone array configuration.
"""
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

# Import the data models from the project's data models module
from m05_data_models.d01_physical_mic_array import MicrophoneArray


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
