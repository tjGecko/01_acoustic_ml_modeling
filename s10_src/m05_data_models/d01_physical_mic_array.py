from typing import List, Optional, Tuple
import numpy as np
from xml.etree import ElementTree as ET
from pydantic import BaseModel, Field, field_validator


class MicrophonePosition(BaseModel):
    """Represents a single microphone's position in 3D space."""
    name: str = Field(..., description="Name/identifier of the microphone")
    x: float = Field(..., description="X-coordinate in meters")
    y: float = Field(..., description="Y-coordinate in meters")
    z: float = Field(..., description="Z-coordinate in meters")

    @classmethod
    def from_xml_element(cls, elem: ET.Element) -> 'MicrophonePosition':
        """Create a MicrophonePosition from an XML element."""
        return cls(
            name=elem.attrib["Name"],
            x=float(elem.attrib["x"]),
            y=float(elem.attrib["y"]),
            z=float(elem.attrib["z"])
        )


class MicrophoneArrayOrientation(BaseModel):
    """Represents the orientation of a microphone array in 3D space."""
    azimuth_deg: float = Field(
        default=0.0,
        description="Azimuth angle in degrees (0-360), 0 is along positive X-axis, 90 is along positive Y-axis"
    )
    elevation_deg: float = Field(
        default=0.0,
        description="Elevation angle in degrees (-90 to 90), 0 is horizontal, 90 is straight up"
    )

    def get_direction_vector(self) -> np.ndarray:
        """
        Calculate the unit direction vector based on azimuth and elevation.
        
        Returns:
            np.ndarray: Unit vector [x, y, z] pointing in the array's forward direction
        """
        az_rad = np.radians(self.azimuth_deg)
        el_rad = np.radians(self.elevation_deg)
        
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        return np.array([x, y, z])


class MicrophoneArray(BaseModel):
    """Represents a microphone array configuration with position and orientation."""
    name: str = Field(..., description="Name of the microphone array")
    microphones: List[MicrophonePosition] = Field(
        default_factory=list,
        description="List of microphone positions in the array"
    )
    orientation: MicrophoneArrayOrientation = Field(
        default_factory=MicrophoneArrayOrientation,
        description="Orientation of the microphone array"
    )
    position: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Optional position of the array center in 3D space [x, y, z] in meters"
    )

    @field_validator('orientation', mode='before')
    @classmethod
    def validate_orientation(cls, v):
        if isinstance(v, dict):
            return MicrophoneArrayOrientation(**v)
        return v

    def get_forward_direction(self) -> np.ndarray:
        """
        Get the forward direction vector of the array.
        
        Returns:
            np.ndarray: Unit vector in the array's forward direction
        """
        return self.orientation.get_direction_vector()

    @classmethod
    def from_xml_file(
        cls,
        file_path: str,
        orientation: Optional[MicrophoneArrayOrientation] = None,
        position: Optional[Tuple[float, float, float]] = None
    ) -> 'MicrophoneArray':
        """
        Load microphone array configuration from an XML file.
        
        Args:
            file_path: Path to the XML configuration file
            orientation: Optional orientation of the array
            position: Optional position of the array center [x, y, z] in meters
            
        Returns:
            MicrophoneArray: Configured microphone array instance
            
        Raises:
            FileNotFoundError: If the XML file doesn't exist
            ET.ParseError: If the XML is malformed
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        array = cls(name=root.attrib["name"])
        
        for elem in root.findall("pos"):
            array.microphones.append(MicrophonePosition.from_xml_element(elem))
        
        return array
    
    def get_microphone_by_name(self, name: str) -> Optional[MicrophonePosition]:
        """
        Get a microphone by its name.
        
        Args:
            name: Name of the microphone to find
            
        Returns:
            Optional[MicrophonePosition]: The found microphone or None if not found
        """
        for mic in self.microphones:
            if mic.name == name:
                return mic
        return None
