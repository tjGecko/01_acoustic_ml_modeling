from typing import List, Optional
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field


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


class MicrophoneArray(BaseModel):
    """Represents a microphone array configuration."""
    name: str = Field(..., description="Name of the microphone array")
    microphones: List[MicrophonePosition] = Field(
        default_factory=list,
        description="List of microphone positions in the array"
    )

    @classmethod
    def from_xml_file(cls, file_path: str | str) -> 'MicrophoneArray':
        """
        Load microphone array configuration from an XML file.
        
        Args:
            file_path: Path to the XML configuration file
            
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
