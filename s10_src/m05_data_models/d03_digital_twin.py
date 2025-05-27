from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
import yaml
import os

from .d01_physical_mic_array import MicrophoneArray


def calculate_azimuth_elevation(
    origin: np.ndarray,
    target: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate azimuth (degrees), elevation (degrees), and distance from origin to target.
    
    Args:
        origin: Origin point [x, y, z]
        target: Target point [x, y, z]
        
    Returns:
        Tuple of (azimuth_deg, elevation_deg, distance)
    """
    # Convert to numpy arrays if they're not already
    origin = np.array(origin, dtype=float)
    target = np.array(target, dtype=float)
    
    # Calculate the vector from origin to target
    vec = target - origin
    x, y, z = vec
    
    # Calculate distance (radius)
    distance = np.linalg.norm(vec)
    
    # Calculate azimuth (in degrees, 0-360, 0 is +x, 90 is +y)
    azimuth_rad = np.arctan2(y, x)
    azimuth_deg = np.degrees(azimuth_rad) % 360
    
    # Calculate elevation (in degrees, -90 to 90, 0 is horizontal, 90 is straight up)
    if distance > 0:
        elevation_rad = np.arcsin(z / distance)
        elevation_deg = np.degrees(elevation_rad)
    else:
        elevation_deg = 0.0
    
    return azimuth_deg, elevation_deg, distance


class NumpyArrayModel(BaseModel):
    """Base model that handles numpy array serialization."""
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        }
    
    def model_dump_yaml(self, **kwargs) -> str:
        """Dump model to YAML string."""
        return yaml.safe_dump(self.model_dump(mode='json', **kwargs), sort_keys=False)
    
    @classmethod
    def model_validate_yaml(cls, yaml_str: str) -> 'NumpyArrayModel':
        """Load model from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)
    
    def save_to_yaml(self, file_path: Union[str, os.PathLike]) -> None:
        """Save model to YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.safe_dump(
                self.model_dump(mode='json'),
                f,
                sort_keys=False,
                default_flow_style=None
            )


class RoomDimensions(NumpyArrayModel):
    """Represents the dimensions of a room in 3D space."""
    width: float = Field(..., gt=0, description="Width of the room (x-axis) in meters")
    length: float = Field(..., gt=0, description="Length of the room (y-axis) in meters")
    height: float = Field(..., gt=0, description="Height of the room (z-axis) in meters")
    
    def to_array(self) -> np.ndarray:
        """Convert dimensions to numpy array [width, length, height]."""
        return np.array([self.width, self.length, self.height])

    @classmethod
    def create_example(cls) -> 'RoomDimensions':
        """Create an example room configuration."""
        return cls(width=75.0, length=50.0, height=30.0)


class VirtualSpeaker(NumpyArrayModel):
    """Represents a virtual speaker in 3D space."""
    position: np.ndarray = Field(..., description="3D position [x, y, z] in meters")
    direction: Optional[np.ndarray] = Field(
        None,
        description="Optional direction vector [dx, dy, dz] (will be normalized)"
    )
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        }
    )
    
    @field_validator('position', mode='before')
    @classmethod
    def convert_position_to_np_array(cls, v):
        if not isinstance(v, np.ndarray):
            return np.array(v, dtype=float)
        return v.astype(float)
    
    @field_validator('direction', mode='before')
    @classmethod
    def validate_direction(cls, v, info):
        if v is not None:
            v = np.array(v, dtype=float) if not isinstance(v, np.ndarray) else v.astype(float)
            norm = np.linalg.norm(v)
            if norm > 0:
                return v / norm
        return None
    
    @classmethod
    def from_direction(
        cls,
        origin: np.ndarray,
        direction: np.ndarray,
        distance: float
    ) -> 'VirtualSpeaker':
        """Create a speaker at a given distance from origin in the specified direction."""
        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        position = origin + direction * distance
        return cls(position=position, direction=direction)
    
    @classmethod
    def create_example(cls, origin: np.ndarray = None) -> 'VirtualSpeaker':
        """Create an example speaker configuration."""
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0])
        return cls.from_direction(
            origin=origin,
            direction=np.array([1.0, 1.0, 1.0]),
            distance=25.0
        )


class SceneVisualizer:
    """Visualizes the digital twin scene with room, microphone array, and speakers."""
    
    def __init__(self, room_dims: RoomDimensions):
        """Initialize the visualizer with room dimensions."""
        self.room_dims = room_dims
        self.fig = None
        self.ax = None
    
    def setup_plot(self):
        """Set up the 3D plot with room dimensions."""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels and limits
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_zlabel('z [m]')
        
        dims = self.room_dims.to_array()
        self.ax.set_xlim(0, dims[0])
        self.ax.set_ylim(0, dims[1])
        self.ax.set_zlim(0, dims[2])
        self.ax.set_box_aspect([1, 1, dims[2]/max(dims[:2])])
        
        # Draw room wireframe (grey cube)
        for x in [0, dims[0]]:
            for y in [0, dims[1]]:
                self.ax.plot([x, x], [y, y], [0, dims[2]], color='grey', lw=0.5)
        for x in [0, dims[0]]:
            for z in [0, dims[2]]:
                self.ax.plot([x, x], [0, dims[1]], [z, z], color='grey', lw=0.5)
        for y in [0, dims[1]]:
            for z in [0, dims[2]]:
                self.ax.plot([0, dims[0]], [y, y], [z, z], color='grey', lw=0.5)
    
    def _add_azimuth_elevation_triangles(self, origin: np.ndarray, target: np.ndarray, radius: float):
        """Add triangles showing azimuth and elevation from origin to target."""
        # Calculate azimuth and elevation
        azim_deg, elev_deg, distance = calculate_azimuth_elevation(origin, target)
        
        # Convert to radians for calculations
        azim_rad = np.radians(azim_deg)
        elev_rad = np.radians(elev_deg)
        
        # Projection radius (scaled for visualization)
        proj_radius = min(radius * 0.6, distance * 0.6)
        
        # Calculate points for azimuth triangle (xy plane)
        azim_x = origin[0] + proj_radius * np.cos(azim_rad)
        azim_y = origin[1] + proj_radius * np.sin(azim_rad)
        
        # Calculate points for elevation triangle (vertical plane)
        elev_xy_dist = proj_radius * np.cos(elev_rad)
        elev_x = origin[0] + elev_xy_dist * np.cos(azim_rad)
        elev_y = origin[1] + elev_xy_dist * np.sin(azim_rad)
        elev_z = origin[2] + proj_radius * np.sin(elev_rad)
        
        # Draw azimuth triangle (xy plane)
        self.ax.plot(
            [origin[0], azim_x, azim_x, origin[0]],
            [origin[1], origin[1], azim_y, origin[1]],
            [origin[2], origin[2], origin[2], origin[2]],
            'b--', alpha=0.7, linewidth=1
        )
        
        # Draw elevation triangle (vertical plane)
        self.ax.plot(
            [origin[0], elev_x, elev_x, origin[0]],
            [origin[1], elev_y, elev_y, origin[1]],
            [origin[2], origin[2], elev_z, origin[2]],
            'g--', alpha=0.7, linewidth=1
        )
        
        # Add angle labels
        label_offset = proj_radius * 0.3
        self.ax.text(
            origin[0] + label_offset * np.cos(azim_rad/2),
            origin[1] + label_offset * np.sin(azim_rad/2),
            origin[2],
            f"Az: {azim_deg:.1f}°",
            color='blue', fontsize=8
        )
        
        self.ax.text(
            origin[0] + label_offset * np.cos(azim_rad) * 0.7,
            origin[1] + label_offset * np.sin(azim_rad) * 0.7,
            origin[2] + label_offset * 0.5,
            f"El: {elev_deg:.1f}°",
            color='green', fontsize=8
        )
        
        return azim_deg, elev_deg, distance
    
    def add_microphone_array(self, mic_array: MicrophoneArray, mic_center: np.ndarray):
        """Add a microphone array to the plot."""
        self.mic_center = np.array(mic_center, dtype=float)
        self.mic_array = mic_array
        
        # Plot mic center
        self.ax.scatter(*self.mic_center, color='black', s=50, label='Mic array center')
        
        # Plot array front direction (+x)
        self.ax.quiver(
            *self.mic_center, 5, 0, 0,
            color='red', linewidth=2, arrow_length_ratio=0.1,
            label='Array front (+x)'
        )
    
    def add_spherical_quadrant(
        self,
        center: np.ndarray,
        radius: float,
        color: str = 'yellow',
        alpha: float = 0.3,
        resolution: int = 30
    ):
        """
        Add a spherical quadrant (1/8th of a sphere) centered at the mic array.
        
        Args:
            center: Center point of the quadrant [x, y, z]
            radius: Radius of the spherical quadrant
            color: Color of the quadrant
            alpha: Transparency (0-1)
            resolution: Number of points for the mesh (higher = smoother)
        """
        # Create spherical coordinates for one octant (x>0, y>0, z>0)
        phi = np.linspace(0, np.pi/2, resolution)  # 0 to 90 degrees
        theta = np.linspace(0, np.pi/2, resolution)  # 0 to 90 degrees
        phi, theta = np.meshgrid(phi, theta)
        
        # Convert to Cartesian coordinates
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        # Create all 4 quadrants in the positive x half-space
        quadrants = [
            (x, y, z),                     # +y, +z
            (x, -y, z),                    # -y, +z
            (x, y, -z),                    # +y, -z
            (x, -y, -z)                    # -y, -z
        ]
        
        # Plot each quadrant
        for quad_x, quad_y, quad_z in quadrants:
            # Shift to the mic array center
            x_shifted = quad_x + center[0]
            y_shifted = quad_y + center[1]
            z_shifted = quad_z + center[2]
            
            self.ax.plot_surface(
                x_shifted, y_shifted, z_shifted,
                color=color,
                alpha=alpha,
                linewidth=0,
                antialiased=False,
                shade=True
            )
    
    def add_speaker(self, speaker: VirtualSpeaker, color: str = 'green', label: str = 'Virtual speaker'):
        """Add a speaker to the plot with azimuth and elevation visualization."""
        speaker_pos = np.array(speaker.position, dtype=float)
        
        # Store speaker info for legend
        self.speaker_info = {
            'position': speaker_pos,
            'color': color,
            'label': label,
            'direction': speaker.direction
        }
        
        # Plot the speaker
        self.ax.scatter(*speaker_pos, color=color, s=50, label=label)
        
        # Draw line from mic to speaker
        self.ax.plot(
            [self.mic_center[0], speaker_pos[0]],
            [self.mic_center[1], speaker_pos[1]],
            [self.mic_center[2], speaker_pos[2]],
            'r-', alpha=0.5, linewidth=1
        )
        
        # Add azimuth and elevation triangles and get the values
        azim_deg, elev_deg, distance = self._add_azimuth_elevation_triangles(
            self.mic_center, speaker_pos, radius=25.0
        )
        
        # Add distance label near the speaker
        self.ax.text(
            speaker_pos[0], speaker_pos[1], speaker_pos[2] + 1.0,
            f"{distance:.1f}m",
            color='red', fontsize=8
        )
        
        # If direction is specified, draw an arrow
        if speaker.direction is not None:
            self.ax.quiver(
                *speaker_pos, *speaker.direction * 2,
                color='blue', linewidth=2, arrow_length_ratio=0.1,
                label=f'{label} direction'
            )
        
        return azim_deg, elev_deg, distance
    
    def show(self, title: str = 'Digital Twin Scene'):
        """Display the plot with enhanced legend."""
        # Create custom legend entries
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Mic Array Center'),
            plt.Line2D([0], [0], color='red', lw=2, label='Array Front (+X)'),
            plt.Line2D([0], [0], color='blue', lw=1, linestyle='--', label='Azimuth'),
            plt.Line2D([0], [0], color='green', lw=1, linestyle='--', label='Elevation'),
        ]
        
        # Add speaker info to legend if available
        if hasattr(self, 'speaker_info'):
            azim_deg, elev_deg, distance = calculate_azimuth_elevation(
                self.mic_center, self.speaker_info['position']
            )
            
            legend_elements.extend([
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.speaker_info['color'], 
                           markersize=8, label=f"{self.speaker_info['label']}"),
                plt.Line2D([0], [0], color='w', label=f"Distance: {distance:.1f}m"),
                plt.Line2D([0], [0], color='w', label=f"Azimuth: {azim_deg:.1f}°"),
                plt.Line2D([0], [0], color='w', label=f"Elevation: {elev_deg:.1f}°")
            ])
        
        # Add legend to the plot
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Set title and display
        self.ax.set_title(title)
        plt.tight_layout()
        plt.show()


class DigitalTwinConfig(NumpyArrayModel):
    """Configuration for the digital twin scene."""
    room: RoomDimensions = Field(default_factory=RoomDimensions.create_example)
    mic_center: np.ndarray = Field(default_factory=lambda: np.array([0.0, 25.0, 0.3048]))
    speaker: VirtualSpeaker = Field(default_factory=VirtualSpeaker.create_example)
    hemisphere_radius: float = 25.0
    view_limits: Dict[str, Tuple[float, float]] = Field(
        default_factory=lambda: {
            'x': (-5.0, 30.0),
            'y': (0.0, 50.0),
            'z': (0.0, 30.0)
        },
        description="View limits for the 3D plot"
    )
    
    @field_validator('mic_center', mode='before')
    @classmethod
    def validate_mic_center(cls, v):
        if not isinstance(v, np.ndarray):
            return np.array(v, dtype=float)
        return v.astype(float)
    
    @classmethod
    def load_from_yaml(cls, file_path: Union[str, os.PathLike]) -> 'DigitalTwinConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
    
    def save_to_yaml(self, file_path: Union[str, os.PathLike]) -> None:
        """Save configuration to YAML file."""
        super().save_to_yaml(file_path)
        print(f"Configuration saved to {file_path}")
    
    @classmethod
    def create_example(cls, save_path: Union[str, os.PathLike] = None) -> 'DigitalTwinConfig':
        """Create and optionally save an example configuration."""
        config = cls()
        if save_path:
            config.save_to_yaml(save_path)
        return config