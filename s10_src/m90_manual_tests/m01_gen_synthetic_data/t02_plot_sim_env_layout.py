#!/usr/bin/env python3
"""
Visualize the digital twin environment layout.

This script loads the scene configuration from a YAML file and visualizes it
in a 3D plot using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Import the data models
from s10_src.m05_data_models.d03_digital_twin import (
    DigitalTwinConfig,
    MicrophoneArray,
    calculate_azimuth_elevation
)


class SceneVisualizer:
    """Class for visualizing the digital twin scene."""
    
    def __init__(self, room_dims):
        """Initialize the visualizer with room dimensions."""
        self.room_dims = room_dims
        self.fig = None
        self.ax = None
    
    def setup_plot(self):
        """Set up the 3D plot."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Set aspect ratio
        self.ax.set_box_aspect([1, 1, 0.5])
    
    def add_spherical_quadrant(
        self,
        center: np.ndarray,
        radius: float,
        color: str = 'yellow',
        alpha: float = 0.3,
        resolution: int = 30
    ):
        """
        Add a spherical quadrant (1/8th of a sphere) centered at the given point.
        
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
            # Shift to the center point
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
    
    def add_microphone_array(self, mic_array: MicrophoneArray, mic_center: np.ndarray):
        """Add microphone array to the plot."""
        self.ax.scatter(*mic_center, color='black', s=50, label='Mic array center')
        self.ax.quiver(
            *mic_center, 5, 0, 0,
            color='red', linewidth=2,
            arrow_length_ratio=0.1,
            label='Array front (+x)'
        )
    
    def add_speaker(self, speaker):
        """Add speaker to the plot with direction indicator."""
        self.ax.scatter(*speaker.position, color='green', s=100, label='Speaker')
        
        if speaker.direction is not None:
            # Draw direction vector
            self.ax.quiver(
                *speaker.position, *speaker.direction * 3,
                color='green', linewidth=2,
                arrow_length_ratio=0.1,
                label='Speaker direction'
            )
    
    def add_azimuth_elevation_triangles(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        radius: float = 10.0,
        color: str = 'red',
        alpha: float = 0.5
    ):
        """Add triangles showing azimuth and elevation angles."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Patch
        
        # Calculate azimuth and elevation
        azimuth_deg, elevation_deg, _ = calculate_azimuth_elevation(origin, target)
        
        # Convert to radians for calculations
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)
        
        # Create vertices for azimuth triangle (XY plane)
        # Points: origin, point on x-axis, point at angle in XY plane
        az_vertices = [
            [origin[0], origin[1], origin[2]],  # Origin
            [origin[0] + radius, origin[1], origin[2]],  # Along x-axis
            [origin[0] + radius, origin[1] + radius * np.tan(azimuth), origin[2]]  # At angle in XY plane
        ]
        
        # Create vertices for elevation triangle (XZ plane)
        # Points: origin, point on x-axis, point at angle in XZ plane
        el_vertices = [
            [origin[0], origin[1], origin[2]],  # Origin
            [origin[0] + radius, origin[1], origin[2]],  # Along x-axis
            [origin[0] + radius, origin[1], origin[2] + radius * np.tan(elevation)]  # At angle in XZ plane
        ]
        
        # Create triangle collections
        az_triangle = Poly3DCollection(
            [az_vertices],
            alpha=alpha,
            facecolors=color,
            linewidths=0.5,
            edgecolors='black'
        )
        
        el_triangle = Poly3DCollection(
            [el_vertices],
            alpha=alpha,
            facecolors='blue',
            linewidths=0.5,
            edgecolors='black'
        )
        
        # Add triangles to the plot
        self.ax.add_collection3d(az_triangle)
        self.ax.add_collection3d(el_triangle)
        
        # Create proxy artists for the legend
        az_proxy = Patch(facecolor=color, alpha=alpha, label=f'Azimuth: {azimuth_deg:.1f}°')
        el_proxy = Patch(facecolor='blue', alpha=alpha, label=f'Elevation: {elevation_deg:.1f}°')
        
        # Store proxies for legend
        if not hasattr(self, 'legend_proxies'):
            self.legend_proxies = []
        self.legend_proxies.extend([az_proxy, el_proxy])
        
        # Add angle labels
        self.ax.text(
            origin[0] + radius * 0.7,
            origin[1] + radius * np.tan(azimuth) * 0.35,
            origin[2],
            f'Az: {azimuth_deg:.1f}°',
            color=color,
            fontsize=8,
            fontweight='bold'
        )
        self.ax.text(
            origin[0] + radius * 0.7,
            origin[1],
            origin[2] + radius * np.tan(elevation) * 0.35,
            f'El: {elevation_deg:.1f}°',
            color='blue',
            fontsize=8,
            fontweight='bold'
        )
    
    def show(self, title: str = "Digital Twin Scene"):
        """Show the plot."""
        self.ax.set_title(title)
        
        # Add legend using proxy artists if they exist
        if hasattr(self, 'legend_proxies') and self.legend_proxies:
            self.ax.legend(handles=self.legend_proxies)
            
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the visualization."""
    # Path to the configuration file
    config_path = Path("05_config/c10_digital_twin.yaml")
    
    # Create example config if it doesn't exist
    if not config_path.exists():
        print(f"Creating example configuration at {config_path}")
        config = DigitalTwinConfig.create_example(save_path=config_path)
    else:
        # Load the configuration
        print(f"Loading configuration from {config_path}")
        config = DigitalTwinConfig.load_from_yaml(config_path)
    
    # Create visualizer
    viz = SceneVisualizer(config.room)
    viz.setup_plot()
    
    # Add elements to the plot
    viz.add_spherical_quadrant(config.mic_center, config.hemisphere_radius)
    
    # Add microphone array (using a dummy array for visualization)
    mic_array = MicrophoneArray(name="example_array", microphones=[])
    viz.add_microphone_array(mic_array, config.mic_center)
    
    # Add speaker
    viz.add_speaker(config.speaker)
    
    # Add azimuth and elevation triangles
    viz.add_azimuth_elevation_triangles(config.mic_center, config.speaker.position)
    
    # Set view limits
    viz.ax.set_xlim(*config.view_limits['x'])
    viz.ax.set_ylim(*config.view_limits['y'])
    viz.ax.set_zlim(*config.view_limits['z'])
    
    # Show the plot
    viz.show("Digital Twin Environment Layout")


if __name__ == "__main__":
    main()