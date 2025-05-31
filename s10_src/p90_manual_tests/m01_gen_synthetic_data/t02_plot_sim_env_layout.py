#!/usr/bin/env python3
"""
Visualize the digital twin environment layout.

This script loads the scene configuration from a YAML file and visualizes it
in a 3D plot using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the data models
from s10_src.p05_data_models.d03_digital_twin import (
    DigitalTwinConfig,
    MicrophoneArray,
    SceneVisualizer as ModelSceneVisualizer,
    calculate_azimuth_elevation
)


class SceneVisualizer(ModelSceneVisualizer):
    """Extended SceneVisualizer with additional visualization features."""
    
    def __init__(self, room_dims):
        """Initialize with room dimensions."""
        super().__init__(room_dims)
        self.legend_proxies = []
    
    def add_microphone_array(self, mic_array, mic_center):
        """Add microphone array to the plot."""
        super().add_microphone_array(mic_array, mic_center)
        # Add to legend
        from matplotlib.patches import Patch
        self.legend_proxies.append(
            Patch(facecolor='black', label='Mic array center')
        )
    
    def add_speaker(self, speaker, color='green', label='Speaker'):
        """Add speaker to the plot with direction indicator."""
        super().add_speaker(speaker, color, label)
        # Add to legend
        from matplotlib.patches import Patch
        self.legend_proxies.append(
            Patch(facecolor=color, label=label)
        )
    
    def show(self, title: str = "Digital Twin Scene"):
        """Show the plot with legend."""
        self.ax.set_title(title)
        
        # Add legend using proxy artists if they exist
        if self.legend_proxies:
            self.ax.legend(handles=self.legend_proxies)
            
        plt.tight_layout()
        plt.show()


def save_plot_with_angles(viz: SceneVisualizer, config: DigitalTwinConfig, azim_deg: float, elev_deg: float):
    """Save the plot with specific view angles."""
    # Set the view angles
    viz.ax.view_init(elev=elev_deg, azim=azim_deg)
    
    # Ensure the directory exists
    save_dir = Path(config.png_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Format the filename with angles
    azim_str = f"{int(round(azim_deg/5)*5):03d}"  # Round to nearest 5 degrees
    elev_str = f"{int(round(elev_deg/5)*5):03d}"  # Round to nearest 5 degrees
    filename = f"3d_acoustic_env_az{azim_str}_el{elev_str}.png"
    
    # Save the figure
    save_path = save_dir / filename
    viz.fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

def main():
    """Main function to run the visualization."""
    # Path to the configuration file
    config_path = Path("/home/tj/02_Windsurf_Projects/r03_Gimbal_Angle_Root/05_config/c10_digital_twin.yaml")
    
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
    triangle_radius = config.hemisphere_radius * 0.9
    viz._add_azimuth_elevation_triangles(
        config.mic_center, 
        config.speaker.position,
        radius=triangle_radius
    )
    
    # Set view limits
    viz.ax.set_xlim(*config.view_limits['x'])
    viz.ax.set_ylim(*config.view_limits['y'])
    viz.ax.set_zlim(*config.view_limits['z'])
    
    # Save the plot with different angles
    for azim in range(0, 360, 5):  # Azimuth from 0 to 355 degrees in 5-degree steps
        for elev in range(0, 91, 5):  # Elevation from 0 to 90 degrees in 5-degree steps
            save_plot_with_angles(viz, config, azim, elev)
    
    # Show one of the views interactively
    viz.ax.view_init(elev=30, azim=45)  # Default view
    viz.show("Digital Twin Environment Layout")


if __name__ == "__main__":
    main()