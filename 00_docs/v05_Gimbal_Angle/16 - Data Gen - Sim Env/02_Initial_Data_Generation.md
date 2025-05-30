# Synthetic Data Generation for Source Localization

This document provides a comprehensive overview of the synthetic data generation process used for training source localization networks in the Gimbal Angle project.

## Coordinate System and Geometry

### Microphone Array Configuration
- **Array Type**: UMA-16 MEMS microphone array (miniDSP)
- **Number of Mics**: 16
- **Array Configuration**: 2D planar array with mics arranged in a rectangular grid pattern
- **Array Center**: Positioned at [0.5, 25.0, 1.0] meters in the room coordinate system
- **Mic Positions**: Defined in `uma_16_mic_array.xml` with coordinates relative to array center
  - X-axis: Positive to the right (looking from array center to room center)
  - Y-axis: Positive forward (away from array)
  - Z-axis: Positive upward

### Room Configuration
- **Dimensions**: 50.0m (L) × 50.0m (W) × 30.0m (H)
- **Walls**: Absorptive material (panel_fabric_covered_8pcf)
- **Floor/Ceiling**: Same absorptive material as walls
- **Air Absorption**: Enabled to simulate frequency-dependent attenuation
- **Reflection Model**: Free-field condition (no reflections, max_order=0)

### Source Positioning System
- **Coordinate System**: Spherical coordinates relative to mic array center
  - **Azimuth (φ)**: 0° = front, 90° = right, 180° = back, 270° = left
  - **Elevation (θ)**: 0° = horizontal, +90° = directly above, -90° = directly below
- **Distance**: Fixed at 25.0m from mic array center
- **Grid Coverage**:
  - Azimuth: 0° to 355° in 5° steps
  - Elevation: 0° to 90° in 5° steps

## Acoustic Simulation Setup

### Simulation Parameters
- **Sampling Rate**: 16,000 Hz
- **Temperature**: Assumed room temperature (20°C, affects speed of sound)
- **Speed of Sound**: ~343 m/s at 20°C
- **Simulation Engine**: Pyroomacoustics
- **Reflection Model**: Free-field (direct path only)
- **Air Absorption**: Enabled with default parameters

### Signal Processing
- **Source Signals**: Real drone recordings (Bebop and Membo drones)
- **Signal Levels**: Normalized to ensure consistent loudness
- **Noise**: No additional noise added (clean signals)
- **File Format**: Output as 32-bit floating point WAV files

## Data Generation Workflow

### Input Data
1. **Source Audio**:
   - Clean drone recordings from Bebop and Membo drones
   - Stored in WAV format
   - Sampled at 16 kHz
   - Filtered to include only high-SNR segments

2. **Microphone Array Configuration**:
   - Defined in `uma_16_mic_array.xml`
   - Positions specified in meters relative to array center
   - Array is assumed to be mounted horizontally

### Simulation Process
1. For each source audio file:
   - Load the WAV file
   - For each azimuth/elevation combination:
     1. Calculate source position in room coordinates
     2. Verify position is within room boundaries
     3. Configure Pyroomacoustics simulation
     4. Simulate sound propagation
     5. Save multi-channel output

2. Output Naming Convention:
   `{source_name}_az{az:03d}_el{el:+03d}.npy`
   - `source_name`: Base name of source WAV file
   - `az`: Azimuth angle in degrees (000-355)
   - `el`: Elevation angle in degrees (-90 to +90)

## Key Assumptions

1. **Free-Field Assumption**:
   - No reflections or reverberation
   - Direct path only between source and each microphone
   - Valid for anechoic conditions or short time windows

2. **Source Characteristics**:
   - Point source radiation pattern
   - No near-field effects (far-field assumption)
   - Stationary source during each capture

3. **Microphone Array**:
   - Ideal omnidirectional response
   - No frequency-dependent directivity
   - Perfectly matched frequency response across mics
   - No self-noise or preamp noise

## Usage in Training

### Data Augmentation
- Multiple source positions provide natural data augmentation
- Covers full sphere around the array
- Enables learning of direction-dependent features

### Expected Model Input/Output
- **Input**: Multi-channel time series (16 channels × N samples)
- **Output**: Azimuth and elevation angles (or direction vector)

### Validation Split
- Recommend 80/20 train/validation split
- Ensure same source positions aren't in both sets
- Consider holding out entire source files for testing

## Future Enhancements

1. **Environmental Effects**:
   - Add background noise
   - Include room reflections
   - Model temperature variations

2. **Source Modeling**:
   - Near-field effects
   - Directional source patterns
   - Moving sources

3. **Sensor Imperfections**:
   - Mic frequency response
   - Self-noise
   - Gain mismatches

## Related Files
- `s10_src/m90_manual_tests/m01_gen_synthetic_data/t04_define_synthetic_data_capture.py`: Main data generation script
- `05_config/c11_syn_acoustic_data_gen.yaml`: Configuration file
- `05_config/uma_16_mic_array.xml`: Microphone array geometry
- `05_config/clean_wav_registry.json`: Source audio file registry