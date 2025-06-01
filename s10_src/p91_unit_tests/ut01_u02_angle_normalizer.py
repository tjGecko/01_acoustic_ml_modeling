import pytest
import numpy as np
from pathlib import Path
import sys

from s10_src.p20_ml_model.u02_angle_normalizer import AngleNormalizer

# Test cases for normalization
NORMALIZATION_CASES = [
    # (azimuth, elevation, expected_az_norm, expected_el_norm, az_range, el_range)
    # Default range tests (-90 to 90 for both)
    (0, 0, 0.0, 0.0, (-90, 90), (-90, 90)),  # Center
    (-90, -90, -1.0, -1.0, (-90, 90), (-90, 90)),  # Min azimuth and elevation
    (90, 90, 1.0, 1.0, (-90, 90), (-90, 90)),  # Max azimuth and elevation
    (45, -45, 0.5, -0.5, (-90, 90), (-90, 90)),  # Mid-range values
    
    # Custom range tests
    (0, 0, -1.0, -1.0, (0, 180), (0, 180)),  # Center of custom range
    (180, 180, 1.0, 1.0, (0, 180), (0, 180)),  # Max of custom range
    (90, 90, 0.0, 0.0, (0, 180), (0, 180)),  # Mid of custom range
    
    # Edge cases
    (0, 0, -1.0, -1.0, (0, 1), (0, 1)),  # Small range
    (1, 1, 1.0, 1.0, (0, 1), (0, 1)),  # Small range max
]

# Test cases for denormalization
DENORMALIZATION_CASES = [
    # (normalized_az, normalized_el, expected_az, expected_el, az_range, el_range)
    (0.0, 0.0, 0.0, 0.0, (-90, 90), (-90, 90)),  # Center
    (-1.0, -1.0, -90.0, -90.0, (-90, 90), (-90, 90)),  # Min
    (1.0, 1.0, 90.0, 90.0, (-90, 90), (-90, 90)),  # Max
    (0.5, -0.5, 45.0, -45.0, (-90, 90), (-90, 90)),  # Mid-range
]

# Round-trip test cases
ROUND_TRIP_CASES = [
    # (azimuth, elevation, az_range, el_range)
    (0, 0, (-90, 90), (-90, 90)),
    (-90, -90, (-90, 90), (-90, 90)),
    (90, 90, (-90, 90), (-90, 90)),
    (45, -45, (-90, 90), (-90, 90)),
    (0, 0, (0, 180), (0, 180)),
    (180, 180, (0, 180), (0, 180)),
    (90, 90, (0, 180), (0, 180)),
]


@pytest.mark.parametrize("az,el,exp_az_norm,exp_el_norm,az_range,el_range", NORMALIZATION_CASES)
def test_normalize(az, el, exp_az_norm, exp_el_norm, az_range, el_range):
    """Test normalization of angles to [-1, 1] range."""
    normalizer = AngleNormalizer(az_range=az_range, el_range=el_range)
    az_norm, el_norm = normalizer.normalize(az, el)
    
    assert np.isclose(az_norm, exp_az_norm, atol=1e-6), \
        f"Azimuth normalization failed for {az}° (expected {exp_az_norm}, got {az_norm})"
    assert np.isclose(el_norm, exp_el_norm, atol=1e-6), \
        f"Elevation normalization failed for {el}° (expected {exp_el_norm}, got {el_norm})"


@pytest.mark.parametrize("norm_az,norm_el,exp_az,exp_el,az_range,el_range", DENORMALIZATION_CASES)
def test_denormalize(norm_az, norm_el, exp_az, exp_el, az_range, el_range):
    """Test denormalization of values back to original angle ranges."""
    normalizer = AngleNormalizer(az_range=az_range, el_range=el_range)
    az, el = normalizer.denormalize(norm_az, norm_el)
    
    assert np.isclose(az, exp_az, atol=1e-6), \
        f"Azimuth denormalization failed for {norm_az} (expected {exp_az}°, got {az}°)"
    assert np.isclose(el, exp_el, atol=1e-6), \
        f"Elevation denormalization failed for {norm_el} (expected {exp_el}°, got {el}°)"


@pytest.mark.parametrize("az,el,az_range,el_range", ROUND_TRIP_CASES)
def test_round_trip(az, el, az_range, el_range):
    """Test that normalizing then denormalizing returns the original angles."""
    normalizer = AngleNormalizer(az_range=az_range, el_range=el_range)
    
    # Normalize then denormalize
    norm_az, norm_el = normalizer.normalize(az, el)
    result_az, result_el = normalizer.denormalize(norm_az, norm_el)
    
    # Check if we get back to the original values (within floating point precision)
    assert np.isclose(result_az, az, atol=1e-6), \
        f"Round-trip failed for azimuth: {az}° -> {result_az}°"
    assert np.isclose(result_el, el, atol=1e-6), \
        f"Round-trip failed for elevation: {el}° -> {result_el}°"


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Test with single-value range
    normalizer = AngleNormalizer(az_range=(0, 1), el_range=(0, 1))
    assert normalizer.normalize(0.5, 0.5) == (0.0, 0.0)
    
    # Test with reversed range
    normalizer = AngleNormalizer(az_range=(90, -90), el_range=(90, -90))
    assert normalizer.normalize(0, 0) == (0.0, 0.0)
    assert normalizer.normalize(90, 90) == (-1.0, -1.0)
    assert normalizer.normalize(-90, -90) == (1.0, 1.0)
    
    # Test denormalization with out-of-range values
    normalizer = AngleNormalizer()
    assert normalizer.denormalize(-2.0, 2.0) == (-180.0, 180.0)  # Outside [-1, 1] range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])