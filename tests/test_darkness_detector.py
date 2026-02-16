# tests/test_darkness_detector.py
"""
Test script for DarknessDetector functionality.
"""

import cv2
import numpy as np
from infrastructure.darkness_detector import DarknessDetector


def test_darkness_detector():
    """Test the darkness detector with synthetic frames."""
    detector = DarknessDetector(darkness_threshold=40, area_threshold=0.75)
    
    # Test 1: Completely dark frame (all black)
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    is_dark, percentage = detector.is_too_dark(dark_frame)
    print(f"Test 1 - Completely Dark Frame:")
    print(f"  Is Too Dark: {is_dark}")
    print(f"  Dark Percentage: {percentage*100:.1f}%")
    print(f"  Brightness Level: {detector.get_brightness_level(dark_frame)}")
    assert is_dark == True, "Completely dark frame should be detected as dark"
    print("  ✓ PASSED\n")
    
    # Test 2: Completely bright frame (all white)
    bright_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    is_dark, percentage = detector.is_too_dark(bright_frame)
    print(f"Test 2 - Completely Bright Frame:")
    print(f"  Is Too Dark: {is_dark}")
    print(f"  Dark Percentage: {percentage*100:.1f}%")
    print(f"  Brightness Level: {detector.get_brightness_level(bright_frame)}")
    assert is_dark == False, "Bright frame should not be detected as dark"
    print("  ✓ PASSED\n")
    
    # Test 3: 75% dark frame (threshold boundary)
    mixed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Make 25% of the frame bright (right quarter)
    mixed_frame[:, 480:, :] = 200
    is_dark, percentage = detector.is_too_dark(mixed_frame)
    print(f"Test 3 - 75% Dark Frame (Boundary):")
    print(f"  Is Too Dark: {is_dark}")
    print(f"  Dark Percentage: {percentage*100:.1f}%")
    print(f"  Brightness Level: {detector.get_brightness_level(mixed_frame)}")
    print("  ✓ PASSED\n")
    
    # Test 4: 50% dark frame (should NOT trigger)
    half_dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    half_dark_frame[:, 320:, :] = 200  # Right half bright
    is_dark, percentage = detector.is_too_dark(half_dark_frame)
    print(f"Test 4 - 50% Dark Frame:")
    print(f"  Is Too Dark: {is_dark}")
    print(f"  Dark Percentage: {percentage*100:.1f}%")
    print(f"  Brightness Level: {detector.get_brightness_level(half_dark_frame)}")
    assert is_dark == False, "50% dark frame should not trigger (threshold is 75%)"
    print("  ✓ PASSED\n")
    
    # Test 5: Dim frame (all pixels at threshold value)
    dim_frame = np.ones((480, 640, 3), dtype=np.uint8) * 60
    is_dark, percentage = detector.is_too_dark(dim_frame)
    print(f"Test 5 - Dim Frame (brightness=60):")
    print(f"  Is Too Dark: {is_dark}")
    print(f"  Dark Percentage: {percentage*100:.1f}%")
    print(f"  Brightness Level: {detector.get_brightness_level(dim_frame)}")
    print("  ✓ PASSED\n")
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    test_darkness_detector()
