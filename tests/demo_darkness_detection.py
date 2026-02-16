# tests/demo_darkness_detection.py
"""
Visual demonstration of darkness detection feature.
Creates sample frames and shows how the system responds.
"""

import cv2
import numpy as np
from infrastructure.darkness_detector import DarknessDetector


def create_demo_frames():
    """Create a series of demo frames with varying darkness levels."""
    frames = []
    
    # Frame 1: Normal bright scene
    bright = np.ones((480, 640, 3), dtype=np.uint8) * 180
    frames.append(("Normal Bright Scene", bright))
    
    # Frame 2: Dim scene
    dim = np.ones((480, 640, 3), dtype=np.uint8) * 80
    frames.append(("Dim Scene", dim))
    
    # Frame 3: 50% dark (should NOT trigger)
    half_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    half_dark[:, 320:, :] = 150
    frames.append(("50% Dark (Should NOT Trigger)", half_dark))
    
    # Frame 4: 75% dark (threshold - SHOULD trigger)
    threshold_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    threshold_dark[:, 480:, :] = 150
    frames.append(("75% Dark (THRESHOLD - Triggers)", threshold_dark))
    
    # Frame 5: 90% dark (SHOULD trigger)
    very_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    very_dark[:, 576:, :] = 150
    frames.append(("90% Dark (Triggers)", very_dark))
    
    # Frame 6: Completely dark
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    frames.append(("Completely Dark (Triggers)", black))
    
    return frames


def main():
    """Run the demonstration."""
    detector = DarknessDetector(darkness_threshold=40, area_threshold=0.75)
    frames = create_demo_frames()
    
    print("\n" + "=" * 70)
    print("DARKNESS DETECTION DEMONSTRATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Darkness Threshold: {detector.darkness_threshold} (pixels below this are 'dark')")
    print(f"  - Area Threshold: {detector.area_threshold * 100}% (triggers when this % is dark)")
    print("=" * 70 + "\n")
    
    for i, (name, frame) in enumerate(frames, 1):
        is_dark, dark_percentage = detector.is_too_dark(frame)
        brightness_level = detector.get_brightness_level(frame)
        
        # Add text overlay to the frame
        display_frame = frame.copy()
        
        # Background for text
        cv2.rectangle(display_frame, (10, 10), (630, 150), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (630, 150), (255, 255, 255), 2)
        
        # Title
        cv2.putText(display_frame, name, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stats
        status_color = (0, 0, 255) if is_dark else (0, 255, 0)
        status_text = "TOO DARK - VLM BLOCKED" if is_dark else "OK - VLM ALLOWED"
        
        cv2.putText(display_frame, f"Status: {status_text}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(display_frame, f"Dark Pixels: {dark_percentage*100:.1f}%", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Brightness: {brightness_level}", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Console output
        print(f"Frame {i}: {name}")
        print(f"  Status: {'🚫 TOO DARK' if is_dark else '✓ OK'}")
        print(f"  Dark Percentage: {dark_percentage*100:.1f}%")
        print(f"  Brightness Level: {brightness_level}")
        print(f"  VLM Processing: {'BLOCKED' if is_dark else 'ALLOWED'}")
        
        if is_dark:
            print(f"  → User Message: 'Your view is too dark ({dark_percentage*100:.0f}% dark). Please move to a brighter area.'")
        
        print()
        
        # Save the frame
        output_path = f"tests/resources/darkness_demo_frame_{i}.png"
        cv2.imwrite(output_path, display_frame)
        print(f"  Saved: {output_path}\n")
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("  - Frames 1-3: Normal/dim lighting - VLM processing allowed")
    print("  - Frames 4-6: Too dark (≥75%) - VLM processing blocked")
    print("  - User is notified when environment is too dark")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
