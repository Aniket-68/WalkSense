# infrastructure/darkness_detector.py
"""
Darkness detection utility for WalkSense.
Determines if the camera view is too dark for reliable VLM processing.
"""

import cv2
import numpy as np
from loguru import logger


class DarknessDetector:
    """
    Detects if a significant portion of the camera view is too dark.
    """
    
    def __init__(self, darkness_threshold: int = 40, area_threshold: float = 0.75):
        """
        Initialize the darkness detector.
        
        Args:
            darkness_threshold: Pixel brightness value below which is considered "dark" (0-255).
            area_threshold: Fraction of the image that must be dark to trigger (0.0-1.0).
        """
        self.darkness_threshold = darkness_threshold
        self.area_threshold = area_threshold
        logger.info(f"DarknessDetector initialized: threshold={darkness_threshold}, area={area_threshold*100}%")
    
    def is_too_dark(self, frame) -> tuple[bool, float]:
        """
        Analyzes a frame to determine if it's too dark for VLM processing.
        
        Args:
            frame: OpenCV BGR image frame (numpy array).
            
        Returns:
            A tuple of (is_dark: bool, dark_percentage: float).
            - is_dark: True if the frame is too dark
            - dark_percentage: Percentage of the frame that is dark (0.0-1.0)
        """
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Count pixels below darkness threshold
        dark_pixels = np.sum(gray < self.darkness_threshold)
        total_pixels = gray.size
        
        # Calculate percentage of dark pixels
        dark_percentage = dark_pixels / total_pixels
        
        # Determine if too dark
        is_dark = dark_percentage >= self.area_threshold
        
        if is_dark:
            logger.debug(f"Frame is too dark: {dark_percentage*100:.1f}% dark pixels")
        
        return is_dark, dark_percentage
    
    def get_brightness_level(self, frame) -> str:
        """
        Returns a human-readable brightness level description.
        
        Args:
            frame: OpenCV BGR image frame.
            
        Returns:
            A string describing the brightness level.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 30:
            return "Very Dark"
        elif avg_brightness < 60:
            return "Dark"
        elif avg_brightness < 120:
            return "Dim"
        elif avg_brightness < 180:
            return "Normal"
        else:
            return "Bright"
