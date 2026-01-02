# safety/frame_capture.py

import cv2

class Camera:
    def __init__(self, cam_id=0, width=1280, height=720):
        # Use DirectShow backend for better Windows compatibility
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"[Camera] ERROR: Could not open camera {cam_id}")
            # Try default backend as fallback
            self.cap = cv2.VideoCapture(cam_id)
        
        # Set higher resolution for better viewport
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] Resolution set to {int(actual_w)}x{int(actual_h)}")

    def stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
