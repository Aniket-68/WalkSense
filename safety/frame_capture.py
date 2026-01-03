# safety/frame_capture.py

import cv2

class Camera:
    def __init__(self, cam_id=None, width=None, height=None):
        from utils.config_loader import Config
        
        # Determine source from Switch
        # We allow parameters to override the registry choice
        source_type = Config.get("active_registry.camera_source", "hardware") if cam_id is None else "hardware"
        
        if source_type == "simulation" and cam_id is None:
            video_path = Config.get("camera_registry.simulation.video_path")
            self.cap = cv2.VideoCapture(video_path)
            self.loop = Config.get("camera_registry.simulation.loop", True)
            self.source_path = video_path
            print(f"[Camera] Simulation Mode: Reading from {video_path}")
        else:
            # Hardware mode
            cam_id = Config.get("camera_registry.hardware.id", 0)
            width = Config.get("camera_registry.hardware.width", 1280)
            height = Config.get("camera_registry.hardware.height", 720)

            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_id)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"[Camera] Hardware Mode: Device {cam_id}")

        self.source_type = source_type

    def stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if hasattr(self, "loop") and self.loop:
                    # Restart video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield frame

    def release(self):
        self.cap.release()
