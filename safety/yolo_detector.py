# safety/yolo_detector.py

from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="models/yolo/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": self.model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()
                })

        return detections
