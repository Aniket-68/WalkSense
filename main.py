# main.py
from input.camera import Camera
from input.frame_sampler import FrameSampler
from perception.yolo_detector import YoloDetector
from perception.hazard_rules import SafetyRules
from orchestration.fusion_engine import FusionEngine
from audio.tts import speak

def main():
    camera = Camera()
    sampler = FrameSampler(sample_every_n_frames=15)

    yolo = YoloDetector()
    safety = SafetyRules()
    fusion = FusionEngine()

    print("[WalkSense] System started")

    for frame in camera.stream():
        # 🔴 SAFETY FLOW (ALWAYS ON)
        detections = yolo.detect(frame)
        alert = safety.evaluate(detections)

        if alert:
            speak(alert)
            continue  # SAFETY OVERRIDES EVERYTHING

        # 🟡 VISION REASONING FLOW (BEST EFFORT)
        if sampler.should_sample():
            fusion.process_scene(frame, detections)

if __name__ == "__main__":
    main()
