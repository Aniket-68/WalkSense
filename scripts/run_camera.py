# scripts/run_camera.py

import cv2
import time

from safety.frame_capture import Camera
from safety.yolo_detector import YoloDetector
from safety.safety_rules import SafetyRules

from inference.fusion_engine import FusionEngine
from interaction.listening_layer import ListeningLayer
from audio.tts import TTSEngine

from reasoning.qwen_vlm import QwenVLM
from utils.frame_sampler import FrameSampler


# =========================================================
# VISUALIZATION ONLY (NOT SAFETY LOGIC)
# =========================================================

def hazard_color(label):
    """
    UI-only hazard coloring.
    Does NOT influence safety decisions.
    """
    label = label.lower()

    if label in {
        "knife", "gun", "fire", "stairs",
        "car", "bus", "truck", "bike"
    }:
        return (0, 0, 255)      # Red (critical)

    if label in {
        "person", "dog", "animal",
        "bicycle", "wall", "glass"
    }:
        return (0, 255, 255)    # Yellow (warning)

    if label in {"chair", "table", "bag"}:
        return (255, 0, 0)      # Blue (info)

    return (0, 255, 0)          # Green (safe)


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"][0])
        label = d["label"]
        conf = d["confidence"]

        color = hazard_color(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1 - 22),
            (x1 + len(text) * 9, y1),
            color,
            -1
        )

        cv2.putText(
            frame, text, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1
        )

    return frame


def draw_overlay(frame, status, description):
    h, w, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        frame, f"STATUS: {status}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, f"Description: {description}",
        (10, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )

    cv2.putText(
        frame, "Press 'S' to START | 'Q' to QUIT",
        (10, h - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (200, 200, 200), 1
    )

    return frame


# =========================================================
# MAIN
# =========================================================

def main():
    # =========================================================
    # CONFIGURATION - Change these settings as needed
    # =========================================================
    
    # Load central configuration
    from utils.config_loader import Config
    
    # Configuration
    QWEN_PROVIDER = Config.get("vlm.provider", "lm_studio")
    LM_STUDIO_URL = Config.get("vlm.url", "http://localhost:1234/v1")
    QWEN_MODEL_ID = Config.get("vlm.model_id", "qwen/qwen3-vl-4b")
    
    SAMPLING = Config.get("perception.sampling_interval", 150)
    SCENE_THRESH = Config.get("perception.scene_threshold", 0.15)
    
    camera = Camera()
    detector = YoloDetector()
    safety = SafetyRules()
    
    tts = TTSEngine()
    fusion = FusionEngine(tts)
    listener = ListeningLayer(None, fusion)
    
    print("[INIT] Loading Qwen...")
    qwen = QwenVLM(
        backend=QWEN_PROVIDER,
        model_id=QWEN_MODEL_ID,
        lm_studio_url=LM_STUDIO_URL
    )
    
    # Slow down Qwen to prevents audio spam (every 5 seconds approx)
    sampler = FrameSampler(every_n_frames=SAMPLING)

    started = False
    
    # --- ASYNC WORKER SETUP ---
    import threading
    import queue
    
    class QwenWorker:
        def __init__(self, qwen_instance):
            self.qwen = qwen_instance
            self.input_queue = queue.Queue(maxsize=1) # Only keep latest frame
            self.output_queue = queue.Queue()
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.is_busy = False
            
        def _run(self):
            while self.running:
                try:
                    # Wait for frame, timeout to check running flag
                    item = self.input_queue.get(timeout=0.1)
                    frame, context_str = item
                    
                    self.is_busy = True
                    try:
                        # Run inference (blocking call, but in this thread)
                        start = time.time()
                        desc = self.qwen.describe_scene(frame, context=context_str)
                        duration = time.time() - start
                        
                        self.output_queue.put((desc, duration))
                    except Exception as e:
                        print(f"[WORKER ERROR] {e}")
                    finally:
                        self.is_busy = False
                        self.input_queue.task_done()
                        
                except queue.Empty:
                    continue
                    
        def process(self, frame, context_str):
            # Only add if not busy and queue empty (drop old frames)
            if not self.is_busy and self.input_queue.empty():
                self.input_queue.put((frame.copy(), context_str))
                return True
            return False
            
        def get_result(self):
            try:
                return self.output_queue.get_nowait()
            except queue.Empty:
                return None
                
        def stop(self):
            self.running = False
            
    # Initialize Worker
    qwen_worker = QwenWorker(qwen)

    description = "System idle. Press S to start."

    # Setup window with larger size
    cv2.namedWindow("WalkSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WalkSense", 1280, 720)

    # Smart Scene Detector
    from utils.scene_change import SceneChangeDetector
    scene_detector = SceneChangeDetector(threshold=SCENE_THRESH) # Use value from config

    current_user_query = None

    print("[WalkSense] Running")

    for frame in camera.stream():
        detections = detector.detect(frame)
        frame = draw_detections(frame, detections)

        # 🔴 SAFETY (AUTHORITATIVE)
        safety_result = safety.evaluate(detections)
        if safety_result:
            alert_type, message = safety_result
            fusion.handle_safety_alert(message, alert_type)
            description = message

        # 🟡 QWEN (BEST-EFFORT - ASYNC)
        is_critical = safety_result and safety_result[0] == "CRITICAL_ALERT"
        
        if started and not is_critical:
            # 1. Check for new results from worker
            result = qwen_worker.get_result()
            if result:
                new_desc, duration = result
                print(f"[QWEN] {duration:.2f}s | {new_desc}")
                description = new_desc
                
                if current_user_query:
                    fusion.handle_user_query_response(description)
                    current_user_query = None
                else:
                    fusion.handle_scene_description(description)
            
            # 2. DECIDE TO RUN AI
            # Only run if:
            # A) User asked a question (Priority)
            # B) Time passed AND Scene has Changed (Passive)
            
            has_query = (current_user_query is not None)
            time_to_sample = sampler.should_sample()
            
            # Efficient Check: Don't run histogram if not even time to sample
            should_run_qwen = False
            
            if has_query:
                should_run_qwen = True
            elif time_to_sample:
                # Check for visual change
                if scene_detector.has_changed(frame):
                    should_run_qwen = True
            
            if should_run_qwen:
                # Format context strings from detections
                context_str = ", ".join([d["label"] for d in detections]) if detections else ""
                
                # Pass query if it exists
                if has_query:
                    context_str += f". USER QUESTION: {current_user_query}"
                
                if qwen_worker.process(frame, context_str):
                    # Visual feedback
                    cv2.putText(frame, "Thinking...", (frame.shape[1]-120, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if has_query:
                         cv2.putText(frame, "Targeting Query...", (frame.shape[1]-200, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status = "RUNNING" if started else "WAITING"
        frame = draw_overlay(frame, status, description)

        cv2.imshow("WalkSense", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('S')):
            started = True
            tts.speak("WalkSense started")

        if key in (ord('l'), ord('L')):
            # Push-to-Talk
            print("[WalkSense] key 'L' pressed - Listening...")
            # Capture query directly here
            query = listener.stt.listen_once()
            if query:
                print(f"[WalkSense] User asked: {query}")
                current_user_query = query
                fusion.handle_user_query(query) # Acknowledge receipt

        if key in (ord('m'), ord('M')):
            # Toggle Mute
            is_muted = fusion.router.toggle_mute()
            state = "MUTED" if is_muted else "ACTIVE"
            print(f"[WalkSense] Audio {state}")

        if key in (ord('q'), ord('Q')):
            break

    qwen_worker.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
