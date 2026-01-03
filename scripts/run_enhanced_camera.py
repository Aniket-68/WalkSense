# scripts/run_enhanced_camera.py
"""
Enhanced WalkSense demo with:
- Spatial-temporal object tracking
- LLM-based query answering
- Context-aware reasoning
"""

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
from utils.scene_change import SceneChangeDetector

import threading
import queue


# =========================================================
# VISUALIZATION (Same as before)
# =========================================================

def hazard_color(label):
    label = label.lower()
    if label in {"knife", "gun", "fire", "stairs", "car", "bus", "truck", "bike"}:
        return (0, 0, 255)
    if label in {"person", "dog", "animal", "bicycle", "wall", "glass"}:
        return (0, 255, 255)
    if label in {"chair", "table", "bag"}:
        return (255, 0, 0)
    return (0, 255, 0)


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"][0])
        label = d["label"]
        conf = d["confidence"]
        color = hazard_color(label)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(text) * 9, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def draw_overlay(frame, status, description, spatial_summary, vlm_ok=False, llm_ok=False):
    h, w, _ = frame.shape
    
    # Status bar
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {status}", (10, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # AI HEALTH INDICATORS (Status Dots)
    # VLM Indicator
    vlm_color = (0, 255, 0) if vlm_ok else (0, 0, 255)
    cv2.circle(frame, (w-40, 20), 8, vlm_color, -1)
    cv2.putText(frame, "VLM", (w-90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # LLM Indicator
    llm_color = (0, 255, 0) if llm_ok else (0, 0, 255)
    cv2.circle(frame, (w-130, 20), 8, llm_color, -1)
    cv2.putText(frame, "LLM", (w-180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Spatial summary (NEW)
    cv2.rectangle(frame, (0, 40), (w, 75), (40, 40, 40), -1)
    cv2.putText(frame, f"Tracking: {spatial_summary}", (10, 63),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Description + Controls
    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Description: {description}", (10, h - 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "S: START | L: ASK QUESTION | M: MUTE | Q: QUIT",
               (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


# =========================================================
# ASYNC WORKER (Same as before)
# =========================================================

class QwenWorker:
    def __init__(self, qwen_instance):
        self.qwen = qwen_instance
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.is_busy = False
        
    def _run(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                frame, context_str = item
                
                self.is_busy = True
                try:
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


# =========================================================
# MAIN
# =========================================================

def main():
    # Load central configuration
    from utils.config_loader import Config
    
    # 🟢 Camera Configuration
    CAMERA_SOURCE = Config.get("camera.source", 0)
    
    # 🟢 VLM Configuration
    VLM_PROVIDER = Config.get("vlm.active_provider", "lm_studio")
    vlm_config_path = f"vlm.providers.{VLM_PROVIDER}"
    VLM_URL = Config.get(f"{vlm_config_path}.url")
    VLM_MODEL = Config.get(f"{vlm_config_path}.model_id")
    
    # 🟢 VLM Input Mode & Frame Rate (NEW)
    VLM_INPUT_MODE = Config.get("vlm.input_mode", "yolo")  # "direct" or "yolo"
    VLM_FPS = Config.get("vlm.frames_per_second", 1.0)      # Frames per second to VLM
    VLM_INCLUDE_YOLO = Config.get("vlm.include_yolo_context", True)  # Include YOLO labels as text
    VLM_FRAME_INTERVAL = 1.0 / VLM_FPS  # Convert FPS to interval in seconds
    
    print(f"[CONFIG] VLM Mode: {VLM_INPUT_MODE} | FPS: {VLM_FPS} | YOLO Context: {VLM_INCLUDE_YOLO}")
    
    # 🟢 LLM Configuration
    LLM_PROVIDER = Config.get("llm.active_provider", "ollama")
    llm_config_path = f"llm.providers.{LLM_PROVIDER}"
    LLM_URL = Config.get(f"{llm_config_path}.url")
    LLM_MODEL = Config.get(f"{llm_config_path}.model_id")
    
    # 🟢 TTS Configuration handled internally in audio/speak.py via config.json
    
    # 🟢 Perception Thresholds
    SAMPLING = Config.get("perception.sampling_interval", 150)
    SCENE_THRESH = Config.get("perception.scene_threshold", 0.15)
    WINDOW_WIDTH = Config.get("camera.hardware.width", 1280)
    WINDOW_HEIGHT = Config.get("camera.hardware.height", 720)
    
    camera = Camera()
    detector = YoloDetector()
    safety = SafetyRules()
    
    tts = TTSEngine()
    
    # Enhanced Fusion Engine with LLM
    print("[INIT] Creating Enhanced Fusion Engine...")
    fusion = FusionEngine(
        tts, 
        llm_backend=LLM_PROVIDER, 
        llm_url=LLM_URL,
        llm_model=LLM_MODEL
    )
    
    # Interaction Layer
    listener = ListeningLayer(None, fusion)
    
    print("[INIT] Loading Qwen VLM...")
    qwen = QwenVLM(
        backend=VLM_PROVIDER,
        model_id=VLM_MODEL,
        lm_studio_url=VLM_URL
    )
    
    sampler = FrameSampler(every_n_frames=SAMPLING)
    scene_detector = SceneChangeDetector(threshold=SCENE_THRESH)
    qwen_worker = QwenWorker(qwen)
    
    # UI Variables
    started = False
    description = "System idle. Press S to start."
    current_user_query = None
    llm_response = ""
    llm_response_timer = 0
    is_listening = False
    last_vlm_time = 0  # For FPS-based VLM control

    cv2.namedWindow("WalkSense Enhanced", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WalkSense Enhanced", WINDOW_WIDTH, WINDOW_HEIGHT)
    

    # Threaded Listener Wrapper
    def threaded_listen():
        nonlocal current_user_query, is_listening
        is_listening = True
        print("[WalkSense] Listening thread started...")
        query = listener.stt.listen_once()
        if query:
            print(f"[WalkSense] User asked: {query}")
            current_user_query = query
            fusion.handle_user_query(query)
        is_listening = False

    for frame in camera.stream():
        current_time = time.time()
        
        # YOLO Detection
        detections = detector.detect(frame)
        frame = draw_detections(frame, detections)
        
        # 🔴 SAFETY LAYER
        safety_result = safety.evaluate(detections)
        if safety_result:
            alert_type, message = safety_result
            fusion.handle_safety_alert(message, alert_type)
            description = f"ALERT: {message}"
        
        # 🟢 SPATIAL TRACKING
        if started:
            fusion.update_spatial_context(detections, current_time, frame.shape[1])
        
        # 🟡 VLM & LLM PROCESSING
        is_critical = safety_result and safety_result[0] == "CRITICAL_ALERT"
        
        if started and not is_critical:
            # 1. Harvest Results
            result = qwen_worker.get_result()
            if result:
                new_desc, duration = result
                print(f"[WalkSense] VLM: {new_desc} ({duration:.1f}s)")
                description = new_desc
                
                # Retrieve LLM answer if one was generated during this cycle
                # We need to ask FusionEngine if it has a pending answer
                # (Ideally FusionEngine returns it, or we intercept it via a callback)
                # For now, we'll check if we had a query pending
                if current_user_query:
                    # Trigger LLM explicitly here if Fusion doesn't auto-return
                    # But Fusion.handle_vlm_description calls LLM internally if query exists
                    answer = fusion.handle_vlm_description(new_desc)
                    if answer:
                        llm_response = f"AI: {answer}"
                        llm_response_timer = time.time() + 10 # Show for 10s
                        print(f"[WalkSense] LLM Answer: {answer}")
                        # Clear query now that it's answered
                        current_user_query = None
                else:
                    fusion.handle_scene_description(new_desc)
            
            # 2. VLM Trigger Logic (FPS-based)
            has_query = (current_user_query is not None)
            time_since_vlm = current_time - last_vlm_time
            vlm_time_ready = time_since_vlm >= VLM_FRAME_INTERVAL
            
            should_run_qwen = False
            
            if has_query:
                should_run_qwen = True  # User query always triggers VLM
            elif vlm_time_ready and VLM_INPUT_MODE == "direct":
                # Direct mode: send frame to VLM at configured FPS
                should_run_qwen = True
            elif vlm_time_ready and VLM_INPUT_MODE == "yolo":
                # YOLO mode: only trigger if scene changed
                if scene_detector.has_changed(frame):
                    should_run_qwen = True
            
            if should_run_qwen:
                # Build context string based on config
                if VLM_INCLUDE_YOLO and detections:
                    context_str = ", ".join([d["label"] for d in detections])
                else:
                    context_str = ""  # Direct mode - no YOLO context
                
                if has_query:
                    context_str += f". USER QUESTION: {current_user_query}" if context_str else f"USER QUESTION: {current_user_query}"
                
                # Send to worker (non-blocking)
                if qwen_worker.process(frame, context_str):
                    last_vlm_time = current_time  # Update timer
                    status_text = "Reasoning..." if has_query else "Scanning..."
                    cv2.putText(frame, status_text, (frame.shape[1]-150, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # UI Overlay Construction
        status = "RUNNING" if started else "PAUSED"
        if is_listening: status = "LISTENING..."
        
        # Show LLM text if fresh
        display_desc = description
        if time.time() < llm_response_timer:
            # Override description or append? Let's draw it distinctly
            cv2.rectangle(frame, (0, frame.shape[0]-130), (frame.shape[1], frame.shape[0]-90), (50, 50, 0), -1)
            cv2.putText(frame, llm_response, (10, frame.shape[0]-105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Periodic Health check
        vlm_ok = True
        llm_ok = (sampler.counter % 30 != 0) or fusion.llm.check_health()
        
        spatial_summary = fusion.get_spatial_summary()
        frame = draw_overlay(frame, status, display_desc, spatial_summary, vlm_ok, llm_ok)
        
        cv2.imshow("WalkSense Enhanced", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # --- CONTROLS ---
        if key in (ord('s'), ord('S')):
            started = not started
            state = "Started" if started else "Paused"
            print(f"[WalkSense] System {state}")
            tts.speak(f"System {state}")

        elif key in (ord('l'), ord('L')):
            if not is_listening:
                threading.Thread(target=threaded_listen, daemon=True).start()

        elif key in (ord('k'), ord('K')):
            # HARDCODED TEST QUERY
            print("[WalkSense] Injecting Hardcoded Query...")
            current_user_query = "What obstacles are in front of me?"
            fusion.handle_user_query(current_user_query)
            tts.speak("Analyzing obstacles")

        elif key in (ord('m'), ord('M')):
            is_muted = fusion.router.toggle_mute()
            print(f"[WalkSense] Muted: {is_muted}")

        elif key in (ord('q'), ord('Q')):
            break
    
    qwen_worker.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
