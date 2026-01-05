# scripts/run_enhanced_camera.py
"""
Enhanced WalkSense demo with:
- Spatial-temporal object tracking
- LLM-based query answering
- Context-aware reasoning
"""

import cv2
import time

from perception_layer.camera import Camera
from perception_layer.detector import YoloDetector
from perception_layer.rules import SafetyRules

from fusion_layer.engine import FusionEngine
from interaction_layer.stt import ListeningLayer
from interaction_layer.tts import TTSEngine

from reasoning_layer.vlm import QwenVLM
from infrastructure.sampler import FrameSampler
from infrastructure.scene import SceneChangeDetector

import threading
import queue
from loguru import logger
from infrastructure.performance import tracker


# =========================================================
# VISUALIZATION (Same as before)
# =========================================================

def hazard_color(label: str) -> tuple[int, int, int]:
    """
    Returns a BGR color based on the object label for visualization.
    
    Args:
        label: The object label string.
        
    Returns:
        A tuple of (B, G, R) integers.
    """
    label = label.lower()
    if label in {"knife", "gun", "fire", "stairs", "car", "bus", "truck", "bike"}:
        return (0, 0, 255)  # Red for high danger
    if label in {"person", "dog", "animal", "bicycle", "wall", "glass"}:
        return (0, 255, 255)  # Yellow/Cyan for awareness
    if label in {"chair", "table", "bag"}:
        return (255, 0, 0)  # Blue for static objects
    return (0, 255, 0)  # Green for others


def draw_detections(frame, detections: list[dict]) -> object:
    """
    Draws bounding boxes and labels on the image frame.
    
    Args:
        frame: The OpenCV image frame (numpy array).
        detections: List of detection dictionaries containing 'bbox', 'label', and 'confidence'.
        
    Returns:
        The annotated image frame.
    """
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


def draw_overlay(frame, status: str, description: str, spatial_summary: str, vlm_ok: bool = False, llm_ok: bool = False, timeline: list = [], is_listening: bool = False) -> object:
    """
    Draws the UI overlay including status, spatial summary, and AI health indicators.
    
    Args:
        frame: The OpenCV image frame.
        status: Current system status string (e.g., 'RUNNING', 'PAUSED').
        description: Current scene description or alert message.
        spatial_summary: Summary of tracked objects in spatial context.
        vlm_ok: Health status of the VLM engine.
        llm_ok: Health status of the LLM engine.
        
    Returns:
        The image frame with overlay drawn.
    """
    h, w, _ = frame.shape
    import time
    t = time.time()
    
    # 1. TOP STATUS BAR (Glassmorphism effect)
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
    status_color = (0, 255, 0) if status == "RUNNING" else (0, 165, 255)
    if is_listening: status_color = (0, 255, 255) # Cyan for listening
    
    cv2.putText(frame, f"WALKSENSE OS // {status}", (20, 35),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)
    
    # 3. AI HEALTH (Right-aligned)
    vlm_c = (0, 255, 0) if vlm_ok else (0, 0, 255)
    cv2.circle(frame, (w-40, 25), 6, vlm_c, -1)
    cv2.putText(frame, "VLM", (w-85, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    llm_c = (0, 255, 0) if llm_ok else (0, 0, 255)
    cv2.circle(frame, (w-120, 25), 6, llm_c, -1)
    cv2.putText(frame, "LLM", (w-165, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 4. DIALOGUE TIMELINE (Left Floating Panel)
    if timeline:
        start_y = 120
        panel_h = len(timeline[-3:]) * 40 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, start_y - 30), (int(w*0.45), start_y + panel_h - 30), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        for entry in timeline[-3:]:
            prefix = entry.split(":")[0]
            color = (0, 255, 255) if "USER" in prefix else (100, 255, 100)
            cv2.circle(frame, (35, start_y - 5), 4, color, -1)
            cv2.putText(frame, entry, (50, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            start_y += 40

    # 5. JARVIS CORE (Bottom Right)
    import math
    core_x, core_y = w - 100, h - 100
    pulse = (math.sin(t * 6) + 1) / 2
    inner_r = 15 + int(pulse * 10)
    outer_r = 35 + int(pulse * 5)
    
    core_color = (0, 255, 255) if is_listening else (255, 150, 0) 
    if "ALERT" in description.upper(): core_color = (0, 0, 255)
    
    cv2.circle(frame, (core_x, core_y), outer_r, core_color, 1)
    cv2.circle(frame, (core_x, core_y), inner_r, core_color, -1)
    
    angle = (t * 150) % 360
    cv2.ellipse(frame, (core_x, core_y), (45, 45), 0, angle, angle + 60, core_color, 2)
    cv2.ellipse(frame, (core_x, core_y), (45, 45), 0, angle + 180, angle + 240, core_color, 2)
    
    if is_listening:
        cv2.putText(frame, "LISTENING", (core_x - 35, core_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, core_color, 1)
    
    # 6. BOTTOM DESCRIPTION PANEL
    cv2.rectangle(frame, (0, h - 60), (w, h), (10, 10, 10), -1)
    cv2.putText(frame, f"SCENE :: {description}", (20, h - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    cv2.putText(frame, f"SPATIAL SENSE: {spatial_summary}", (20, h - 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    return frame


# =========================================================
# ASYNC WORKER (Same as before)
# =========================================================

class QwenWorker:
    """
    Asynchronous worker for handling VLM scene description requests.
    Prevents blocking the main UI/capture thread during slow VLM inference.
    """
    def __init__(self, qwen_instance: object):
        """
        Initializes the worker with a QwenVLM instance.
        """
        self.qwen = qwen_instance
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.is_busy = False
        
    def _run(self) -> None:
        """
        Internal worker loop that listens for input frames and triggers VLM inference.
        """
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
                
    def process(self, frame, context_str: str) -> bool:
        """
        Submits a frame and context for processing if the worker is not busy.
        
        Args:
            frame: The image frame to process.
            context_str: String containing detection labels and user query.
            
        Returns:
            True if the task was submitted, False if worker was busy or queue full.
        """
        if not self.is_busy and self.input_queue.empty():
            self.input_queue.put((frame.copy(), context_str))
            return True
        return False
        
    def get_result(self) -> tuple[str, float] | None:
        """
        Retrieves the latest result from the output queue if available.
        
        Returns:
            A tuple containing (description, duration) or None if no result is ready.
        """
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop(self) -> None:
        """
        Signals the worker thread to stop.
        """
        self.running = False


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Main execution loop for WalkSense Enhanced.
    Initializes hardware, model engines, and enters the real-time processing loop.
    """
    # Load central configuration
    from infrastructure.config import Config
    
    # 🟢 Camera Configuration
    CAMERA_SOURCE = Config.get("camera.source", 0)
    
    # 🟢 VLM Configuration
    VLM_PROVIDER = Config.get("vlm.active_provider", "lm_studio")
    vlm_config_path = f"vlm.providers.{VLM_PROVIDER}"
    VLM_URL = Config.get(f"{vlm_config_path}.url")
    VLM_MODEL = Config.get(f"{vlm_config_path}.model_id")
    
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
    logger.info("Creating Enhanced Fusion Engine...")
    fusion = FusionEngine(
        tts, 
        llm_backend=LLM_PROVIDER, 
        llm_url=LLM_URL,
        llm_model=LLM_MODEL
    )
    
    # Interaction Layer
    listener = ListeningLayer(None, fusion)
    
    logger.info("Loading Qwen VLM...")
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
    dialogue_history = []  # Stores last few interactions

    cv2.namedWindow("WalkSense Enhanced", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WalkSense Enhanced", WINDOW_WIDTH, WINDOW_HEIGHT)
    

    # Threaded Listener Wrapper
    def threaded_listen() -> None:
        """
        Runs the speech-to-text listener in a separate thread to avoid UI lag.
        Updates shared 'current_user_query' upon successful recognition.
        """
        nonlocal current_user_query, is_listening
        is_listening = True
        # 🔴 STOP TTS IMMEDIATELY to prevent interference
        tts.stop()
        
        logger.info("Listening thread started...")
        tracker.start_timer("stt")
        query = listener.stt.listen_once()
        tracker.stop_timer("stt")
        if query:
            logger.info(f"USER: {query}")
            current_user_query = query
            dialogue_history.append(f"USER: {query}")
            fusion.handle_user_query(query)
        is_listening = False

    for frame in camera.stream():
        current_time = time.time()
        tracker.start_timer("frame_total")
        
        # YOLO Detection
        tracker.start_timer("yolo")
        detections = detector.detect(frame)
        tracker.stop_timer("yolo")
        frame = draw_detections(frame, detections)
        
        # 🔴 SAFETY LAYER
        safety_result = safety.evaluate(detections)
        if safety_result:
            alert_type, message = safety_result
            logger.warning(f"Safety Alert: {message}")
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
                logger.info(f"VLM Description: {new_desc}")
                description = new_desc
                
                if current_user_query:
                    tracker.start_timer("llm_reasoning")
                    answer = fusion.handle_vlm_description(new_desc)
                    tracker.stop_timer("llm_reasoning")
                    if answer:
                        llm_response = f"AI: {answer}"
                        llm_response_timer = time.time() + 10 # Show for 10s
                        dialogue_history.append(f"AI: {answer}")
                        logger.info(f"LLM Answer: {answer}")
                        current_user_query = None
                else:
                    fusion.handle_scene_description(new_desc)
            
            # 2. Trigger Logic
            has_query = (current_user_query is not None)
            time_to_sample = sampler.should_sample()
            should_run_qwen = False
            
            if has_query:
                should_run_qwen = True # Priority
            elif time_to_sample and scene_detector.has_changed(frame):
                should_run_qwen = True
            
            if should_run_qwen:
                context_str = ", ".join([d["label"] for d in detections])
                if has_query:
                    context_str += f". USER QUESTION: {current_user_query}"
                
                # Send to worker (non-blocking)
                if qwen_worker.process(frame, context_str):
                    status_text = "Reasoning..." if has_query else "Scanning..."
                    cv2.putText(frame, status_text, (frame.shape[1]-150, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # UI Overlay Construction
        status = "RUNNING" if started else "PAUSED"
        if is_listening: status = "LISTENING..."
        
        vlm_ok = True
        llm_ok = (sampler.counter % 30 != 0) or fusion.llm.check_health()
        
        spatial_summary = fusion.get_spatial_summary()
        frame = draw_overlay(
            frame, status, description, spatial_summary, 
            vlm_ok, llm_ok, 
            timeline=dialogue_history,
            is_listening=is_listening
        )
        
        tracker.stop_timer("frame_total")
        
        # Occasional Performance Print
        if sampler.counter % 100 == 0 and sampler.counter > 0:
            logger.info(f"Performance Stats: {tracker.get_summary()}")
        
        cv2.imshow("WalkSense Enhanced", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # --- CONTROLS ---
        if key in (ord('s'), ord('S')):
            started = not started
            state = "Started" if started else "Paused"
            logger.info(f"System {state}")
            tts.speak(f"System {state}")

        elif key in (ord('l'), ord('L')):
            if not is_listening:
                threading.Thread(target=threaded_listen, daemon=True).start()

        elif key in (ord('k'), ord('K')):
            # HARDCODED TEST QUERY
            logger.info("Injecting Hardcoded Query...")
            current_user_query = "What obstacles are in front of me?"
            fusion.handle_user_query(current_user_query)
            tts.speak("Analyzing obstacles")

        elif key in (ord('m'), ord('M')):
            is_muted = fusion.router.toggle_mute()
            logger.info(f"Muted: {is_muted}")

        elif key in (ord('q'), ord('Q')):
            logger.info("Quitting system...")
            tracker.plot_metrics()
            break
    
    qwen_worker.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
