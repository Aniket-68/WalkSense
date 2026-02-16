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
from infrastructure.darkness_detector import DarknessDetector

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


def draw_overlay(frame, status: str, description: str, spatial_summary: str, vlm_ok: bool = False, llm_ok: bool = False, timeline: list = [], is_listening: bool = False, current_query: str = None, query_start_time: float = None, **kwargs) -> object:
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
    
    cv2.putText(frame, f"WALKSENSE OS // {status} // {kwargs.get('mode', 'NORMAL')}", (20, 35),
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
        last_int_ts = kwargs.get('last_interaction_ts', t)
        elapsed_int = t - last_int_ts
        
        # Fade out after 10 seconds, completely gone by 13 seconds
        alpha_hist = 1.0
        if elapsed_int > 10.0:
            alpha_hist = max(0, 1.0 - (elapsed_int - 10.0) / 3.0)
            
        if alpha_hist > 0.01:
            start_y = 120
            # Truncate to last 3
            visible_items = timeline[-3:]
            
            panel_h = len(visible_items) * 40 + 20
            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, start_y - 30), (int(w*0.45), start_y + panel_h - 30), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.5 * alpha_hist, frame, 1.0 - (0.5 * alpha_hist), 0, frame)
            
            for entry in visible_items:
                prefix = entry.split(":")[0]
                color = (0, 255, 255) if "USER" in prefix else (100, 255, 100)
                
                # Manual text truncation if too long
                display_txt = entry
                if len(display_txt) > 55: display_txt = display_txt[:52] + "..."
                
                # Apply alpha to text (dim colors)
                if alpha_hist < 1.0:
                     color = (int(color[0]*alpha_hist), int(color[1]*alpha_hist), int(color[2]*alpha_hist))
                
                cv2.circle(frame, (35, start_y - 5), 4, color, -1)
                cv2.putText(frame, display_txt, (50, start_y),
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
        cv2.putText(frame, "LISTENING...", (w//2 - 100, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
        
    # NEW: Show Pending Query Prominently (Only if NOT currently answering)
    # OR Show Fading Query
    target_query = current_query
    is_fading = False
    alpha = 1.0
    
    if not target_query and (t - getattr(draw_overlay, 'last_query_ts', 0) < 5.0):
        # We are in fading state
        target_query = kwargs.get('fading_text')
        is_fading = True
        elapsed_fade = t - getattr(draw_overlay, 'last_query_ts', 0)
        alpha = max(0, 1.0 - (elapsed_fade / 5.0))
        
    # Store timestamp for next frame if provided
    if kwargs.get('last_ts'):
        draw_overlay.last_query_ts = kwargs.get('last_ts')

    if target_query and "AI:" not in description:
        # Draw a highlight box
        qh, qw = 60, 800
        qx, qy = (w - qw)//2, h - 150
        
        # Glassmorphism background
        overlay = frame.copy()
        cv2.rectangle(overlay, (qx, qy), (qx + qw, qy + qh), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85 * alpha, frame, 1.0 - (0.85 * alpha), 0, frame)
        
        # Border
        border_color = (0, 255, 255) if not is_fading else (0, 255, 0)
        # Apply alpha to border manually not easy in CV2, just assume solid for now or skip
        if not is_fading:
             cv2.rectangle(frame, (qx, qy), (qx + qw, qy + qh), border_color, 1)
        
        # Text with status
        text_color = (255, 255, 255)
        # We can't easily do alpha text in pure CV2 without another overlay, 
        # so we'll just dim the color for fading
        if is_fading:
             val = int(255 * alpha)
             text_color = (val, val, val)
             
        status_suffix = ""
        if not is_fading:
            elapsed = time.time() - (query_start_time if query_start_time else time.time())
            status_suffix = f" (Processing {elapsed:.1f}s...)"
            
        display_text = f"QUERY: {target_query}{status_suffix}"
        # Truncate if too long
        if len(display_text) > 60: display_text = display_text[:57] + "..."
        
        cv2.putText(frame, display_text, (qx + 20, qy + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # 6. BOTTOM DESCRIPTION PANEL
    cv2.rectangle(frame, (0, h - 60), (w, h), (10, 10, 10), -1)
    
    # Use Green color if it's an AI response
    text_color = (100, 255, 100) if "AI:" in description else (255, 255, 255)
    
    cv2.putText(frame, f"WalkSense :: {description}", (20, h - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
    
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
    darkness_detector = DarknessDetector(darkness_threshold=40, area_threshold=0.75)
    qwen_worker = QwenWorker(qwen)
    
    # UI Variables
    started = False
    description = "System idle. Press S to start."
    current_user_query = None
    llm_response = ""
    llm_response_timer = 0
    llm_response_timer = 0
    llm_response_timer = 0
    is_listening = False
    is_asking = False
    query_start_time = None
    last_query_ts = 0 
    fading_query_text = None
    last_interaction_ts = 0 # Timestamp of last message added to timeline
    dialogue_history = []  # Stores last few interactions
    is_too_dark = False  # Track if environment is too dark
    last_darkness_alert_time = 0  # Prevent spamming darkness alerts

    cv2.namedWindow("WalkSense Enhanced", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WalkSense Enhanced", WINDOW_WIDTH, WINDOW_HEIGHT)
    

    # Threaded Listener Wrapper
    def threaded_listen() -> None:
        """
        Runs the speech-to-text listener in a separate thread to avoid UI lag.
        Updates shared 'current_user_query' upon successful recognition.
        """
        nonlocal current_user_query, is_listening, is_asking, llm_response, llm_response_timer, query_start_time, last_query_ts, fading_query_text, last_interaction_ts
        # is_listening = True # MOVED DOWN
        
        # 🔴 STOP TTS IMMEDIATELY to prevent interference
        tts.stop()
        
        max_retries = 2
        for attempt in range(max_retries):
            # PROMPT USER
            is_asking = True # Status update
            if attempt == 0:
                tts.speak("What do you want to know?")
            else:
                tts.speak("I didn't hear you. What do you want to know?")
                
            update_status("ASKING...")
            time.sleep(1.8) # Wait for TTS to finish roughly
            is_asking = False # Done asking
            
            # NOW we are actually listening
            is_listening = True
            logger.info(f"Listening thread started (Attempt {attempt+1})...")
            tracker.start_timer("stt")
            query = listener.stt.listen_once()
            tracker.stop_timer("stt")
            is_listening = False # Reset immediately after interact
            
            if query:
                print(f"\n[USER QUERY] >>> {query} <<<\n")
                logger.info(f"USER: {query}")
                current_user_query = query
                query_start_time = time.time()
                
                user_entry = f"USER: {query}"
                if not dialogue_history or dialogue_history[-1] != user_entry:
                    dialogue_history.append(user_entry)
                    last_interaction_ts = time.time()
                
                # Get immediate answer (Stage 1)
                immediate_ans = fusion.handle_user_query(query)
                if immediate_ans:
                    llm_response = f"AI: {immediate_ans}"
                    llm_response_timer = time.time() + 10
                    
                    ai_entry = f"AI: {immediate_ans}"
                    if not dialogue_history or dialogue_history[-1] != ai_entry:
                        dialogue_history.append(ai_entry)
                        last_interaction_ts = time.time()
                break # Success
            
            # If query is None (Timeout), loop continues
        
        is_listening = False
    
    # Helper to update status in the thread if needed (requires thread safety but variables are simple)
    def update_status(msg):
        nonlocal description
        # description = msg # Optional: don't override scene desc
        pass

    # 🔴 REMOTE CONTROL LISTENER (Bluetooth Headset Support)
    try:
        from pynput import keyboard
        logger.info("Initializing Global Key Listener for Bluetooth Controls...")
        
        def on_remote_press(key):
            try:
                # Check for Media Play/Pause (common on headsets)
                if key == keyboard.Key.media_play_pause:
                    logger.info("[REMOTE] Play/Pause detected - Triggering Listen")
                    if not is_listening and not is_asking:
                        threading.Thread(target=threaded_listen, daemon=True).start()
            except AttributeError:
                pass

        remote_listener = keyboard.Listener(on_press=on_remote_press)
        remote_listener.start()
        logger.info("✓ Remote Listener Active (Press Play/Pause on Headset to Speak)")
    except Exception as e:
        logger.warning(f"Could not setup global listener: {e}")

    for frame in camera.stream():
        current_time = time.time()
        tracker.start_timer("frame_total")
        
        # YOLO Detection
        tracker.start_timer("yolo")
        detections = detector.detect(frame)
        tracker.stop_timer("yolo")
        
        # Keep clean frame for VLM/Scene detection
        clean_frame = frame.copy()
        
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
                        
                        ai_entry = f"AI: {answer}"
                        # Simple deduplication: don't look back just at the last one, 
                        # sometimes "USER" message is in between.
                        # But here, we just want to avoid appending if already there recently.
                        if not dialogue_history or dialogue_history[-1] != ai_entry:
                             dialogue_history.append(ai_entry)
                             last_interaction_ts = time.time()
                        
                        logger.info(f"LLM Answer: {answer}")
                        
                        # TTS is already handled by the router when routing the RESPONSE event
                        # No need to call tts.speak() here as it would be redundant
                        
                        fading_query_text = current_user_query # Save for fading effect
                        last_query_ts = time.time()
                        current_user_query = None
                else:
                    fusion.handle_scene_description(new_desc)
                    logger.info("VLM Update (Silent)")
            
            # 2. Darkness Check (before VLM processing)
            is_too_dark, dark_percentage = darkness_detector.is_too_dark(clean_frame)
            
            # 3. Trigger Logic
            has_query = (current_user_query is not None)
            time_to_sample = sampler.should_sample()
            should_run_qwen = False
            
            if is_too_dark:
                # Environment is too dark - skip VLM and notify user
                if current_time - last_darkness_alert_time > 10.0:  # Alert every 10 seconds max
                    brightness_level = darkness_detector.get_brightness_level(clean_frame)
                    darkness_msg = f"Your view is too dark ({dark_percentage*100:.0f}% dark). Please move to a brighter area."
                    logger.warning(f"Darkness detected: {dark_percentage*100:.1f}% - Skipping VLM")
                    
                    # Update description and speak
                    description = f"DARK ENVIRONMENT: {brightness_level} - {darkness_msg}"
                    tts.speak(darkness_msg)
                    
                    last_darkness_alert_time = current_time
                
                # If user has a query but it's too dark, clear the query and notify
                if has_query and current_time - last_darkness_alert_time < 2.0:
                    logger.info(f"Clearing query due to darkness: {current_user_query}")
                    fading_query_text = current_user_query
                    last_query_ts = time.time()
                    current_user_query = None
                    
                # Skip VLM processing
                should_run_qwen = False
            else:
                # Normal lighting - proceed with VLM logic
                if has_query:
                    should_run_qwen = True # Priority
                elif time_to_sample and scene_detector.has_changed(clean_frame):
                    should_run_qwen = True
            
            if should_run_qwen:
                context_str = ", ".join([d["label"] for d in detections])
                if has_query:
                    context_str += f". USER QUESTION: {current_user_query}"
                
                # Send to worker (non-blocking)
                if qwen_worker.process(clean_frame, context_str):
                    status_text = "Reasoning..." if has_query else "Scanning..."
                    cv2.putText(frame, status_text, (frame.shape[1]-150, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # UI Overlay Construction
        status = "RUNNING" if started else "PAUSED"
        if is_asking: status = "WAIT..."
        if is_listening: status = "LISTENING..."
        
        # Check if we are in the 'prompting' phase (heuristically, if is_listening but mic not open yet/during sleep)
        # Actually, let's just rely on LISTENING... for now, or add a shared variable.
        # But for simplicity, let the user see 'LISTENING...' while the AI asks. 
        # It serves as a 'Attention' signal.
        
        vlm_ok = True
        llm_ok = (sampler.counter % 30 != 0) or fusion.llm.check_health()
        
        # Override description with LLM response if active
        display_text = description
        if time.time() < llm_response_timer and llm_response:
             display_text = llm_response

        spatial_summary = fusion.get_spatial_summary()
        frame = draw_overlay(
            frame, status, display_text, spatial_summary, 
            vlm_ok, llm_ok, 
            timeline=dialogue_history,
            is_listening=is_listening,
            current_query=current_user_query,
            query_start_time=query_start_time,
            mode=fusion.router.mode,
            last_ts=last_query_ts,
            fading_text=fading_query_text,
            last_interaction_ts=last_interaction_ts
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

        elif key in (ord('c'), ord('C')):
            # TOGGLE CRITICAL MODE
            new_mode = fusion.router.toggle_mode()
            logger.info(f"Mode toggled to: {new_mode}")

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
