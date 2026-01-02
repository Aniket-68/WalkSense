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


def draw_overlay(frame, status, description, spatial_summary):
    h, w, _ = frame.shape
    
    # Status bar
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {status}", (10, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
    # Configuration
    QWEN_BACKEND = "lm_studio"
    LM_STUDIO_URL = "http://localhost:1234/v1"
    QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    
    # LLM Configuration (NEW)
    LLM_BACKEND = "lm_studio"
    LLM_URL = "http://localhost:1234/v1"
    
    camera = Camera()
    detector = YoloDetector()
    safety = SafetyRules()
    
    tts = TTSEngine()
    
    # Enhanced Fusion Engine with LLM
    print("[INIT] Creating Enhanced Fusion Engine...")
    fusion = FusionEngine(tts, llm_backend=LLM_BACKEND, llm_url=LLM_URL)
    
    # Interaction Layer
    listener = ListeningLayer(None, fusion)
    
    print("[INIT] Loading Qwen VLM...")
    qwen = QwenVLM(
        backend=QWEN_BACKEND,
        model_id=QWEN_MODEL_ID,
        lm_studio_url=LM_STUDIO_URL
    )
    
    sampler = FrameSampler(every_n_frames=150)
    scene_detector = SceneChangeDetector(threshold=0.15)
    qwen_worker = QwenWorker(qwen)
    
    started = False
    description = "System idle. Press S to start."
    current_user_query = None
    
    cv2.namedWindow("WalkSense Enhanced", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WalkSense Enhanced", 1280, 720)
    
    print("[WalkSense] Enhanced system running with spatial tracking + LLM reasoning")
    
    for frame in camera.stream():
        current_time = time.time()
        
        # YOLO Detection
        detections = detector.detect(frame)
        frame = draw_detections(frame, detections)
        
        # 🔴 SAFETY LAYER (CRITICAL)
        safety_result = safety.evaluate(detections)
        if safety_result:
            alert_type, message = safety_result
            fusion.handle_safety_alert(message, alert_type)
            description = message
        
        # 🟢 SPATIAL TRACKING (NEW)
        if started:
            fusion.update_spatial_context(detections, current_time, frame.shape[1])
        
        # 🟡 VLM PROCESSING (BEST-EFFORT)
        is_critical = safety_result and safety_result[0] == "CRITICAL_ALERT"
        
        if started and not is_critical:
            # Check for VLM results
            result = qwen_worker.get_result()
            if result:
                new_desc, duration = result
                print(f"[QWEN] {duration:.2f}s | {new_desc}")
                description = new_desc
                
                # Send to fusion engine (will trigger LLM if query pending)
                fusion.handle_vlm_description(new_desc)
            
            # Decide to run VLM
            has_query = (current_user_query is not None)
            time_to_sample = sampler.should_sample()
            
            should_run_qwen = False
            if has_query:
                should_run_qwen = True
            elif time_to_sample and scene_detector.has_changed(frame):
                should_run_qwen = True
            
            if should_run_qwen:
                context_str = ", ".join([d["label"] for d in detections]) if detections else ""
                
                if has_query:
                    context_str += f". USER QUESTION: {current_user_query}"
                
                if qwen_worker.process(frame, context_str):
                    cv2.putText(frame, "Thinking...", (frame.shape[1]-120, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if has_query:
                        cv2.putText(frame, "LLM Reasoning...", (frame.shape[1]-200, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # UI Overlay
        status = "RUNNING" if started else "WAITING"
        spatial_summary = fusion.get_spatial_summary()
        frame = draw_overlay(frame, status, description, spatial_summary)
        
        cv2.imshow("WalkSense Enhanced", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord('s'), ord('S')):
            started = True
            tts.speak("WalkSense enhanced mode started")
        
        if key in (ord('l'), ord('L')):
            print("[WalkSense] Listening for user question...")
            query = listener.stt.listen_once()
            if query:
                print(f"[WalkSense] User asked: {query}")
                current_user_query = query
                fusion.handle_user_query(query)  # Will be answered by LLM
        
        if key in (ord('m'), ord('M')):
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
