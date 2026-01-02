from safety.alerts import AlertEvent
from inference.runtime_state import RuntimeState
from inference.decision_router import DecisionRouter
from inference.spatial_context_manager import SpatialContextManager
from inference.llm_reasoner import LLMReasoner
from typing import Optional, List, Dict
import time


class FusionEngine:
    """
    Enhanced Fusion Engine with Spatial Context + LLM Reasoning
    
    Pipeline:
    1. YOLO Detections → SpatialContextManager → Track objects
    2. VLM Description → Store in context memory
    3. User Query + Context → LLM Reasoner → Answer
    4. All outputs → DecisionRouter → TTS/Haptics
    """
    
    def __init__(self, tts_engine, llm_backend="lm_studio", llm_url="http://localhost:1234/v1"):
        self.router = DecisionRouter(tts_engine)
        self.runtime = RuntimeState()
        
        # NEW: Spatial-temporal context manager
        self.spatial_context = SpatialContextManager(
            movement_threshold=30.0,  # pixels
            time_threshold=10.0       # seconds
        )
        
        # NEW: LLM for reasoning
        self.llm = LLMReasoner(backend=llm_backend, api_url=llm_url)
        
        # Track latest VLM description
        self.latest_vlm_description: Optional[str] = None
        self.pending_user_query: Optional[str] = None
        
        print(f"[FusionEngine] Initialized with spatial tracking + LLM reasoning")

    def update_spatial_context(self, detections: List[Dict], timestamp: float, frame_width=1280) -> None:
        """
        Update spatial-temporal context with new detections
        
        Args:
            detections: List from YoloDetector
            timestamp: Current time
            frame_width: Frame width for direction estimation
        """
        events = self.spatial_context.update(detections, timestamp, frame_width)
        
        # Announce new objects or significant movements
        for event in events:
            if event['type'] == 'NEW_OBJECT':
                message = f"{event['label']} detected on {event['direction']}"
                self.handle_spatial_event(message, "INFO")
            
            elif event['type'] == 'OBJECT_MOVED':
                message = f"{event['label']} moving {event['direction']}"
                self.handle_spatial_event(message, "WARNING")

    def handle_spatial_event(self, message: str, event_type: str = "INFO"):
        """Handle spatial awareness events"""
        event = AlertEvent(event_type, message)
        
        if self.runtime.should_emit(event.type, message):
            self.router.route(event)

    def handle_safety_alert(self, message, alert_type="CRITICAL_ALERT"):
        """Original safety alert handling (unchanged)"""
        event = AlertEvent(alert_type, message)
        
        if self.runtime.should_emit(event.type, message):
            self.router.route(event)

    def handle_vlm_description(self, description: str):
        """
        Handle VLM scene description
        
        Pipeline:
        1. Store in spatial context memory
        2. If user query pending → Answer with LLM
        3. Otherwise → Speak if not redundant
        """
        self.latest_vlm_description = description
        self.spatial_context.add_scene_description(description)
        
        # If user asked a question, answer it now
        if self.pending_user_query:
            self._answer_user_query()
        else:
            # Passive scene description
            event = AlertEvent("SCENE_DESC", description)
            self.router.route(event)

    def handle_user_query(self, query: str):
        """
        Handle user voice query
        
        Pipeline:
        1. Store query
        2. If VLM description available → Answer immediately
        3. Otherwise → Wait for next VLM update
        """
        self.pending_user_query = query
        
        # Acknowledge receipt
        event = AlertEvent("RESPONSE", "Processing your question...")
        self.router.route(event)
        
        # Try to answer immediately if we have context
        if self.latest_vlm_description:
            self._answer_user_query()

    def _answer_user_query(self):
        """Use LLM to answer user query with full context"""
        if not self.pending_user_query:
            return
        
        print(f"[FusionEngine] Answering query: {self.pending_user_query}")
        
        # Get spatial context
        spatial_context = self.spatial_context.get_context_for_llm()
        
        # Use LLM to generate answer
        answer = self.llm.answer_query(
            user_query=self.pending_user_query,
            spatial_context=spatial_context,
            scene_description=self.latest_vlm_description
        )
        
        print(f"[FusionEngine] LLM Answer: {answer}")
        
        # Speak the answer
        event = AlertEvent("RESPONSE", answer)
        self.router.route(event)
        
        # Clear pending query
        self.pending_user_query = None

    def handle_scene_description(self, text):
        """
        Backward compatibility wrapper
        """
        self.handle_vlm_description(text)

    def handle_user_query_response(self, text):
        """
        Backward compatibility - direct response
        """
        event = AlertEvent("RESPONSE", text)
        self.router.route(event)
    
    def get_spatial_summary(self) -> str:
        """Get brief summary of current spatial state"""
        return self.spatial_context.get_summary()
    
    def analyze_scene_safety(self) -> Optional[str]:
        """
        Use LLM to analyze scene for unreported safety concerns
        
        Returns:
            Safety alert or None
        """
        if not self.latest_vlm_description:
            return None
        
        spatial_context = self.spatial_context.get_context_for_llm()
        safety_alert = self.llm.analyze_safety(spatial_context, self.latest_vlm_description)
        
        if safety_alert:
            print(f"[FusionEngine] LLM Safety Alert: {safety_alert}")
            event = AlertEvent("WARNING", safety_alert)
            self.router.route(event)
        
        return safety_alert
