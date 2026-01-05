from perception_layer.alerts import AlertEvent
from fusion_layer.state import RuntimeState
from fusion_layer.router import DecisionRouter
from reasoning_layer.llm import LLMReasoner
from fusion_layer.context import SpatialContextManager

class FusionEngine:
    def __init__(self, tts_engine, llm_backend="lm_studio", llm_url="http://localhost:1234/v1", llm_model="qwen/qwen3-vl-4b"):
        """
        Orchestration layer that handles Safety, Reasoning, and Interaction.
        """
        self.router = DecisionRouter(tts_engine)
        self.runtime = RuntimeState()
        
        # Enhanced Reasoning Modules
        self.llm = LLMReasoner(backend=llm_backend, api_url=llm_url, model_name=llm_model)
        self.spatial = SpatialContextManager()
        
        # State for query handling
        self.pending_query = None

    def handle_safety_alert(self, message, alert_type="CRITICAL_ALERT"):
        """Handle immediate physical hazards"""
        event = AlertEvent(alert_type, message)
        
        # Only emit if it passes the system-level cooldown (RuntimeState)
        if self.runtime.should_emit(event.type, message):
            self.router.route(event)

    def update_spatial_context(self, detections, timestamp, frame_width=1280):
        """Update tracking and spatial memory"""
        events = self.spatial.update(detections, timestamp, frame_width)
        
        # If any new/moved objects are 'Close', tell the router
        for event in events:
            if event["distance"] in ["very close", "close"]:
                msg = f"{event['label']} {event['distance']} to your {event['direction']}"
                self.router.route(AlertEvent("WARNING", msg))

    def get_spatial_summary(self):
        """Brief text for the UI overlay"""
        return self.spatial.get_summary()

    def handle_vlm_description(self, text):
        """
        Receives scene understanding from VLM (Qwen).
        Decides if this should be spoken or used to answer a query.
        """
        # Save to memory
        self.spatial.add_scene_description(text)
        
        if self.pending_query:
            # We were waiting for a fresh description to answer a user query!
            ans = self._generate_llm_answer(self.pending_query, text)
            self.pending_query = None
            return ans
        else:
            # Regular passive description
            event = AlertEvent("SCENE_DESC", text)
            self.router.route(event)
            return text

    def handle_scene_description(self, text):
        """Compatibility wrapper"""
        self.handle_vlm_description(text)

    def handle_user_query(self, query):
        """
        User asked a question. 
        We acknowledge it, then wait for the next VLM frame to answer accurately.
        """
        self.pending_query = query
        
        # Immediate acknowledgement
        ack_event = AlertEvent("RESPONSE", f"Checking on: {query}")
        self.router.route(ack_event)

    def _generate_llm_answer(self, query, vlm_desc):
        """Internal: Use LLM to fuse spatial context and VLM and solve user query"""
        spatial_ctx = self.spatial.get_context_for_llm()
        
        # Use LLM to reason over everything
        answer = self.llm.answer_query(
            user_query=query,
            spatial_context=spatial_ctx,
            scene_description=vlm_desc
        )

        # FACTUAL GROUNDING CHECK (Anti-Hallucination)
        # If the LLM says "Yes" but VLM mentions a different number than the query, flag it.
        # Example: Query has "50", VLM has "100", LLM says "Yes".
        import re
        numbers_in_query = re.findall(r'\d+', query)
        if numbers_in_query:
            for num in numbers_in_query:
                # If query mentions a number not in VLM, but LLM is being positive...
                if num not in vlm_desc and ("yes" in answer.lower() or "confirm" in answer.lower()):
                    vlm_nums = re.findall(r'\d+', vlm_desc)
                    if vlm_nums:
                        correction = f"Wait, the camera actually sees {vlm_nums[0]}. Not {num}."
                        from loguru import logger
                        logger.warning(f"Grounding Alert: Hallucination detected for {num}. Overriding LLM.")
                        answer = correction
                        break
        
        # Send the final 'thought' to the router
        self.router.route(AlertEvent("RESPONSE", answer))
        return answer

    def handle_user_query_response(self, text):
        """Manual override or fallback for query response"""
        event = AlertEvent("RESPONSE", text)
        self.router.route(event)
