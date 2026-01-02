from safety.alerts import AlertEvent
from inference.runtime_state import RuntimeState
from inference.decision_router import DecisionRouter

class FusionEngine:
    def __init__(self, tts_engine):
        self.router = DecisionRouter(tts_engine)
        self.runtime = RuntimeState()

    def handle_safety_alert(self, message, alert_type="CRITICAL_ALERT"):
        event = AlertEvent(alert_type, message)
        
        # RuntimeState handles "Alert Throttling" (System Level)
        # DecisionRouter handles "Priority & Context" (User Level)
        if self.runtime.should_emit(event.type, message):
            self.router.route(event)

    def handle_scene_description(self, text):
        """
        Handle background reasoning descriptions
        """
        event = AlertEvent("SCENE_DESC", text)
        self.router.route(event)

    def handle_user_query(self, query):
        """
        Handle incoming user voice query
        """
        # For now, acknowledge the query. 
        # Future: Send to LLM/VLM for answer
        response_text = f"Processing query: {query}"
        event = AlertEvent("RESPONSE", response_text)
        self.router.route(event)

    def handle_user_query_response(self, text):
        """
        Handle direct responses to user
        """
        event = AlertEvent("RESPONSE", text)
        self.router.route(event)
