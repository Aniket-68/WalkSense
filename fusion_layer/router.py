from interaction_layer.tts import TTSEngine
from fusion_layer.redundancy import ContextManager
from interaction_layer.auxiliary import AuxController

class DecisionRouter:
    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.context_manager = ContextManager()
        self.aux = AuxController() # Initialize Auxiliary Controller
        self.muted = False

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.tts.stop()
            self.tts.speak("Audio Muted")
        else:
            self.tts.speak("Audio Active")
        return self.muted
        
    def route(self, event):
        """
        Route an event to Interaction Layer (TTS + Aux) based on priority.
        """
        from infrastructure.config import Config
        suppress = Config.get("safety.suppression.enabled", False)
        warn_thresh = Config.get("safety.suppression.redundancy_threshold", 0.8)
        warn_timeout = Config.get("safety.suppression.warning_timeout", 5.0)
        scene_timeout = Config.get("safety.suppression.scene_timeout", 20.0)

        severity = event.type
        message = event.message
        
        # 1. CRITICAL SAFETY: Strong Feedback (Overrides Mute)
        if severity == "CRITICAL_ALERT":
            self.aux.trigger_haptic("HIGH")
            self.aux.trigger_buzzer("ALARM")
            
            self.tts.speak(f"Danger! {message}")
            self.context_manager.update_context(message)
            return

        # Check Mute for non-critical events
        if self.muted:
            return

        # 2. WARNINGS: Medium Feedback
        if severity == "WARNING":
            if not self.context_manager.is_redundant(message, threshold=warn_thresh, timeout=warn_timeout):
                self.aux.trigger_haptic("MEDIUM")
                self.aux.trigger_buzzer("WARNING")
                
                self.tts.speak(f"Warning! {message}")
                self.context_manager.update_context(message)
            return

        # 3. RESPONSE (Generic): Confirmation Feedback
        if severity == "RESPONSE" or severity == "INFO":
             self.aux.trigger_haptic("PULSE")
             
             self.tts.speak(message)
             self.context_manager.update_context(message)
             self.context_manager.set_silence_window(8)
             return

        # 4. SCENE DESCRIPTION: No Physical Feedback (Passive)
        if severity == "SCENE_DESC":
            # Strict Redundancy Check
            if not self.context_manager.is_redundant(message, threshold=warn_thresh - 0.1, timeout=scene_timeout):
                self.tts.speak(message)
                self.context_manager.update_context(message)
            return
