# LLM-based Navigation Narrator
# Accumulates detections and generates intelligent guidance periodically

import time

class NavigationNarrator:
    """
    Accumulates YOLO detections and uses LLM to generate
    concise, intelligent navigation guidance at controlled intervals
    """
    
    def __init__(self, llm_reasoner, interval=2.0, max_words=15):
        self.llm = llm_reasoner
        self.interval = interval
        self.max_words = max_words
        self.last_narration_time = 0
        self.accumulated_detections = []
    
    def add_detections(self, detections):
        """Add current frame's detections to accumulator"""
        self.accumulated_detections.extend(detections)
    
    def should_narrate(self):
        """Check if it's time for next narration"""
        return (time.time() - self.last_narration_time) >= self.interval
    
    def generate_narration(self, tts_engine):
        """Generate and speak ONE navigation line"""
        if not self.should_narrate():
            return
        
        if not self.accumulated_detections:
            self.accumulated_detections = []
            self.last_narration_time = time.time()
            return
        
        # Get LLM narration
        narration = self.llm.narrate_environment(
            self.accumulated_detections,
            max_words=self.max_words
        )
        
        # Reset accumulator
        self.accumulated_detections = []
        self.last_narration_time = time.time()
        
        # Speak if we got something
        if narration and narration.strip():
            print(f"[NAV] {narration}")
            tts_engine.speak(narration)
