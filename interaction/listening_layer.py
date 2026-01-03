# interaction/listening_layer.py

import speech_recognition as sr
from safety.alerts import AlertEvent

class STTListener:
    """
    Converts user voice into text (on-demand)
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen_once(self, timeout=None):
        from utils.config_loader import Config
        if timeout is None:
            timeout = Config.get("audio.stt_timeout", 5)

        try:
            with self.mic as source:
                print("[STT] Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5) # Shorten adjust time
                audio = self.recognizer.listen(source, timeout=timeout)
                
            text = self.recognizer.recognize_google(audio)
            print(f"[STT] Heard: {text}")
            return text
        except sr.WaitTimeoutError:
            print("[STT] Timeout: No speech detected")
            return None
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return None


class ListeningLayer:
    """
    Orchestrates STT input and TTS output
    Handles events from FusionEngine
    """

    def __init__(self, tts_engine, fusion_engine):
        self.tts = tts_engine
        self.fusion = fusion_engine
        self.stt = STTListener()

    def listen_for_query(self):
        """Listen for user voice query and process it"""
        query = self.stt.listen_once()
        if query:
            print(f"[ListeningLayer] Processing query: {query}")
            self.fusion.handle_user_query(query)

    def handle_event(self, event: AlertEvent):
        """Handle events from FusionEngine and speak them out"""
        print(f"[ListeningLayer] Event: {event.type} - {event.message}")
        self.tts.speak(event.message)
