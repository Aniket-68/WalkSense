# interaction/stt_listener.py

import speech_recognition as sr

class STTListener:
    """
    Converts user voice into text (on-demand)
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen_once(self, timeout=5):
        with self.mic as source:
            print("[STT] Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=timeout)

        try:
            text = self.recognizer.recognize_google(audio)
            print(f"[STT] Heard: {text}")
            return text
        except Exception:
            return None
