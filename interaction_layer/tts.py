# audio/tts.py

import subprocess
import sys
import os

class TTSEngine:
    def __init__(self):
        self.script_path = os.path.join(os.path.dirname(__file__), "audio_worker.py")
        self.current_process = None

    def _kill_process(self):
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=0.1)
            except Exception:
                pass

    def speak(self, text):
        """Speak text (interrupts previous speech)"""
        if text:
            self._kill_process()
            from loguru import logger
            logger.info(f"AI: {text}")
            try:
                self.current_process = subprocess.Popen([sys.executable, self.script_path, text])
            except Exception as e:
                logger.error(f"TTS ERROR: {e}")
            
    def stop(self):
        """Stop current speech immediately"""
        self._kill_process()
