# interaction/listening_layer.py

import speech_recognition as sr
import os
from safety.alerts import AlertEvent

class STTListener:
    """
    Converts user voice into text (on-demand)
    """

    def __init__(self):
        from utils.config_loader import Config
        self.recognizer = sr.Recognizer()
        
        # New Top-Level Microphone Config:
        self.device_index = Config.get("microphone.hardware.id") if Config.get("microphone.mode") == "hardware" else None
        self.cal_duration = Config.get("microphone.hardware.calibration_duration", 2.0)
        self.energy_thresh = Config.get("microphone.hardware.energy_threshold", 50)
        self.pause_thresh = Config.get("microphone.hardware.pause_threshold", 1.0)
        self.dynamic_energy = Config.get("microphone.hardware.dynamic_energy", True)
        
        try:
            self.mic = sr.Microphone(device_index=self.device_index)
            # Log the device name for confirmation
            mics = sr.Microphone.list_microphone_names()
            dev_name = mics[self.device_index] if (self.device_index is not None and self.device_index < len(mics)) else "System Default"
            print(f"[STT] Initialized Microphone ID: {self.device_index} ({dev_name})")
        except Exception as e:
            print(f"[STT ERROR] Failed to init mic {self.device_index}: {e}")
            self.mic = sr.Microphone() # Fallback

    def listen_once(self, timeout=None):
        from utils.config_loader import Config
        
        provider = Config.get("stt.active_provider", "google")
        config_path = f"stt.providers.{provider}"
        
        if timeout is None:
            timeout = Config.get(f"{config_path}.timeout", 10)
        
        limit = Config.get(f"{config_path}.phrase_time_limit", 15)

        try:
            with self.mic as source:
                print(f"[STT] Calibrating for environment noise ({self.cal_duration}s)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=self.cal_duration)
                
                # Apply config settings
                self.recognizer.energy_threshold = self.energy_thresh
                self.recognizer.dynamic_energy_threshold = self.dynamic_energy
                self.recognizer.pause_threshold = self.pause_thresh
                self.recognizer.non_speaking_duration = 0.5
                
                print(f"\n[STT] >>> LISTENING NOW ({provider}). SPEAK IN HINDI OR ENGLISH. <<<")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=limit)
                print("[STT] Recording finished. Transcribing...")
            
            # Route to appropriate recognition method
            detailed_info = ""
            if provider == "google":
                text = self.recognizer.recognize_google(audio)
                detailed_info = "Language: en-US (Google Default)"
            
            elif provider == "whisper_api":
                # ... (rest of whisper_api logic)
                api_key = os.getenv(Config.get(f"{config_path}.api_key_env", "OPENAI_API_KEY"))
                if not api_key:
                    print("[STT ERROR] OPENAI_API_KEY not set in environment")
                    return None
                
                text = self.recognizer.recognize_whisper_api(
                    audio,
                    api_key=api_key,
                    model=Config.get(f"{config_path}.model", "whisper-1"),
                    language=Config.get(f"{config_path}.language", "en")
                )
                detailed_info = f"Provider: Whisper API"
            
            elif provider == "whisper_local":
                # Local Whisper (faster-whisper or OpenAI whisper)
                try:
                    # Try faster-whisper first (much faster)
                    text, lang_info = self._recognize_faster_whisper(
                        audio,
                        model_size=Config.get(f"{config_path}.model_size", "base"),
                        device=Config.get(f"{config_path}.device", "cpu"),
                        language=Config.get(f"{config_path}.language", "en")
                    )
                    detailed_info = lang_info
                except ImportError:
                    # Fallback to OpenAI's whisper
                    text = self._recognize_openai_whisper(
                        audio,
                        model_size=Config.get(f"{config_path}.model_size", "base"),
                        language=Config.get(f"{config_path}.language", "en")
                    )
                    detailed_info = lang_info
            
            else:
                print(f"[STT ERROR] Unknown provider: {provider}")
                return None
            
            if text:
                print("\n" + "═"*45)
                print(f" 🎙️  STT DEBUG | {detailed_info}")
                print(f" 🗣️  USER SAID: {text}")
                print("═"*45 + "\n")
            else:
                print("[STT] No text transcribed.")
                
            return text
            
        except sr.WaitTimeoutError:
            print("[STT] Timeout: No speech detected")
            return None
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return None
    
    def _recognize_faster_whisper(self, audio, model_size="base", device="cpu", language="en"):
        """Use faster-whisper for local transcription (recommended)"""
        from faster_whisper import WhisperModel
        import io
        import wave
        
        # Convert AudioData to WAV bytes
        wav_data = io.BytesIO(audio.get_wav_data())
        
        # Load model (cached after first use)
        if not hasattr(self, '_whisper_model') or self._whisper_model_size != model_size:
            print(f"[STT] Loading faster-whisper model: {model_size} into models/whisper")
             # Use the local models directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(project_root, "models", "whisper")
            self._whisper_model = WhisperModel(model_size, device=device, compute_type="int8", download_root=model_dir)
            self._whisper_model_size = model_size
        
        # Transcribe
        segments, info = self._whisper_model.transcribe(wav_data, language=language)
        text = " ".join([segment.text for segment in segments])
        
        lang_info = f"Detected: {info.language} ({int(info.language_probability*100)}% prob)"
        return text.strip(), lang_info
    
    def _recognize_openai_whisper(self, audio, model_size="base", language="en"):
        """Fallback: Use OpenAI's whisper for local transcription"""
        import whisper
        import io
        import tempfile
        
        # Load model (cached)
        if not hasattr(self, '_whisper_model') or self._whisper_model_size != model_size:
            print(f"[STT] Loading whisper model: {model_size} from models/whisper")
            # Use the local models directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(project_root, "models", "whisper")
            self._whisper_model = whisper.load_model(model_size, download_root=model_dir)
            self._whisper_model_size = model_size
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio.get_wav_data())
            temp_path = f.name
        
        try:
            result = self._whisper_model.transcribe(temp_path, language=language)
            text = result["text"].strip()
            lang_info = f"Detected: {result.get('language', 'unknown')}"
            return text, lang_info
        finally:
            import os
            os.unlink(temp_path)


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
