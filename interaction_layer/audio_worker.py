# audio/speak.py
import sys
import pyttsx3

def speak(text):
    try:
        import os
        import sys
        # Add project root to path so speak.py can find 'utils'
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)

        from infrastructure.config import Config
        
        provider = Config.get("tts.active_provider", "pyttsx3")
        config_path = f"tts.providers.{provider}"
        
        rate = Config.get(f"{config_path}.rate", 150)
        voice = Config.get(f"{config_path}.voice", "default")
        
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        
        # Basic voice selection
        if voice != "default":
             voices = engine.getProperty('voices')
             for v in voices:
                 if voice.lower() in v.name.lower():
                     engine.setProperty('voice', v.id)
                     break
                     
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[SPEAK.PY ERROR] {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        speak(text)
