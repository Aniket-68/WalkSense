
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interaction_layer.stt import STTListener
from infrastructure.config import Config
from loguru import logger

def test_stt():
    print("=== STT DIAGNOSTIC TEST ===")
    
    # 1. Config Check
    mic_id = Config.get("microphone.hardware.id")
    print(f"Configured Mic ID: {mic_id}")
    
    # 2. Initialize Listener
    print("Initializing STT Listener (this loads models)...")
    try:
        listener = STTListener()
    except Exception as e:
        print(f"CRITICAL: Failed to init STTListener: {e}")
        return

    # 3. Test Listening
    print("\n-------------------------------------------")
    print("Please Speak something into the microphone...")
    print("-------------------------------------------")
    
    start = time.time()
    text = listener.listen_once()
    duration = time.time() - start
    
    print("\n-------------------------------------------")
    print(f"Result: {text}")
    print(f"Time Taken: {duration:.2f}s")
    print("-------------------------------------------")

if __name__ == "__main__":
    test_stt()
