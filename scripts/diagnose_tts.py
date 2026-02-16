import pyttsx3
import sys

print("=== TTS Diagnostic Test ===\n")

try:
    # Initialize engine
    print("1. Initializing pyttsx3...")
    engine = pyttsx3.init()
    print("   ✓ Engine initialized\n")
    
    # Get current settings
    print("2. Current Settings:")
    rate = engine.getProperty('rate')
    volume = engine.getProperty('volume')
    voice = engine.getProperty('voice')
    print(f"   Rate: {rate}")
    print(f"   Volume: {volume}")
    print(f"   Voice: {voice}\n")
    
    # List voices
    print("3. Available Voices:")
    voices = engine.getProperty('voices')
    for i, v in enumerate(voices):
        print(f"   [{i}] {v.name}")
        print(f"       ID: {v.id}")
    print()
    
    # Test with Zira
    print("4. Testing with Zira voice...")
    for v in voices:
        if "zira" in v.name.lower():
            engine.setProperty('voice', v.id)
            print(f"   Selected: {v.name}")
            break
    
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 150)
    
    print("\n5. Speaking test message...")
    print("   (You should hear: 'WalkSense audio test successful')\n")
    
    engine.say("WalkSense audio test successful")
    engine.runAndWait()
    
    print("✓ Test completed. Did you hear the audio? (Y/N)")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
