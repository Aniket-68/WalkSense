import speech_recognition as sr
import time

def test_mic_signal():
    r = sr.Recognizer()
    print("Testing Microphone Signal for 5 seconds...")
    print("Please Speak or Make Noise!")
    
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            print(f"Initial Ambient Energy Threshold: {r.energy_threshold}")
            
            # Record for 4 seconds
            print("Listening for audio signal...")
            try:
                # We use listen with a timeout but we just want to see if it triggers
                audio = r.listen(source, timeout=5, phrase_time_limit=4)
                print("Signal detected and recorded!")
                print(f"Final Energy Threshold: {r.energy_threshold}")
            except sr.WaitTimeoutError:
                print("No signal detected! (Silence)")
            except Exception as e:
                print(f"Error during listening: {e}")
                
    except Exception as e:
        print(f"Could not open microphone: {e}")

if __name__ == "__main__":
    test_mic_signal()
