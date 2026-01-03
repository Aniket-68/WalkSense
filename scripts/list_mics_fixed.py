import speech_recognition as sr

def find_mics():
    print("Searching for active microphones...")
    mic_list = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_list):
        # Clean up name for printing
        clean_name = name.encode('ascii', errors='ignore').decode('ascii')
        print(f"Index {i}: {clean_name}")

if __name__ == "__main__":
    find_mics()
