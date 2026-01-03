import speech_recognition as sr

def list_microphones():
    print("Available Microphones:")
    mics = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mics):
        print(f"ID {i}: {name}")

if __name__ == "__main__":
    list_microphones()
