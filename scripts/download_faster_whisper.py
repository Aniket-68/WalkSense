from faster_whisper import WhisperModel
import os

def download_faster_whisper(model_name="base", download_root="models/whisper"):
    print(f"Downloading Faster-Whisper model '{model_name}' to {download_root}...")
    if not os.path.exists(download_root):
        os.makedirs(download_root)
    
    # This will download/convert the model to the specified directory
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=download_root)
    print("Download complete.")

if __name__ == "__main__":
    download_faster_whisper()
