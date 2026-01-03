import whisper
import os

def download_model(model_name="base", download_root="models/whisper"):
    print(f"Downloading Whisper model '{model_name}' to {download_root}...")
    if not os.path.exists(download_root):
        os.makedirs(download_root)
    
    # This will download the model to the specified directory
    model = whisper.load_model(model_name, download_root=download_root)
    print("Download complete.")

if __name__ == "__main__":
    download_model()
