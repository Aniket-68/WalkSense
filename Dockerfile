# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - build-essential: for compiling python packages
# - libgl1, libglib2.0-0: for OpenCV
# - portaudio19-dev: for PyAudio (microphone)
# - espeak, libespeak-dev: for pyttsx3 (TTS)
# - ffmpeg: for audio processing
# - git: for pulling dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    espeak \
    libespeak-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Note: pyaudio often needs manual installation or specific pip flags on Linux, 
# but usually works with portaudio19-dev installed.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for logs and models if they don't exist
RUN mkdir -p logs models docs/plots

# Expose port (if web interface is added later, e.g., 5000 or 8000)
# EXPOSE 8000

# Default command
# Users can override this to run specific scripts, e.g.:
# docker run --device /dev/video0:/dev/video0 walksense python scripts/run_enhanced_camera.py
CMD ["python", "scripts/run_enhanced_camera.py"]
