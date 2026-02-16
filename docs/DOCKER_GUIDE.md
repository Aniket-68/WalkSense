# How to Build and Push WalkSense to Docker Hub

This guide explains how to build the Docker image for WalkSense and push it to a container registry like Docker Hub.

## Prerequisites

1.  **Docker Desktop** installed and running.
2.  A **Docker Hub** account.

## 1. Build the Image

Run the following command in the root of the project (where the Dockerfile is):

```bash
# Replace 'your-username' with your actual Docker Hub username
docker build -t your-username/walksense:latest .
```

*Note: This may take a few minutes as it installs system dependencies like OpenCV and PyTorch.*

## 2. Test the Image Locally

Before pushing, verify the image works. Since WalkSense requires hardware access (camera/mic), testing in Docker is tricky on Windows.

**Minimal Test (No Hardware):**
```bash
docker run --rm -it your-username/walksense:latest python --version
```

**Running with Hardware (Linux Only):**
On Linux, you can pass devices:
```bash
docker run --device /dev/video0:/dev/video0 --device /dev/snd:/dev/snd -it your-username/walksense:latest
```

**Running on Windows:**
Docker on Windows generally **does not** support USB passthrough (webcam/mic) easily to Linux containers. This image is best used:
1.  For deployment on a Linux edge device (Jetson, Raspberry Pi).
2.  For backend processing if you decouple the camera capture.

## 3. Login to Docker Hub

```bash
docker login
```

Enter your username and password/token when prompted.

## 4. Push the Image

```bash
docker push your-username/walksense:latest
```

## 5. Pulling and Running Elsewhere

On another machine:

```bash
docker pull your-username/walksense:latest
docker run -it your-username/walksense:latest
```
