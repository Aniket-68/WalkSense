"""
FastAPI server for WalkSense.
Provides REST endpoints, WebSocket state streaming, and MJPEG camera feed.

Run with:
    cd backend && python -m api.server
"""

import asyncio
import json
import time
import sys
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from loguru import logger

# Ensure backend root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.manager import SystemManager

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(title="WalkSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev server + any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SystemManager()


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────

@app.post("/api/system/start")
async def system_start():
    """Start the processing pipeline."""
    manager.start()
    return {"status": "started"}


@app.post("/api/system/stop")
async def system_stop():
    """Stop the processing pipeline."""
    manager.stop()
    return {"status": "stopped"}


@app.get("/api/system/status")
async def system_status():
    """Get current system state."""
    return manager.get_state()


@app.post("/api/query")
async def submit_query(req: QueryRequest):
    """Submit a text query to the system."""
    manager.submit_query(req.text)
    return {"status": "submitted", "query": req.text}


@app.post("/api/voice-query")
async def voice_query(audio: UploadFile = File(...)):
    """Accept audio from browser mic, transcribe, and submit as query."""

    try:
        audio_bytes = await audio.read()
        logger.info(f"[VoiceQuery] Received audio: {len(audio_bytes)} bytes, type={audio.content_type}")

        # Run blocking work (ffmpeg + whisper) in thread pool
        text = await asyncio.to_thread(_process_voice_audio, audio_bytes, audio.content_type)

        if not text:
            return JSONResponse(
                status_code=200,
                content={"status": "no_speech", "text": ""}
            )

        # Submit as query
        manager.submit_query(text)
        return {"status": "submitted", "text": text}

    except Exception as e:
        logger.error(f"[VoiceQuery] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


def _process_voice_audio(audio_bytes: bytes, content_type: str) -> Optional[str]:
    """Convert audio to WAV and transcribe. Runs in thread pool."""
    import tempfile
    import subprocess

    # Convert to WAV if needed (browser sends WebM/OGG)
    content_type = content_type or ""
    if "wav" not in content_type:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src_f:
            src_f.write(audio_bytes)
            src_path = src_f.name

        wav_path = src_path.replace(".webm", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, timeout=10
            )
            with open(wav_path, "rb") as wav_f:
                audio_bytes = wav_f.read()
            logger.info(f"[VoiceQuery] Converted to WAV: {len(audio_bytes)} bytes")
        finally:
            import os
            for p in [src_path, wav_path]:
                if os.path.exists(p):
                    os.unlink(p)

    # Transcribe
    return manager.transcribe_audio(audio_bytes)


@app.post("/api/system/mute")
async def toggle_mute():
    """Toggle audio mute."""
    muted = manager.toggle_mute()
    return {"muted": muted}


# ──────────────────────────────────────────────
# MJPEG Camera Stream
# ──────────────────────────────────────────────

def _mjpeg_generator():
    """Yield JPEG frames as an MJPEG stream."""
    while True:
        frame = manager.get_annotated_frame()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        else:
            # No frame yet — send a tiny delay
            time.sleep(0.05)


@app.get("/api/camera/feed")
async def camera_feed():
    """MJPEG stream of the annotated camera feed."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ──────────────────────────────────────────────
# WebSocket — Real-time state push
# ──────────────────────────────────────────────

connected_clients: set[WebSocket] = set()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push system state to the frontend every ~200ms."""
    await ws.accept()
    connected_clients.add(ws)
    logger.info(f"[WS] Client connected ({len(connected_clients)} total)")

    try:
        while True:
            state = manager.get_state()
            await ws.send_json(state)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"[WS] Connection error: {e}")
    finally:
        connected_clients.discard(ws)
        logger.info(f"[WS] Client disconnected ({len(connected_clients)} total)")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting WalkSense API server...")
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
