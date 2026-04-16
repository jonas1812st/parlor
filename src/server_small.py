"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from chat_engine import LlamaChatEngine

import numpy as np
import uvicorn
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.responses import HTMLResponse, JSONResponse
import tempfile
import shutil

# import tts

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

engine = None
tts_backend = None


def load_models():
    global engine, tts_backend
    print("Loading Gemma 4 E2B from llama.cpp-server...")
    engine = LlamaChatEngine(system_prompt=SYSTEM_PROMPT)
    print("Engine loaded.")

    # tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.post("/chat/message")
async def process_audio_message(audio_file: UploadFile = File(...)):
    """
    Receives an audio file, forwards it to the LlamaChatEngine,
    and returns the transcription and model response as JSON.
    """

    if not engine:
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: Chat engine not initialized yet.",
        )

    if not audio_file.filename:
        raise HTTPException(
            status_code=400, detail="Bad Request. Please provide a valid audio file."
        )

    # 1. Check file extension (optional, but recommended for fallbacks)
    file_extension = (
        audio_file.filename.split(".")[-1].lower()
        if "." in audio_file.filename
        else "wav"
    )

    # 2. Create temporary file so it can be passed to the engine
    temp_fd, temp_path = tempfile.mkstemp(suffix=f".{file_extension}")
    os.close(temp_fd)  # Close descriptor immediately because we use shutil

    try:
        # 3. Copy upload stream into the temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # 4. Feed the saved file to our Llama engine
        print(f"Processing audio: {audio_file.filename} ...")
        result = engine.send_message(audio_path=temp_path)

        # 5. Handle errors if engine returns a string instead of a dict
        if isinstance(result, str) and result.startswith(("Error", "API communication error")):
            raise HTTPException(status_code=500, detail=result)

        # 6. Return successful result (FastAPI converts dict to JSON automatically)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # 7. Cleanup: always delete temp file, whether success or failure
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
