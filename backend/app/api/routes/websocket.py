"""WebSocket routes for real-time audio streaming and transcription.

This module provides WebSocket endpoints for handling real-time audio streaming
and transcription using OpenAI's Whisper API.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
import asyncio
import logging
from openai import AsyncClient, OpenAIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import tempfile
import os
from app.core.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class TemporaryFileManager:
    def __init__(self):
        self.temp_file = None

    async def __aenter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        return self.temp_file

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file:
            self.temp_file.close()
            try:
                os.unlink(self.temp_file.name)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.closed_connections: set = set()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.closed_connections.discard(client_id)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        self.closed_connections.add(client_id)

    async def send_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self.closed_connections:
            return
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except RuntimeError as e:
                if "close message has been sent" in str(e):
                    self.disconnect(client_id)
                else:
                    raise

manager = ConnectionManager()

async def keep_connection_alive(websocket: WebSocket):
    """Keep WebSocket connection alive with ping/pong mechanism."""
    try:
        while True:
            try:
                await websocket.send_json({"type": "ping"})
                await asyncio.sleep(30)  # Send ping every 30 seconds
            except Exception:
                logger.info("WebSocket connection dropped")
                break
    except asyncio.CancelledError:
        logger.info("Keepalive task cancelled")
    except Exception as e:
        logger.error(f"Error in keepalive task: {str(e)}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((OpenAIError, Exception))
)
async def transcribe_with_retry(client: AsyncClient, file_path: str) -> str:
    """Transcribe audio file with retry logic."""
    logger.info(f"Attempting to transcribe file: {file_path}")
    try:
        with open(file_path, "rb") as audio_file:
            logger.info("Sending request to OpenAI API...")
            try:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                logger.info(f"Received response from OpenAI API: {response}")
                return response
            except OpenAIError as e:
                logger.error(f"OpenAI API error details: {str(e)}")
                if hasattr(e, 'response'):
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response headers: {e.response.headers}")
                    logger.error(f"Response body: {e.response.text}")
                raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

@router.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio streaming and transcription."""
    logger.info(f"New WebSocket connection from client: {client_id}")
    
    await manager.connect(websocket, client_id)
    settings = get_settings()
    client = AsyncClient(
        api_key=settings.OPENAI_API_KEY,
        timeout=30.0,  # 30 second timeout
        base_url=settings.AI_SERVICE_URL,
        max_retries=3,
        organization=settings.OPENAI_ORG_ID
    )
    
    # Start connection keepalive task
    keepalive_task = asyncio.create_task(keep_connection_alive(websocket))
    
    try:
        while True:
            # Receive audio data
            audio_data = await websocket.receive_bytes()
            if client_id in manager.closed_connections:
                break
                
            logger.info(f"Received audio data of size: {len(audio_data)} bytes")
            
            try:
                async with TemporaryFileManager() as temp_file:
                    # Write the WAV data directly since it already includes headers
                    temp_file.write(audio_data)
                    temp_file.flush()
                    logger.info(f"Saved audio data to temporary file: {temp_file.name}")
                    
                    try:
                        # Use wait_for instead of timeout
                        response = await asyncio.wait_for(
                            transcribe_with_retry(client, temp_file.name),
                            timeout=30.0  # 30 second timeout
                        )
                        
                        if client_id in manager.closed_connections:
                            break
                            
                        # Send transcription back to client
                        if response:
                            await manager.send_message({
                                "type": "transcription",
                                "text": response
                            }, client_id)
                        else:
                            await manager.send_message({
                                "type": "status",
                                "message": "No speech detected"
                            }, client_id)
                                
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        if client_id not in manager.closed_connections:
                            logger.error("Transcription timed out")
                            await manager.send_message({
                                "type": "error",
                                "message": "Transcription timed out"
                            }, client_id)
                    except OpenAIError as e:
                        if client_id not in manager.closed_connections:
                            logger.error(f"OpenAI API error after retries: {str(e)}")
                            await manager.send_message({
                                "type": "error",
                                "message": f"OpenAI API error: {str(e)}"
                            }, client_id)
                    except Exception as e:
                        if client_id not in manager.closed_connections:
                            logger.error(f"Unexpected error processing audio: {str(e)}")
                            await manager.send_message({
                                "type": "error",
                                "message": str(e)
                            }, client_id)
                        
            except Exception as e:
                if client_id not in manager.closed_connections:
                    logger.error(f"Error handling audio data: {str(e)}")
                    await manager.send_message({
                        "type": "error",
                        "message": str(e)
                    }, client_id)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client: {client_id}")
        manager.disconnect(client_id)
    except asyncio.CancelledError:
        logger.info("WebSocket connection cancelled")
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {str(e)}")
        if client_id not in manager.closed_connections:
            try:
                await manager.send_message({
                    "type": "error",
                    "message": str(e)
                }, client_id)
            except Exception:
                pass
        manager.disconnect(client_id)
    finally:
        # Cancel keepalive task
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
