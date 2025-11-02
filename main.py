from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import List, Optional, Dict
import asyncio
import os
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions,
)
from groq import Groq
import base64
import httpx
import time
import json
import uuid

# WebRTC imports
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaBlackhole
import av
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent: Deepgram STT + Groq LLM + WebRTC")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebRTC peer connections and their associated WebSockets
pcs = set()
webrtc_sessions = {}  # session_id -> {"pc": peer_connection, "ws": websocket, "latency_tracker": tracker}

# Latency tracking class
class LatencyTracker:
    """Track latency metrics for the voice pipeline"""
    
    def __init__(self):
        self.timestamps: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
    
    def mark(self, event: str):
        """Mark a timestamp for an event"""
        self.timestamps[event] = time.time()
    
    def measure(self, start_event: str, end_event: str, metric_name: str):
        """Measure latency between two events"""
        if start_event in self.timestamps and end_event in self.timestamps:
            latency = (self.timestamps[end_event] - self.timestamps[start_event]) * 1000  # ms
            self.metrics[metric_name] = latency
            return latency
        return None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all measured metrics"""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all timestamps and metrics"""
        self.timestamps.clear()
        self.metrics.clear()


class ConnectionManager:
    """Manages active WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")


manager = ConnectionManager()


class AudioTransformTrack(MediaStreamTrack):
    """
    A WebRTC audio track that processes incoming audio and sends it to Deepgram.
    Also plays back TTS audio responses.
    """
    kind = "audio"

    def __init__(self, track, websocket, latency_tracker: LatencyTracker):
        super().__init__()
        self.track = track
        self.websocket = websocket
        self.latency_tracker = latency_tracker
        self.deepgram_connection = None
        self.groq_client = None
        self.conversation_history = []
        self.loop = asyncio.get_event_loop()
        self.audio_buffer = []
        self.is_processing = False
        
        # Initialize API keys
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize Deepgram and Groq
        asyncio.create_task(self.initialize_services())
    
    async def initialize_services(self):
        """Initialize Deepgram and Groq services"""
        try:
            logger.info("🔧 Initializing services for WebRTC...")
            
            # Wait a bit for websocket to be fully connected
            await asyncio.sleep(0.5)
            
            # Initialize Groq
            if self.groq_api_key:
                try:
                    self.groq_client = Groq(api_key=self.groq_api_key)
                    self.conversation_history.append({
                        "role": "system",
                        "content": "You are a helpful voice assistant. Provide concise, natural responses as if in a spoken conversation."
                    })
                    logger.info("✅ Groq initialized for WebRTC")
                    await self.send_to_websocket("🤖 Groq LLM ready!")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Groq: {e}", exc_info=True)
            else:
                logger.warning("⚠️ GROQ_API_KEY not set")
            
            # Initialize Deepgram
            if self.deepgram_api_key:
                try:
                    logger.info("🔧 Connecting to Deepgram...")
                    config = DeepgramClientOptions(options={"keepalive": "true"})
                    deepgram_client = DeepgramClient(self.deepgram_api_key, config)
                    self.deepgram_connection = deepgram_client.listen.live.v("1")
                    
                    # Set up event handlers
                    self.deepgram_connection.on(LiveTranscriptionEvents.Open, self.on_deepgram_open)
                    self.deepgram_connection.on(LiveTranscriptionEvents.Transcript, self.on_deepgram_message)
                    self.deepgram_connection.on(LiveTranscriptionEvents.Error, self.on_deepgram_error)
                    
                    # Start connection
                    options = LiveOptions(
                        model="nova-2",
                        language="en-US",
                        smart_format=True,
                        channels=1,
                        interim_results=True,
                        utterance_end_ms="1000",
                        vad_events=True,
                    )
                    
                    start_result = self.deepgram_connection.start(options)
                    logger.info(f"🔧 Deepgram start result: {start_result}")
                    
                    if start_result:
                        logger.info("✅ Deepgram STT ready for WebRTC")
                        await self.send_to_websocket("🎤 Deepgram STT ready!")
                    else:
                        logger.error("❌ Deepgram failed to start")
                        await self.send_to_websocket("❌ Deepgram failed to start")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Deepgram: {e}", exc_info=True)
                    await self.send_to_websocket(f"❌ Deepgram error: {str(e)}")
            else:
                logger.warning("⚠️ DEEPGRAM_API_KEY not set")
            
            logger.info("✅ Service initialization complete")
                    
        except Exception as e:
            logger.error(f"❌ Error initializing services: {e}", exc_info=True)
            await self.send_to_websocket(f"❌ Init error: {str(e)}")
    
    async def send_to_websocket(self, message: str):
        """Send message to WebSocket"""
        try:
            if self.websocket and self.websocket.client_state.name == "CONNECTED":
                await self.websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
    
    def on_deepgram_open(self, *args, **kwargs):
        """Deepgram connection opened"""
        logger.info("🟢 Deepgram WebRTC connection opened")
    
    def on_deepgram_message(self, result, **kwargs):
        """Handle Deepgram transcript"""
        try:
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            is_final = result.is_final
            
            if is_final:
                # Mark timestamp for latency tracking
                self.latency_tracker.mark("transcript_received")
                
                # Measure STT latency
                stt_latency = self.latency_tracker.measure("audio_received", "transcript_received", "stt_latency")
                logger.info(f"📊 STT Latency: {stt_latency:.2f}ms")
                
                # Send to WebSocket and LLM
                asyncio.run_coroutine_threadsafe(
                    self.send_to_websocket(f"USER_SAID: {sentence}"),
                    self.loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.send_to_websocket(f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}"),
                    self.loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.process_with_llm(sentence),
                    self.loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    self.send_to_websocket(f"INTERIM: {sentence}"),
                    self.loop
                )
        except Exception as e:
            logger.error(f"❌ Error in Deepgram message handler: {e}", exc_info=True)
    
    def on_deepgram_error(self, error, **kwargs):
        """Handle Deepgram error"""
        logger.error(f"🔴 Deepgram WebRTC error: {error}")
    
    async def process_with_llm(self, transcript: str):
        """Process transcript with Groq LLM"""
        try:
            if not self.groq_client:
                return
            
            self.latency_tracker.mark("llm_start")
            
            self.conversation_history.append({"role": "user", "content": transcript})
            
            stream = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            
            full_response = ""
            current_sentence = ""
            first_token = True
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    
                    if first_token:
                        self.latency_tracker.mark("llm_first_token")
                        llm_latency = self.latency_tracker.measure("llm_start", "llm_first_token", "llm_latency")
                        logger.info(f"📊 LLM Latency (first token): {llm_latency:.2f}ms")
                        first_token = False
                    
                    full_response += content
                    current_sentence += content
                    
                    await self.send_to_websocket(f"LLM_RESPONSE: {content}")
                    
                    if any(punct in content for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                        sentence_to_speak = current_sentence.strip()
                        if sentence_to_speak:
                            await self.text_to_speech(sentence_to_speak)
                            current_sentence = ""
            
            if current_sentence.strip():
                await self.text_to_speech(current_sentence.strip())
            
            self.conversation_history.append({"role": "assistant", "content": full_response})
            await self.send_to_websocket("LLM_DONE")
            await self.send_to_websocket(f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}")
            
        except Exception as e:
            logger.error(f"❌ Error processing with LLM: {e}", exc_info=True)
    
    async def text_to_speech(self, text: str):
        """Convert text to speech using Deepgram TTS"""
        try:
            if not self.deepgram_api_key:
                return
            
            self.latency_tracker.mark("tts_start")
            await self.send_to_websocket("TTS_START")
            
            url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
                "Content-Type": "application/json"
            }
            payload = {"text": text}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    self.latency_tracker.mark("tts_received")
                    tts_latency = self.latency_tracker.measure("tts_start", "tts_received", "tts_latency")
                    logger.info(f"📊 TTS Latency: {tts_latency:.2f}ms")
                    
                    audio_data = response.content
                    chunk_size = 8192
                    
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        audio_b64 = base64.b64encode(chunk).decode('utf-8')
                        await self.send_to_websocket(f"TTS_AUDIO: {audio_b64}")
                    
                    await self.send_to_websocket("TTS_DONE")
                    await self.send_to_websocket(f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}")
        except Exception as e:
            logger.error(f"❌ Error in TTS: {e}", exc_info=True)
    
    async def recv(self):
        """Receive and process audio frames from WebRTC"""
        frame = await self.track.recv()
        
        # Mark timestamp for first audio
        if not self.latency_tracker.timestamps.get("audio_received"):
            self.latency_tracker.mark("audio_received")
            logger.info("🎤 First audio frame received!")
        
        # Convert frame to bytes and send to Deepgram
        if self.deepgram_connection:
            try:
                # Convert audio frame to raw PCM bytes
                audio_array = frame.to_ndarray()
                
                # Log frame info for first few frames
                if not hasattr(self, '_frame_count'):
                    self._frame_count = 0
                self._frame_count += 1
                
                if self._frame_count <= 3:
                    logger.info(f"📊 Frame #{self._frame_count}: shape={audio_array.shape}, dtype={audio_array.dtype}, samples={len(audio_array)}")
                
                # Convert to 16-bit PCM
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)
                
                audio_bytes = audio_array.tobytes()
                
                if self._frame_count <= 3:
                    logger.info(f"🔊 Sending {len(audio_bytes)} bytes to Deepgram")
                
                # Send to Deepgram
                self.deepgram_connection.send(audio_bytes)
                
            except Exception as e:
                logger.error(f"❌ Error processing audio frame: {e}", exc_info=True)
        else:
            if not hasattr(self, '_warned_no_deepgram'):
                self._warned_no_deepgram = True
                logger.warning("⚠️ Deepgram connection not initialized, audio frames not being processed")
        
        return frame


@app.get("/")
async def root():
    """Root endpoint with API information"""
    deepgram_status = "configured" if os.getenv("DEEPGRAM_API_KEY") else "not configured"
    groq_status = "configured" if os.getenv("GROQ_API_KEY") else "not configured"
    return {
        "message": "Voice Agent: Deepgram STT + Groq LLM + WebRTC",
        "version": "4.0.0",
        "endpoints": {
            "websocket": "/ws/audio",
            "webrtc_offer": "/webrtc/offer"
        },
        "deepgram_status": deepgram_status,
        "groq_status": groq_status,
        "status": "running",
        "active_connections": len(manager.active_connections),
        "active_webrtc_peers": len(pcs)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "active_webrtc_peers": len(pcs),
        "deepgram_configured": bool(os.getenv("DEEPGRAM_API_KEY")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    }


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request):
    """
    Handle WebRTC offer and return answer.
    This endpoint creates a WebRTC peer connection for audio streaming.
    """
    try:
        data = await request.json()
        offer_sdp = data.get("sdp")
        offer_type = data.get("type")
        
        if not offer_sdp or offer_type != "offer":
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid offer"}
            )
        
        logger.info("📞 Received WebRTC offer")
        
        # Create session ID to link peer connection with WebSocket
        session_id = str(uuid.uuid4())
        logger.info(f"🔑 Created session ID: {session_id}")
        
        # Create RTCPeerConnection
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        # Create latency tracker for this connection
        latency_tracker = LatencyTracker()
        
        # Store session (WebSocket will be added later when it connects)
        webrtc_sessions[session_id] = {
            "pc": pc,
            "ws": None,
            "latency_tracker": latency_tracker,
            "audio_track": None
        }
        
        @pc.on("track")
        async def on_track(track):
            logger.info(f"🎵 Track received: {track.kind}")
            
            if track.kind == "audio":
                logger.info("🎤 Audio track received, waiting for WebSocket connection...")
                
                # Wait for WebSocket to be connected (with timeout)
                max_wait = 10  # seconds
                wait_interval = 0.1
                waited = 0
                
                while waited < max_wait:
                    if webrtc_sessions.get(session_id, {}).get("ws"):
                        break
                    await asyncio.sleep(wait_interval)
                    waited += wait_interval
                
                websocket = webrtc_sessions.get(session_id, {}).get("ws")
                
                if not websocket:
                    logger.error("❌ WebSocket not connected after timeout")
                    return
                
                logger.info("✅ WebSocket linked, starting audio processing...")
                
                # Create audio transform track
                audio_track = AudioTransformTrack(track, websocket, latency_tracker)
                webrtc_sessions[session_id]["audio_track"] = audio_track
                
                # Start consuming audio frames
                async def consume_audio():
                    """Continuously read and process audio frames"""
                    try:
                        logger.info("🎤 Starting audio frame consumer...")
                        frame_count = 0
                        total_bytes = 0
                        while True:
                            frame = await audio_track.recv()
                            frame_count += 1
                            
                            # Estimate bytes processed
                            if hasattr(frame, 'samples'):
                                total_bytes += frame.samples * 2  # 16-bit = 2 bytes per sample
                            
                            if frame_count % 100 == 0:  # Log every 100 frames
                                logger.info(f"📊 Processed {frame_count} audio frames, ~{total_bytes} bytes")
                                # Send stats to frontend
                                await manager.send_message(
                                    f"STATS: {json.dumps({'chunks': frame_count, 'bytes': total_bytes})}",
                                    websocket
                                )
                    except Exception as e:
                        logger.error(f"❌ Error in audio consumer: {e}", exc_info=True)
                
                # Start the consumer task
                asyncio.create_task(consume_audio())
                
                # Notify frontend that audio is being processed
                await manager.send_message("🎤 Audio streaming started - speak now!", websocket)
                
                # Keep the track alive
                @track.on("ended")
                async def on_ended():
                    logger.info("🔴 Track ended")
                    await pc.close()
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"📡 WebRTC connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                pcs.discard(pc)
                if session_id in webrtc_sessions:
                    del webrtc_sessions[session_id]
        
        # Set remote description
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        logger.info("✅ WebRTC answer created")
        
        return JSONResponse(content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id  # Return session ID to client
        })
        
    except Exception as e:
        logger.error(f"❌ Error in WebRTC offer: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# WebRTC signaling WebSocket endpoint
@app.websocket("/ws/webrtc/{session_id}")
async def websocket_webrtc_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for WebRTC signaling and control messages.
    This is used alongside the WebRTC peer connection for sending transcripts and responses.
    """
    await manager.connect(websocket)
    
    try:
        # Link WebSocket to session
        if session_id in webrtc_sessions:
            webrtc_sessions[session_id]["ws"] = websocket
            logger.info(f"✅ WebSocket linked to session {session_id}")
            await websocket.send_text("✅ WebRTC signaling channel connected")
        else:
            logger.error(f"❌ Session {session_id} not found")
            await websocket.send_text(f"❌ Error: Session {session_id} not found")
            await websocket.close()
            return
        
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                message = data["text"]
                logger.info(f"WebRTC signaling message: {message}")
                
                if message == "ping":
                    await manager.send_message("pong", websocket)
                else:
                    await manager.send_message(f"Echo: {message}", websocket)
    
    except WebSocketDisconnect:
        logger.info(f"WebRTC signaling channel disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebRTC signaling: {e}", exc_info=True)
    finally:
        # Clean up session
        if session_id in webrtc_sessions:
            if webrtc_sessions[session_id]["ws"] == websocket:
                webrtc_sessions[session_id]["ws"] = None
        manager.disconnect(websocket)


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving streaming audio data and sending to Deepgram STT.
    
    The frontend sends audio data as binary chunks (bytes).
    Server streams to Deepgram and returns transcriptions.
    """
    await manager.connect(websocket)
    
    # Deepgram connection
    deepgram_connection = None
    deepgram_client = None
    
    # Groq client and conversation history
    groq_client = None
    conversation_history = []
    
    audio_chunks_received = 0
    total_bytes_received = 0
    
    # Latency tracking
    latency_tracker = LatencyTracker()
    
    # Get event loop for thread-safe operations
    loop = asyncio.get_event_loop()
    
    # Helper function to convert text to speech using Deepgram TTS
    async def text_to_speech(text: str):
        """Convert text to speech using Deepgram Aura TTS"""
        try:
            if not deepgram_api_key:
                logger.warning("Deepgram API key not available for TTS")
                return
            
            latency_tracker.mark("tts_start")
            
            logger.info("=" * 60)
            logger.info(f"🔊 CONVERTING TO SPEECH:")
            logger.info(f"   Text: '{text[:100]}...'")
            logger.info("=" * 60)
            
            # Signal that TTS is starting
            await manager.send_message("TTS_START", websocket)
            
            # Use Deepgram TTS API via HTTP
            url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000"
            headers = {
                "Authorization": f"Token {deepgram_api_key}",
                "Content-Type": "application/json"
            }
            payload = {"text": text}
            
            # Make async HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                
                if response.status_code != 200:
                    logger.error(f"Deepgram TTS error: {response.status_code} - {response.text}")
                    await manager.send_message(f"❌ TTS API error: {response.status_code}", websocket)
                    return
                
                latency_tracker.mark("tts_received")
                tts_latency = latency_tracker.measure("tts_start", "tts_received", "tts_latency")
                logger.info(f"📊 TTS Latency: {tts_latency:.2f}ms")
                
                # Get the audio data
                audio_data = response.content
                
                logger.info(f"🔊 Received {len(audio_data)} bytes of audio from Deepgram TTS")
                
                # Split into chunks and send to client
                chunk_size = 8192  # 8KB chunks
                total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    # Encode audio chunk as base64 for WebSocket transmission
                    audio_b64 = base64.b64encode(chunk).decode('utf-8')
                    await manager.send_message(f"TTS_AUDIO: {audio_b64}", websocket)
                
                logger.info(f"🔊 TTS complete: Sent {total_chunks} audio chunks ({len(audio_data)} bytes)")
            
            # Signal that TTS is complete
            await manager.send_message("TTS_DONE", websocket)
            await manager.send_message(f"LATENCY: {json.dumps(latency_tracker.get_metrics())}", websocket)
            
        except Exception as e:
            logger.error(f"❌ Error in TTS: {e}", exc_info=True)
            await manager.send_message(f"❌ TTS error: {str(e)}", websocket)
    
    # Helper function to process transcript with Groq LLM with streaming TTS
    async def process_with_llm(transcript: str):
        """Send transcript to Groq LLM and stream response with TTS"""
        try:
            if not groq_client:
                logger.warning("Groq client not initialized")
                return
            
            latency_tracker.mark("llm_start")
            
            # Add user message to conversation history
            conversation_history.append({
                "role": "user",
                "content": transcript
            })
            
            logger.info("=" * 60)
            logger.info(f"🤖 SENDING TO GROQ LLM:")
            logger.info(f"   User: '{transcript}'")
            logger.info("=" * 60)
            
            # Send to Groq and stream response
            stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",  # Fast, high-quality OSS model
                messages=conversation_history,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            
            # Stream response and convert to speech sentence by sentence
            full_response = ""
            current_sentence = ""
            sentence_count = 0
            first_token = True
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    
                    if first_token:
                        latency_tracker.mark("llm_first_token")
                        llm_latency = latency_tracker.measure("llm_start", "llm_first_token", "llm_latency")
                        logger.info(f"📊 LLM Latency (first token): {llm_latency:.2f}ms")
                        first_token = False
                    
                    full_response += content
                    current_sentence += content
                    
                    # Send each chunk to frontend for text display
                    await manager.send_message(f"LLM_RESPONSE: {content}", websocket)
                    
                    # Check if we have a complete sentence (ends with . ! ? or newline)
                    if any(punct in content for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                        # Found sentence end - convert to speech immediately!
                        sentence_to_speak = current_sentence.strip()
                        
                        if sentence_to_speak:
                            sentence_count += 1
                            logger.info(f"🔊 Sentence #{sentence_count} complete: '{sentence_to_speak[:50]}...'")
                            
                            # Convert this sentence to speech immediately (streaming!)
                            await text_to_speech(sentence_to_speak)
                            
                            # Reset for next sentence
                            current_sentence = ""
            
            # Handle any remaining text (last sentence might not end with punctuation)
            if current_sentence.strip():
                logger.info(f"🔊 Final sentence: '{current_sentence.strip()[:50]}...'")
                await text_to_speech(current_sentence.strip())
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            logger.info("=" * 60)
            logger.info(f"🤖 LLM RESPONSE COMPLETE:")
            logger.info(f"   Assistant: '{full_response}'")
            logger.info(f"   Sentences spoken: {sentence_count}")
            logger.info("=" * 60)
            
            # Signal end of text response
            await manager.send_message("LLM_DONE", websocket)
            await manager.send_message(f"LATENCY: {json.dumps(latency_tracker.get_metrics())}", websocket)
            
        except Exception as e:
            logger.error(f"❌ Error processing with Groq: {e}", exc_info=True)
            await manager.send_message(f"❌ LLM error: {str(e)}", websocket)
    
    try:
        # Check if Deepgram API key is configured
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not deepgram_api_key:
            await manager.send_message(
                "⚠️ Warning: DEEPGRAM_API_KEY not configured. Transcription disabled.",
                websocket
            )
            logger.warning("DEEPGRAM_API_KEY not found in environment variables")
        
        if not groq_api_key:
            await manager.send_message(
                "⚠️ Warning: GROQ_API_KEY not configured. LLM responses disabled.",
                websocket
            )
            logger.warning("GROQ_API_KEY not found in environment variables")
        
        if deepgram_api_key and groq_api_key:
            await manager.send_message("✅ Connected: Deepgram STT + Groq LLM ready!", websocket)
        
        while True:
            # Receive data from the WebSocket
            data = await websocket.receive()
            
            if "bytes" in data:
                # Handle binary audio data
                audio_data = data["bytes"]
                audio_chunks_received += 1
                total_bytes_received += len(audio_data)
                
                # Mark timestamp for first audio chunk (for latency tracking)
                if audio_chunks_received == 1:
                    latency_tracker.mark("audio_received")
                
                logger.info(f"Received audio chunk #{audio_chunks_received}, size: {len(audio_data)} bytes")
                
                # Send audio to Deepgram if connected
                if deepgram_connection:
                    try:
                        deepgram_connection.send(audio_data)
                        if audio_chunks_received % 10 == 0:  # Log every 10 chunks
                            logger.info(f"✅ Sent chunk #{audio_chunks_received} to Deepgram ({len(audio_data)} bytes)")
                    except Exception as e:
                        logger.error(f"❌ Error sending to Deepgram: {e}")
                        await manager.send_message(f"❌ Deepgram error: {str(e)}", websocket)
                else:
                    if audio_chunks_received == 1:
                        logger.warning("⚠️ Deepgram connection not initialized, audio not being transcribed")
                
                # Send periodic acknowledgment
                if audio_chunks_received % 20 == 0:
                    await manager.send_message(
                        f"📊 Processed {audio_chunks_received} chunks, {total_bytes_received} bytes",
                        websocket
                    )
            
            elif "text" in data:
                # Handle text messages (control messages, metadata, etc.)
                message = data["text"]
                logger.info(f"Received text message: {message}")
                
                if message == "start_recording":
                    logger.info("Recording started by client - initializing Deepgram and Groq")
                    
                    # Initialize Groq client
                    if groq_api_key:
                        try:
                            logger.info(f"🔧 Initializing Groq with API key: {groq_api_key[:10]}...")
                            groq_client = Groq(api_key=groq_api_key)
                            
                            # Initialize conversation with system prompt
                            conversation_history.clear()
                            conversation_history.append({
                                "role": "system",
                                "content": "You are a helpful voice assistant. Provide concise, natural responses as if in a spoken conversation. Keep responses brief and conversational."
                            })
                            
                            logger.info("✅ Groq LLM initialized successfully")
                            await manager.send_message("🤖 Groq LLM ready!", websocket)
                        except Exception as e:
                            logger.error(f"❌ Failed to initialize Groq: {e}", exc_info=True)
                            await manager.send_message(f"❌ Groq init failed: {str(e)}", websocket)
                    else:
                        logger.warning("⚠️ groq_api_key is None or empty - cannot initialize Groq client")
                        await manager.send_message("⚠️ Groq API key not found", websocket)
                    
                    # Initialize Deepgram client
                    if deepgram_api_key:
                        try:
                            config = DeepgramClientOptions(
                                options={"keepalive": "true"}
                            )
                            deepgram_client = DeepgramClient(deepgram_api_key, config)
                            
                            # Create live transcription connection
                            deepgram_connection = deepgram_client.listen.live.v("1")
                            
                            # Set up event handlers
                            def on_open(self, open_event, **kwargs):
                                logger.info("=" * 60)
                                logger.info("🟢 DEEPGRAM CONNECTION OPENED")
                                logger.info(f"   Event: {open_event}")
                                logger.info("   Status: Ready to receive audio and send transcripts")
                                logger.info("=" * 60)
                                # Use thread-safe method to send message
                                asyncio.run_coroutine_threadsafe(
                                    manager.send_message(
                                        "🟢 Deepgram connected and ready to transcribe!",
                                        websocket
                                    ),
                                    loop
                                )
                            
                            def on_message(self, result, **kwargs):
                                try:
                                    # Log the full result for debugging
                                    logger.debug(f"🔍 RAW DEEPGRAM RESULT: {result}")
                                    
                                    sentence = result.channel.alternatives[0].transcript
                                    
                                    # Log even empty results for debugging
                                    if len(sentence) == 0:
                                        logger.debug("📭 Empty transcript received (silence or processing)")
                                    else:
                                        is_final = result.is_final
                                        confidence = result.channel.alternatives[0].confidence
                                        
                                        if is_final:
                                            # Mark timestamp for latency tracking
                                            latency_tracker.mark("transcript_received")
                                            
                                            # Measure STT latency
                                            stt_latency = latency_tracker.measure("audio_received", "transcript_received", "stt_latency")
                                            if stt_latency:
                                                logger.info(f"📊 STT Latency: {stt_latency:.2f}ms")
                                        
                                        # Detailed console logging
                                        logger.info("=" * 60)
                                        logger.info(f"📝 DEEPGRAM TRANSCRIPT RECEIVED:")
                                        logger.info(f"   Type: {'FINAL' if is_final else 'INTERIM'}")
                                        logger.info(f"   Text: '{sentence}'")
                                        logger.info(f"   Confidence: {confidence:.2%}" if confidence else "   Confidence: N/A")
                                        logger.info("=" * 60)
                                        
                                        if is_final:
                                            # FINAL transcript - send to LLM for response
                                            logger.info("🎯 Final transcript detected - sending to Groq LLM")
                                            
                                            # Show user's final transcript in UI
                                            asyncio.run_coroutine_threadsafe(
                                                manager.send_message(
                                                    f"USER_SAID: {sentence}",
                                                    websocket
                                                ),
                                                loop
                                            )
                                            
                                            # Send latency metrics
                                            asyncio.run_coroutine_threadsafe(
                                                manager.send_message(
                                                    f"LATENCY: {json.dumps(latency_tracker.get_metrics())}",
                                                    websocket
                                                ),
                                                loop
                                            )
                                            
                                            # Process with LLM (run in event loop)
                                            asyncio.run_coroutine_threadsafe(
                                                process_with_llm(sentence),
                                                loop
                                            )
                                        else:
                                            # INTERIM transcript - show for real-time feedback (optional)
                                            asyncio.run_coroutine_threadsafe(
                                                manager.send_message(
                                                    f"INTERIM: {sentence}",
                                                    websocket
                                                ),
                                                loop
                                            )
                                except AttributeError as e:
                                    logger.error(f"❌ Error parsing transcript result: {e}")
                                    logger.error(f"📄 Result object type: {type(result)}")
                                    logger.error(f"📄 Result object: {result}")
                                except Exception as e:
                                    logger.error(f"❌ Error processing transcript: {e}", exc_info=True)
                            
                            def on_metadata(self, metadata, **kwargs):
                                logger.info(f"📊 DEEPGRAM METADATA: {metadata}")
                                logger.debug(f"   Full metadata object: {metadata}")
                            
                            def on_error(self, error, **kwargs):
                                logger.error("=" * 60)
                                logger.error(f"🔴 DEEPGRAM ERROR:")
                                logger.error(f"   Error: {error}")
                                logger.error(f"   Type: {type(error)}")
                                logger.error(f"   Kwargs: {kwargs}")
                                logger.error("=" * 60)
                                # Use thread-safe method to send message
                                asyncio.run_coroutine_threadsafe(
                                    manager.send_message(
                                        f"❌ Deepgram error: {error}",
                                        websocket
                                    ),
                                    loop
                                )
                            
                            def on_close(self, close_event, **kwargs):
                                logger.info("=" * 60)
                                logger.info("🔴 DEEPGRAM CONNECTION CLOSED")
                                logger.info(f"   Event: {close_event}")
                                logger.info("=" * 60)
                            
                            # Register event handlers
                            deepgram_connection.on(LiveTranscriptionEvents.Open, on_open)
                            deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message)
                            deepgram_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
                            deepgram_connection.on(LiveTranscriptionEvents.Error, on_error)
                            deepgram_connection.on(LiveTranscriptionEvents.Close, on_close)
                            
                            # Configure live transcription options
                            # Note: For browser MediaRecorder (webm/opus), don't specify encoding
                            # Deepgram will auto-detect the format
                            options = LiveOptions(
                                model="nova-2",
                                language="en-US",
                                smart_format=True,
                                # encoding="linear16",  # Removed - let Deepgram auto-detect webm/opus
                                # sample_rate=16000,    # Removed - browser determines this
                                channels=1,
                                interim_results=True,
                                utterance_end_ms="1000",
                                vad_events=True,
                            )
                            
                            # Start the connection
                            logger.info("=" * 60)
                            logger.info("🚀 STARTING DEEPGRAM CONNECTION...")
                            logger.info(f"   Model: {options.model}")
                            logger.info(f"   Language: {options.language}")
                            logger.info(f"   Channels: {options.channels}")
                            logger.info(f"   Interim Results: {options.interim_results}")
                            logger.info("=" * 60)
                            
                            if deepgram_connection.start(options) is False:
                                logger.error("❌ Failed to start Deepgram connection")
                                await manager.send_message("❌ Failed to start Deepgram", websocket)
                            else:
                                logger.info("✅ Deepgram connection started successfully")
                                logger.info("🎤 Waiting for 🟢 OPEN event and audio data...")
                                await manager.send_message("🎤 Deepgram STT ready - speak now!", websocket)
                        
                        except Exception as e:
                            logger.error(f"Error initializing Deepgram: {e}")
                            await manager.send_message(f"❌ Deepgram init error: {str(e)}", websocket)
                    else:
                        await manager.send_message("⚠️ Server ready (STT disabled - no API key)", websocket)
                
                elif message == "stop_recording":
                    logger.info("Recording stopped by client")
                    
                    # Close Deepgram connection
                    if deepgram_connection:
                        try:
                            deepgram_connection.finish()
                            logger.info("Deepgram connection finished")
                            await manager.send_message("✅ Transcription session ended", websocket)
                        except Exception as e:
                            logger.error(f"Error closing Deepgram: {e}")
                        finally:
                            deepgram_connection = None
                            deepgram_client = None
                    
                    await manager.send_message(
                        f"📊 Recording complete: {audio_chunks_received} chunks, {total_bytes_received} bytes",
                        websocket
                    )
                    audio_chunks_received = 0
                    total_bytes_received = 0
                
                elif message == "ping":
                    await manager.send_message("pong", websocket)
                
                else:
                    await manager.send_message(f"Echo: {message}", websocket)
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected. Chunks received: {audio_chunks_received}")
    
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}", exc_info=True)
    
    finally:
        # Clean up Deepgram connection
        if deepgram_connection:
            try:
                deepgram_connection.finish()
                logger.info("Deepgram connection closed on cleanup")
            except Exception as e:
                logger.error(f"Error closing Deepgram on cleanup: {e}")
        
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
