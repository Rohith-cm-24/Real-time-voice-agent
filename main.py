"""
Voice Agent: Deepgram STT + Groq LLM + WebRTC
Refactored version with modular architecture.
"""
import logging
import json
import uuid
import asyncio
from datetime import datetime
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aiortc import RTCPeerConnection, RTCSessionDescription

from app.core.config import settings
from app.core.websocket_manager import ConnectionManager
from app.models.session import WebRTCSession, LatencyMetrics
from app.handlers.webrtc_handler import AudioTransformTrack
from app.handlers.websocket_audio_handler import WebSocketAudioHandler

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=settings.APP_TITLE, version=settings.VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
connection_manager = ConnectionManager()
active_peer_connections: Set[RTCPeerConnection] = set()
webrtc_sessions: dict[str, WebRTCSession] = {}


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": settings.APP_TITLE,
        "version": settings.VERSION,
        "endpoints": {
            "websocket": "/ws/audio",
            "webrtc_offer": "/webrtc/offer",
            "webrtc_signaling": "/ws/webrtc/{session_id}"
        },
        "deepgram_status": "configured" if settings.is_deepgram_configured() else "not configured",
        "groq_status": "configured" if settings.is_groq_configured() else "not configured",
        "status": "running",
        "active_connections": connection_manager.get_connection_count(),
        "active_webrtc_peers": len(active_peer_connections)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": connection_manager.get_connection_count(),
        "active_webrtc_peers": len(active_peer_connections),
        "deepgram_configured": settings.is_deepgram_configured(),
        "groq_configured": settings.is_groq_configured()
    }


# ============================================================================
# WebRTC Endpoints
# ============================================================================

@app.post("/webrtc/offer")
async def webrtc_offer(request: Request):
    """
    Handle WebRTC offer and return answer.
    Creates a WebRTC peer connection for audio streaming.
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

        # Create session
        session_id = str(uuid.uuid4())
        logger.info(f"🔑 Created session ID: {session_id}")

        # Create RTCPeerConnection
        pc = RTCPeerConnection()
        active_peer_connections.add(pc)

        # Create session object
        latency_tracker = LatencyMetrics()
        session = WebRTCSession(
            session_id=session_id,
            peer_connection=pc,
            latency_tracker=latency_tracker
        )
        webrtc_sessions[session_id] = session

        @pc.on("track")
        async def on_track(track):
            logger.info(f"🎵 Track received: {track.kind}")

            if track.kind == "audio":
                logger.info("🎤 Audio track received, waiting for WebSocket...")

                # Wait for WebSocket connection
                max_wait = settings.WEBRTC_SESSION_TIMEOUT
                wait_interval = 0.1
                waited = 0

                while waited < max_wait:
                    session = webrtc_sessions.get(session_id)
                    if session and session.websocket:
                        break
                    await asyncio.sleep(wait_interval)
                    waited += wait_interval

                session = webrtc_sessions.get(session_id)
                if not session or not session.websocket:
                    logger.error("❌ WebSocket not connected after timeout")
                    return

                logger.info("✅ WebSocket linked, starting audio processing...")

                # Create audio transform track
                audio_track = AudioTransformTrack(
                    track,
                    session.websocket,
                    session.latency_tracker,
                    connection_manager
                )
                session.audio_track = audio_track

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

                            if hasattr(frame, 'samples'):
                                total_bytes += frame.samples * 2

                            if frame_count % 100 == 0:
                                logger.info(
                                    f"📊 Processed {frame_count} frames, ~{total_bytes} bytes")
                                await connection_manager.send_message(
                                    f"STATS: {json.dumps({'chunks': frame_count, 'bytes': total_bytes})}",
                                    session.websocket
                                )
                    except Exception as e:
                        logger.error(
                            f"❌ Error in audio consumer: {e}", exc_info=True)

                # Start consumer task
                asyncio.create_task(consume_audio())

                # Notify frontend
                await connection_manager.send_message(
                    "🎤 Audio streaming started - speak now!",
                    session.websocket
                )

                @track.on("ended")
                async def on_ended():
                    logger.info("🔴 Track ended")
                    await pc.close()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"📡 WebRTC connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                active_peer_connections.discard(pc)
                if session_id in webrtc_sessions:
                    del webrtc_sessions[session_id]

        # Set remote description and create answer
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        )

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("✅ WebRTC answer created")

        return JSONResponse(content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id
        })

    except Exception as e:
        logger.error(f"❌ Error in WebRTC offer: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.websocket("/ws/webrtc/{session_id}")
async def websocket_webrtc_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for WebRTC signaling and control messages.
    """
    await connection_manager.connect(websocket)

    try:
        # Link WebSocket to session
        if session_id in webrtc_sessions:
            webrtc_sessions[session_id].websocket = websocket
            logger.info(f"✅ WebSocket linked to session {session_id}")
            await websocket.send_text("✅ WebRTC signaling channel connected")
        else:
            logger.error(f"❌ Session {session_id} not found")
            await websocket.send_text(f"❌ Error: Session {session_id} not found")
            await websocket.close()
            return

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive()

                if "text" in data:
                    message = data["text"]
                    logger.info(f"WebRTC signaling message: {message}")

                    if message == "ping":
                        await connection_manager.send_message("pong", websocket)
                    else:
                        await connection_manager.send_message(f"Echo: {message}", websocket)
            except Exception as e:
                logger.info(f"WebSocket receive stopped: {e}")
                break

    except WebSocketDisconnect:
        logger.info(
            f"WebRTC signaling channel disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebRTC signaling: {e}", exc_info=True)
    finally:
        # Clean up session
        if session_id in webrtc_sessions:
            session = webrtc_sessions[session_id]
            if session.websocket == websocket:
                session.websocket = None
        connection_manager.disconnect(websocket)


# ============================================================================
# WebSocket Audio Endpoint
# ============================================================================

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving streaming audio data.
    Sends audio to Deepgram STT and processes with Groq LLM.
    """
    await connection_manager.connect(websocket)

    # Create handler
    handler = WebSocketAudioHandler(websocket, connection_manager)

    try:
        # Check API keys
        if not settings.is_deepgram_configured():
            await connection_manager.send_message(
                "⚠️ Warning: DEEPGRAM_API_KEY not configured. Transcription disabled.",
                websocket
            )
            logger.warning(
                "DEEPGRAM_API_KEY not found in environment variables")

        if not settings.is_groq_configured():
            await connection_manager.send_message(
                "⚠️ Warning: GROQ_API_KEY not configured. LLM responses disabled.",
                websocket
            )
            logger.warning("GROQ_API_KEY not found in environment variables")

        if settings.is_deepgram_configured() and settings.is_groq_configured():
            await connection_manager.send_message(
                "✅ Connected: Deepgram STT + Groq LLM ready!",
                websocket
            )

        # Main message loop
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Handle binary audio data
                audio_data = data["bytes"]
                await handler.handle_audio_data(audio_data)

            elif "text" in data:
                # Handle text messages
                message = data["text"]
                logger.info(f"Received text message: {message}")

                if message == "start_recording":
                    await handler.handle_start_recording()

                elif message == "stop_recording":
                    await handler.handle_stop_recording()

                elif message == "ping":
                    await connection_manager.send_message("pong", websocket)

                else:
                    await connection_manager.send_message(f"Echo: {message}", websocket)

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}", exc_info=True)

    finally:
        # Cleanup
        handler.deepgram_service.finish()
        connection_manager.disconnect(websocket)


# ============================================================================
# Application Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
