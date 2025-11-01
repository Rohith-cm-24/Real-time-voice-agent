from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from typing import List, Optional
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent: Deepgram STT + Groq LLM")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/")
async def root():
    """Root endpoint with API information"""
    deepgram_status = "configured" if os.getenv("DEEPGRAM_API_KEY") else "not configured"
    groq_status = "configured" if os.getenv("GROQ_API_KEY") else "not configured"
    return {
        "message": "Voice Agent: Deepgram STT + Groq LLM",
        "version": "3.0.0",
        "websocket_endpoint": "/ws/audio",
        "deepgram_status": deepgram_status,
        "groq_status": groq_status,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "deepgram_configured": bool(os.getenv("DEEPGRAM_API_KEY")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    }


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
    
    # Get event loop for thread-safe operations
    loop = asyncio.get_event_loop()
    
    # Helper function to convert text to speech using Deepgram TTS
    async def text_to_speech(text: str):
        """Convert text to speech using Deepgram Aura TTS"""
        try:
            if not deepgram_api_key:
                logger.warning("Deepgram API key not available for TTS")
                return
            
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
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
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
