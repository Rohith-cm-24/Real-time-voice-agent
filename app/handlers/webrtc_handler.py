"""
WebRTC audio track handler and processing.
"""
import logging
import asyncio
import json
import numpy as np
from typing import Optional
from fastapi import WebSocket

from aiortc import MediaStreamTrack
import av

from app.core.config import settings
from app.models.session import LatencyMetrics
from app.services.deepgram_service import DeepgramService
from app.services.llm_service import LLMService
from app.core.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)


class AudioTransformTrack(MediaStreamTrack):
    """
    A WebRTC audio track that processes incoming audio and sends it to Deepgram.
    Also handles LLM processing and TTS responses.
    """
    kind = "audio"

    def __init__(
        self,
        track: MediaStreamTrack,
        websocket: WebSocket,
        latency_tracker: LatencyMetrics,
        manager: ConnectionManager
    ):
        super().__init__()
        self.track = track
        self.websocket = websocket
        self.latency_tracker = latency_tracker
        self.manager = manager
        self.loop = asyncio.get_event_loop()

        # Services
        self.deepgram_service = DeepgramService()
        self.llm_service = LLMService()

        # State
        self.is_processing = False
        self._frame_count = 0

        # Initialize services
        asyncio.create_task(self.initialize_services())

    async def send_to_websocket(self, message: str) -> None:
        """Send message to WebSocket"""
        try:
            if self.websocket and self.websocket.client_state.name == "CONNECTED":
                await self.manager.send_message(message, self.websocket)
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")

    async def initialize_services(self) -> None:
        """Initialize Deepgram and Groq services"""
        try:
            logger.info("🔧 Initializing services for WebRTC...")
            await asyncio.sleep(0.5)  # Wait for websocket

            # Initialize LLM
            if self.llm_service.initialize():
                await self.send_to_websocket("🤖 Groq LLM ready!")

            # Initialize Deepgram STT
            success = await self.deepgram_service.initialize_stt(
                on_open=self.on_deepgram_open,
                on_message=self.on_deepgram_message,
                on_error=self.on_deepgram_error,
                sample_rate=48000,
                encoding=settings.DEEPGRAM_ENCODING
            )

            if success:
                await self.send_to_websocket("🎤 Deepgram STT ready!")
            else:
                await self.send_to_websocket("❌ Deepgram failed to start")

            logger.info("✅ Service initialization complete")

        except Exception as e:
            logger.error(f"❌ Error initializing services: {e}", exc_info=True)
            await self.send_to_websocket(f"❌ Init error: {str(e)}")

    def on_deepgram_open(self, *args, **kwargs) -> None:
        """Deepgram connection opened"""
        logger.info("🟢 Deepgram WebRTC connection opened")
        asyncio.run_coroutine_threadsafe(
            self.send_to_websocket("🟢 Deepgram connected"),
            self.loop
        )

    def on_deepgram_message(self, *args, **kwargs) -> None:
        """Handle Deepgram transcript"""
        try:
            result = kwargs.get('result', args[0] if args else None)
            if not result:
                return

            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return

            is_final = result.is_final

            if is_final:
                self.latency_tracker.mark("transcript_received")
                stt_latency = self.latency_tracker.measure(
                    "audio_received", "transcript_received", "stt_latency"
                )
                logger.info(f"📊 STT Latency: {stt_latency:.2f}ms")

                # Send to WebSocket and LLM
                asyncio.run_coroutine_threadsafe(
                    self.send_to_websocket(f"USER_SAID: {sentence}"),
                    self.loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.send_to_websocket(
                        f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}"
                    ),
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
            logger.error(
                f"❌ Error in Deepgram message handler: {e}", exc_info=True)

    def on_deepgram_error(self, *args, **kwargs) -> None:
        """Handle Deepgram error"""
        error = kwargs.get('error', args[0] if args else 'Unknown error')
        logger.error(f"🔴 Deepgram WebRTC error: {error}")
        asyncio.run_coroutine_threadsafe(
            self.send_to_websocket(f"❌ STT error: {error}"),
            self.loop
        )

    async def process_with_llm(self, transcript: str) -> None:
        """Process transcript with Groq LLM"""
        try:
            await self.send_to_websocket("TTS_START")

            full_response = ""
            current_sentence = ""

            async for chunk in self.llm_service.generate_response(
                transcript,
                self.latency_tracker
            ):
                full_response += chunk
                current_sentence += chunk

                await self.send_to_websocket(f"LLM_RESPONSE: {chunk}")

                # Check for sentence end
                if self.llm_service.detect_sentence_end(chunk):
                    sentence_to_speak = current_sentence.strip()
                    if sentence_to_speak:
                        await self.text_to_speech(sentence_to_speak)
                        current_sentence = ""

            # Speak remaining text
            if current_sentence.strip():
                await self.text_to_speech(current_sentence.strip())

            await self.send_to_websocket("LLM_DONE")
            await self.send_to_websocket(
                f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}"
            )

        except Exception as e:
            logger.error(f"❌ Error processing with LLM: {e}", exc_info=True)

    async def text_to_speech(self, text: str) -> None:
        """Convert text to speech using Deepgram TTS"""
        try:
            audio_data = await self.deepgram_service.text_to_speech(
                text,
                self.latency_tracker
            )

            if audio_data:
                chunks = self.deepgram_service.chunk_audio_data(audio_data)
                for chunk in chunks:
                    await self.send_to_websocket(f"TTS_AUDIO: {chunk}")

                await self.send_to_websocket("TTS_DONE")
                await self.send_to_websocket(
                    f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}"
                )
        except Exception as e:
            logger.error(f"❌ Error in TTS: {e}", exc_info=True)

    async def recv(self) -> av.AudioFrame:
        """Receive and process audio frames from WebRTC"""
        frame = await self.track.recv()

        # Mark timestamp for first audio
        if not self.latency_tracker.timestamps.get("audio_received"):
            self.latency_tracker.mark("audio_received")
            await self.send_to_websocket("🎤 Audio streaming started")
            logger.info(
                f"🎤 First audio frame: format={frame.format}, "
                f"samples={frame.samples}, rate={frame.sample_rate}"
            )

        # Process and send to Deepgram
        if self.deepgram_service.connection:
            try:
                audio_array = frame.to_ndarray()
                self._frame_count += 1

                # Convert to 16-bit PCM
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)

                # Handle stereo to mono conversion
                if len(audio_array.shape) > 1 and audio_array.shape[0] == 2:
                    audio_array = np.mean(audio_array, axis=0).astype(np.int16)
                elif len(audio_array.shape) > 1 and audio_array.shape[0] == 1:
                    audio_array = audio_array.flatten()
                    if len(audio_array) != frame.samples:
                        audio_array = audio_array[::2]

                audio_bytes = audio_array.tobytes()
                self.deepgram_service.send_audio(audio_bytes)

                # Periodic logging
                if self._frame_count % 100 == 0:
                    logger.info(
                        f"📊 Sent {self._frame_count} frames to Deepgram")

            except Exception as e:
                logger.error(
                    f"❌ Error processing frame #{self._frame_count}: {e}")

        return frame
