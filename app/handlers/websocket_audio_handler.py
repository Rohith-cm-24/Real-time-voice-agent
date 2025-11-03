"""
WebSocket audio streaming handler.
"""
import logging
import asyncio
import json
from typing import Optional
from fastapi import WebSocket

from app.core.config import settings
from app.core.websocket_manager import ConnectionManager
from app.services.deepgram_service import DeepgramService
from app.services.llm_service import LLMService
from app.models.session import LatencyMetrics, AudioStats

logger = logging.getLogger(__name__)


class WebSocketAudioHandler:
    """Handler for WebSocket audio streaming"""

    def __init__(self, websocket: WebSocket, manager: ConnectionManager):
        self.websocket = websocket
        self.manager = manager
        self.loop = asyncio.get_event_loop()

        # Services
        self.deepgram_service = DeepgramService()
        self.llm_service = LLMService()

        # State
        self.audio_stats = AudioStats()
        self.latency_tracker = LatencyMetrics()

    async def handle_start_recording(self) -> None:
        """Initialize services when recording starts"""
        try:
            logger.info("Recording started - initializing services")

            # Initialize LLM
            if self.llm_service.initialize():
                await self.manager.send_message("🤖 Groq LLM ready!", self.websocket)

            # Initialize Deepgram with event handlers
            success = await self.deepgram_service.initialize_stt(
                on_open=self._on_deepgram_open,
                on_message=self._on_deepgram_message,
                on_error=self._on_deepgram_error,
                on_close=self._on_deepgram_close
            )

            if success:
                await self.manager.send_message(
                    "🎤 Deepgram STT ready - speak now!",
                    self.websocket
                )
            else:
                await self.manager.send_message(
                    "❌ Failed to start Deepgram",
                    self.websocket
                )

        except Exception as e:
            logger.error(f"Error initializing services: {e}", exc_info=True)
            await self.manager.send_message(
                f"❌ Init error: {str(e)}",
                self.websocket
            )

    async def handle_stop_recording(self) -> None:
        """Clean up when recording stops"""
        try:
            logger.info("Recording stopped by client")

            self.deepgram_service.finish()

            await self.manager.send_message(
                f"📊 Recording complete: {self.audio_stats.chunks_received} chunks, "
                f"{self.audio_stats.total_bytes} bytes",
                self.websocket
            )

            self.audio_stats.reset()

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

    async def handle_audio_data(self, audio_data: bytes) -> None:
        """Process incoming audio data"""
        self.audio_stats.increment(len(audio_data))

        # Mark first audio chunk
        if self.audio_stats.chunks_received == 1:
            self.latency_tracker.mark("audio_received")

        logger.info(
            f"Received audio chunk #{self.audio_stats.chunks_received}, "
            f"size: {len(audio_data)} bytes"
        )

        # Send to Deepgram
        self.deepgram_service.send_audio(audio_data)

        # Periodic status update
        if self.audio_stats.chunks_received % 20 == 0:
            await self.manager.send_message(
                f"📊 Processed {self.audio_stats.chunks_received} chunks, "
                f"{self.audio_stats.total_bytes} bytes",
                self.websocket
            )

    def _on_deepgram_open(self, *args, **kwargs) -> None:
        """Deepgram connection opened"""
        logger.info("🟢 Deepgram connection opened")
        asyncio.run_coroutine_threadsafe(
            self.manager.send_message(
                "🟢 Deepgram connected and ready to transcribe!",
                self.websocket
            ),
            self.loop
        )

    def _on_deepgram_message(self, *args, **kwargs) -> None:
        """Handle Deepgram transcript"""
        try:
            result = kwargs.get('result', args[0] if args else None)
            if not result:
                return

            sentence = result.channel.alternatives[0].transcript

            if len(sentence) == 0:
                return

            is_final = result.is_final
            confidence = result.channel.alternatives[0].confidence

            if is_final:
                # Mark timestamp and measure latency
                self.latency_tracker.mark("transcript_received")
                stt_latency = self.latency_tracker.measure(
                    "audio_received", "transcript_received", "stt_latency"
                )
                if stt_latency:
                    logger.info(f"📊 STT Latency: {stt_latency:.2f}ms")

                logger.info(
                    f"📝 FINAL transcript: '{sentence}' (confidence: {confidence:.2%})")

                # Send to WebSocket
                asyncio.run_coroutine_threadsafe(
                    self.manager.send_message(
                        f"USER_SAID: {sentence}", self.websocket),
                    self.loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.manager.send_message(
                        f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}",
                        self.websocket
                    ),
                    self.loop
                )

                # Process with LLM
                asyncio.run_coroutine_threadsafe(
                    self._process_with_llm(sentence),
                    self.loop
                )
            else:
                # Interim transcript
                asyncio.run_coroutine_threadsafe(
                    self.manager.send_message(
                        f"INTERIM: {sentence}", self.websocket),
                    self.loop
                )

        except Exception as e:
            logger.error(f"❌ Error processing transcript: {e}", exc_info=True)

    def _on_deepgram_error(self, *args, **kwargs) -> None:
        """Handle Deepgram error"""
        error = kwargs.get('error', args[0] if args else 'Unknown error')
        logger.error(f"🔴 Deepgram error: {error}")
        asyncio.run_coroutine_threadsafe(
            self.manager.send_message(
                f"❌ Deepgram error: {error}", self.websocket),
            self.loop
        )

    def _on_deepgram_close(self, *args, **kwargs) -> None:
        """Deepgram connection closed"""
        logger.info("🔴 Deepgram connection closed")

    async def _process_with_llm(self, transcript: str) -> None:
        """Process transcript with LLM and generate TTS"""
        try:
            logger.info(f"🤖 Processing with LLM: '{transcript}'")

            full_response = ""
            current_sentence = ""
            sentence_count = 0

            async for chunk in self.llm_service.generate_response(
                transcript,
                self.latency_tracker
            ):
                full_response += chunk
                current_sentence += chunk

                # Send chunk to frontend
                await self.manager.send_message(
                    f"LLM_RESPONSE: {chunk}",
                    self.websocket
                )

                # Check for sentence end
                if self.llm_service.detect_sentence_end(chunk):
                    sentence_to_speak = current_sentence.strip()
                    if sentence_to_speak:
                        sentence_count += 1
                        logger.info(
                            f"🔊 Sentence #{sentence_count}: '{sentence_to_speak[:50]}...'")
                        await self._text_to_speech(sentence_to_speak)
                        current_sentence = ""

            # Handle remaining text
            if current_sentence.strip():
                logger.info(
                    f"🔊 Final sentence: '{current_sentence.strip()[:50]}...'")
                await self._text_to_speech(current_sentence.strip())

            logger.info(
                f"🤖 LLM complete: '{full_response}' ({sentence_count} sentences)")

            await self.manager.send_message("LLM_DONE", self.websocket)
            await self.manager.send_message(
                f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}",
                self.websocket
            )

        except Exception as e:
            logger.error(f"❌ Error processing with LLM: {e}", exc_info=True)
            await self.manager.send_message(
                f"❌ LLM error: {str(e)}",
                self.websocket
            )

    async def _text_to_speech(self, text: str) -> None:
        """Convert text to speech"""
        try:
            await self.manager.send_message("TTS_START", self.websocket)

            audio_data = await self.deepgram_service.text_to_speech(
                text,
                self.latency_tracker
            )

            if audio_data:
                chunks = self.deepgram_service.chunk_audio_data(audio_data)
                total_chunks = len(chunks)

                for chunk in chunks:
                    await self.manager.send_message(
                        f"TTS_AUDIO: {chunk}",
                        self.websocket
                    )

                logger.info(
                    f"🔊 TTS complete: Sent {total_chunks} chunks ({len(audio_data)} bytes)")

                await self.manager.send_message("TTS_DONE", self.websocket)
                await self.manager.send_message(
                    f"LATENCY: {json.dumps(self.latency_tracker.get_metrics())}",
                    self.websocket
                )

        except Exception as e:
            logger.error(f"❌ Error in TTS: {e}", exc_info=True)
            await self.manager.send_message(
                f"❌ TTS error: {str(e)}",
                self.websocket
            )
