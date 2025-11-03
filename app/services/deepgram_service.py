"""
Deepgram STT and TTS service integration.
"""
import logging
import base64
import httpx
from typing import Optional, Callable
import asyncio

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

from app.core.config import settings
from app.models.session import LatencyMetrics

logger = logging.getLogger(__name__)


class DeepgramService:
    """Service for Deepgram Speech-to-Text and Text-to-Speech"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.DEEPGRAM_API_KEY
        self.connection = None
        self.client = None

    async def initialize_stt(
        self,
        on_open: Optional[Callable] = None,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        sample_rate: int = 48000,
        encoding: Optional[str] = None
    ) -> bool:
        """
        Initialize Deepgram Speech-to-Text connection.

        Args:
            on_open: Callback for connection open event
            on_message: Callback for transcript messages
            on_error: Callback for errors
            on_close: Callback for connection close
            sample_rate: Audio sample rate
            encoding: Audio encoding format

        Returns:
            bool: True if successfully initialized, False otherwise
        """
        if not self.api_key:
            logger.error("Deepgram API key not configured")
            return False

        try:
            logger.info("🔧 Initializing Deepgram STT connection...")

            config = DeepgramClientOptions(options={"keepalive": "true"})
            self.client = DeepgramClient(self.api_key, config)
            self.connection = self.client.listen.live.v("1")

            # Register event handlers
            if on_open:
                self.connection.on(LiveTranscriptionEvents.Open, on_open)
            if on_message:
                self.connection.on(
                    LiveTranscriptionEvents.Transcript, on_message)
            if on_error:
                self.connection.on(LiveTranscriptionEvents.Error, on_error)
            if on_close:
                self.connection.on(LiveTranscriptionEvents.Close, on_close)

            # Configure options
            options = LiveOptions(
                model=settings.DEEPGRAM_MODEL,
                language=settings.DEEPGRAM_LANGUAGE,
                smart_format=True,
                channels=settings.DEEPGRAM_CHANNELS,
                interim_results=settings.DEEPGRAM_INTERIM_RESULTS,
                utterance_end_ms=settings.DEEPGRAM_UTTERANCE_END_MS,
                vad_events=settings.DEEPGRAM_VAD_EVENTS,
            )

            # Add encoding and sample rate if specified (for WebRTC)
            if encoding:
                options.encoding = encoding
                options.sample_rate = sample_rate

            # Start connection
            result = self.connection.start(options)

            if result:
                logger.info("✅ Deepgram STT initialized successfully")
                return True
            else:
                logger.error("❌ Failed to start Deepgram connection")
                return False

        except Exception as e:
            logger.error(
                f"❌ Error initializing Deepgram STT: {e}", exc_info=True)
            return False

    def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to Deepgram for transcription"""
        if self.connection:
            try:
                self.connection.send(audio_data)
            except Exception as e:
                logger.error(f"Error sending audio to Deepgram: {e}")
        else:
            logger.warning("Deepgram connection not initialized")

    def finish(self) -> None:
        """Close the Deepgram connection"""
        if self.connection:
            try:
                self.connection.finish()
                logger.info("Deepgram connection closed")
            except Exception as e:
                logger.error(f"Error closing Deepgram connection: {e}")
            finally:
                self.connection = None
                self.client = None

    async def text_to_speech(
        self,
        text: str,
        latency_tracker: Optional[LatencyMetrics] = None
    ) -> Optional[bytes]:
        """
        Convert text to speech using Deepgram TTS.

        Args:
            text: Text to convert to speech
            latency_tracker: Optional latency tracker for metrics

        Returns:
            Audio data as bytes, or None if failed
        """
        if not self.api_key:
            logger.warning("Deepgram API key not available for TTS")
            return None

        try:
            if latency_tracker:
                latency_tracker.mark("tts_start")

            logger.info(f"🔊 Converting to speech: '{text[:100]}...'")

            url = (
                f"https://api.deepgram.com/v1/speak?"
                f"model={settings.TTS_MODEL}&"
                f"encoding={settings.TTS_ENCODING}&"
                f"sample_rate={settings.TTS_SAMPLE_RATE}"
            )

            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {"text": text}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )

                if response.status_code != 200:
                    logger.error(
                        f"Deepgram TTS error: {response.status_code} - {response.text}"
                    )
                    return None

                if latency_tracker:
                    latency_tracker.mark("tts_received")
                    tts_latency = latency_tracker.measure(
                        "tts_start", "tts_received", "tts_latency"
                    )
                    if tts_latency:
                        logger.info(f"📊 TTS Latency: {tts_latency:.2f}ms")

                audio_data = response.content
                logger.info(
                    f"🔊 Received {len(audio_data)} bytes of audio from TTS")

                return audio_data

        except Exception as e:
            logger.error(f"❌ Error in TTS: {e}", exc_info=True)
            return None

    @staticmethod
    def chunk_audio_data(audio_data: bytes, chunk_size: Optional[int] = None) -> list:
        """
        Split audio data into chunks and encode as base64.

        Args:
            audio_data: Raw audio bytes
            chunk_size: Size of each chunk (default from settings)

        Returns:
            List of base64-encoded audio chunks
        """
        if chunk_size is None:
            chunk_size = settings.TTS_CHUNK_SIZE

        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            audio_b64 = base64.b64encode(chunk).decode('utf-8')
            chunks.append(audio_b64)

        return chunks
