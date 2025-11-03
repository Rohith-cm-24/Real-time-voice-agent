"""
Configuration management for the Voice Agent application.
Handles environment variables and application constants.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration"""

    # API Keys
    DEEPGRAM_API_KEY: Optional[str] = os.getenv("DEEPGRAM_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Application Metadata
    APP_TITLE: str = "Voice Agent: Deepgram STT + Groq LLM + WebRTC"
    VERSION: str = "4.0.0"

    # Deepgram Configuration
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_LANGUAGE: str = "en-US"
    DEEPGRAM_ENCODING: str = "linear16"
    DEEPGRAM_SAMPLE_RATE: int = 48000
    DEEPGRAM_CHANNELS: int = 1
    DEEPGRAM_INTERIM_RESULTS: bool = True
    DEEPGRAM_UTTERANCE_END_MS: str = "1000"
    DEEPGRAM_VAD_EVENTS: bool = True

    # TTS Configuration
    TTS_MODEL: str = "aura-asteria-en"
    TTS_ENCODING: str = "linear16"
    TTS_SAMPLE_RATE: int = 24000
    TTS_CHUNK_SIZE: int = 8192

    # Groq/LLM Configuration
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 150

    # WebRTC Configuration
    WEBRTC_STUN_SERVER: str = "stun:stun.l.google.com:19302"
    WEBRTC_SESSION_TIMEOUT: int = 10  # seconds

    # System Prompts
    SYSTEM_PROMPT: str = (
        "You are a helpful voice assistant. Provide concise, natural responses "
        "as if in a spoken conversation. Keep responses brief and conversational."
    )

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def is_deepgram_configured(cls) -> bool:
        """Check if Deepgram API key is configured"""
        return bool(cls.DEEPGRAM_API_KEY)

    @classmethod
    def is_groq_configured(cls) -> bool:
        """Check if Groq API key is configured"""
        return bool(cls.GROQ_API_KEY)


# Global settings instance
settings = Settings()
