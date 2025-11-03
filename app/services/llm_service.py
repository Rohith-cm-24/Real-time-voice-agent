"""
Groq LLM service integration.
"""
import logging
from typing import Optional, List, Dict, AsyncGenerator
from groq import Groq

from app.core.config import settings
from app.models.session import LatencyMetrics, ConversationMessage

logger = logging.getLogger(__name__)


class LLMService:
    """Service for Groq LLM integration"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.client: Optional[Groq] = None
        self.conversation_history: List[Dict[str, str]] = []

    def initialize(self, system_prompt: Optional[str] = None) -> bool:
        """
        Initialize the Groq client and conversation history.

        Args:
            system_prompt: Optional custom system prompt

        Returns:
            bool: True if successfully initialized
        """
        if not self.api_key:
            logger.error("Groq API key not configured")
            return False

        try:
            self.client = Groq(api_key=self.api_key)

            # Initialize conversation with system prompt
            self.conversation_history = [{
                "role": "system",
                "content": system_prompt or settings.SYSTEM_PROMPT
            }]

            logger.info("✅ Groq LLM initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq: {e}", exc_info=True)
            return False

    def add_user_message(self, content: str) -> None:
        """Add a user message to conversation history"""
        self.conversation_history.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to conversation history"""
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })

    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        Clear conversation history.

        Args:
            keep_system_prompt: If True, keeps the system prompt
        """
        if keep_system_prompt and self.conversation_history:
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []

    async def generate_response(
        self,
        user_message: str,
        latency_tracker: Optional[LatencyMetrics] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.

        Args:
            user_message: User's input message
            latency_tracker: Optional latency tracker
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            Response chunks as they arrive
        """
        if not self.client:
            logger.error("Groq client not initialized")
            return

        try:
            # Add user message to history
            self.add_user_message(user_message)

            if latency_tracker:
                latency_tracker.mark("llm_start")

            logger.info(f"🤖 Sending to Groq LLM: '{user_message}'")

            # Create streaming completion
            stream = self.client.chat.completions.create(
                model=model or settings.GROQ_MODEL,
                messages=self.conversation_history,
                temperature=temperature or settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                stream=True,
            )

            full_response = ""
            first_token = True

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content

                    if first_token and latency_tracker:
                        latency_tracker.mark("llm_first_token")
                        llm_latency = latency_tracker.measure(
                            "llm_start", "llm_first_token", "llm_latency"
                        )
                        if llm_latency:
                            logger.info(
                                f"📊 LLM Latency (first token): {llm_latency:.2f}ms")
                        first_token = False

                    full_response += content
                    yield content

            # Add assistant response to history
            self.add_assistant_message(full_response)

            logger.info(f"🤖 LLM response complete: '{full_response[:100]}...'")

        except Exception as e:
            logger.error(
                f"❌ Error generating LLM response: {e}", exc_info=True)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()

    @staticmethod
    def detect_sentence_end(text: str) -> bool:
        """
        Detect if text contains a sentence ending.

        Args:
            text: Text to check

        Returns:
            bool: True if contains sentence ending punctuation
        """
        return any(punct in text for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n'])
