"""Provider client helpers for the dubbing pipeline."""

from .elevenlabs import (
    DEFAULT_SCRIBE_MODEL_ID,
    DEFAULT_TTS_MODEL_ID,
    ElevenLabsAPIError,
    ElevenLabsClient,
    ElevenLabsError,
    ElevenLabsRateLimitError,
)
from .openrouter import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterAPIError,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterRateLimitError,
)

__all__ = [
    "DEFAULT_OPENROUTER_MODEL",
    "DEFAULT_SCRIBE_MODEL_ID",
    "DEFAULT_TTS_MODEL_ID",
    "ElevenLabsAPIError",
    "ElevenLabsClient",
    "ElevenLabsError",
    "ElevenLabsRateLimitError",
    "OpenRouterAPIError",
    "OpenRouterClient",
    "OpenRouterError",
    "OpenRouterRateLimitError",
]
