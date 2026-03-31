"""Modular dubbing pipeline package."""

from .models import (
    JobManifest,
    PipelineSettings,
    SourceMedia,
    StageName,
    StageRecord,
    StageStatus,
    TranscriptSegment,
    TranscriptWord,
    TranslationSegment,
    VoiceProfile,
)
from .state import load_manifest, save_manifest

__all__ = [
    "JobManifest",
    "PipelineSettings",
    "SourceMedia",
    "StageName",
    "StageRecord",
    "StageStatus",
    "TranscriptSegment",
    "TranscriptWord",
    "TranslationSegment",
    "VoiceProfile",
    "load_manifest",
    "save_manifest",
]

__version__ = "0.1.0"
