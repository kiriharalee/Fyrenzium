"""Core data models for the dubbing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class StageName(str, Enum):
    """Canonical stage identifiers used by the pipeline manifest."""

    SOURCE_SEPARATION = "source_separation"
    TRANSCRIPTION = "transcription"
    TRANSCRIPT_REVIEW = "transcript_review"
    TRANSLATION = "translation"
    TRANSLATION_REVIEW = "translation_review"
    VOICE_PREP = "voice_prep"
    SYNTHESIS = "synthesis"
    ALIGNMENT = "alignment"
    FINAL_MIX = "final_mix"


class StageStatus(str, Enum):
    """Execution state for a pipeline stage."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class SourceMedia:
    """Source assets selected for a job."""

    video_path: Path
    audio_path: Optional[Path] = None
    language: str = "ru"
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        data["audio_path"] = str(self.audio_path) if self.audio_path else None
        return data


@dataclass
class VoiceProfile:
    """Voice identity and supporting samples for a speaker."""

    speaker_label: str
    voice_id: Optional[str] = None
    sample_paths: List[Path] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sample_paths"] = [str(path) for path in self.sample_paths]
        return data


@dataclass
class TranscriptWord:
    """Word-level transcription output."""

    word: str
    start_sec: float
    end_sec: float
    confidence: Optional[float] = None
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """Reviewable transcript segment with timing and diarization metadata."""

    segment_id: str
    speaker: str
    start_sec: float
    end_sec: float
    text: str
    words: List[TranscriptWord] = field(default_factory=list)
    confidence: Optional[float] = None
    needs_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["words"] = [word.to_dict() for word in self.words]
        return data


@dataclass
class TranslationSegment:
    """Target-language segment aligned to a source transcript segment."""

    segment_id: str
    speaker: str
    start_sec: float
    end_sec: float
    source_text: str
    translated_text: str = ""
    target_syllables: Optional[int] = None
    overflow: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineSettings:
    """User-configurable settings captured during the wizard."""

    source_language: str = "ru"
    target_language: str = "en"
    source_separation_runner: str = "uvr5"
    source_separation_command: str = ""
    elevenlabs_api_key_env: str = "ELEVENLABS_API_KEY"
    elevenlabs_scribe_model: str = "scribe_v2"
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    translation_model: str = "qwen/qwen3.6-plus-preview:free"
    scribe_keyterms: List[str] = field(default_factory=list)
    translation_glossary: List[str] = field(default_factory=list)
    speaker_voice_map: Dict[str, str] = field(default_factory=dict)
    estimated_speakers: Optional[int] = None
    syllables_per_second: float = 4.5
    max_duration_stretch: float = 0.15
    low_confidence_threshold: float = 0.75
    segment_gap_seconds: float = 0.8
    max_segment_seconds: float = 8.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageRecord:
    """Manifest entry describing one pipeline stage."""

    name: StageName
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["name"] = self.name.value
        data["status"] = self.status.value
        return data


@dataclass
class JobManifest:
    """Top-level persisted state for a dubbing job."""

    job_name: str
    job_dir: Path
    source_media: SourceMedia
    settings: PipelineSettings
    stage_records: List[StageRecord]
    created_at: str
    updated_at: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "job_dir": str(self.job_dir),
            "source_media": self.source_media.to_dict(),
            "settings": self.settings.to_dict(),
            "stage_records": [record.to_dict() for record in self.stage_records],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }
