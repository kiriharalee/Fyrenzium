"""CSV helpers for manual transcript and translation review.

These helpers keep the review checkpoints boring and predictable: dataclass
records in, CSV files out, and the inverse when resuming a job.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar, Union

PathLike = Union[str, Path]

TRANSCRIPT_REVIEW_FIELDS = [
    "segment_id",
    "speaker",
    "start_sec",
    "end_sec",
    "text_ru_raw",
    "text_ru_final",
    "speaker_final",
    "approved",
    "notes",
]

TRANSLATION_REVIEW_FIELDS = [
    "segment_id",
    "speaker",
    "start_sec",
    "end_sec",
    "text_ru_final",
    "text_en_draft",
    "text_en_final",
    "duration_sec",
    "syllables_per_sec",
    "overflow",
    "approved",
    "notes",
]

T = TypeVar("T")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


@dataclass
class TranscriptReviewRow:
    segment_id: str
    speaker: str
    start_sec: float
    end_sec: float
    text_ru_raw: str = ""
    text_ru_final: str = ""
    speaker_final: str = ""
    approved: bool = False
    notes: str = ""

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "start_sec": f"{self.start_sec:.6f}",
            "end_sec": f"{self.end_sec:.6f}",
            "text_ru_raw": self.text_ru_raw,
            "text_ru_final": self.text_ru_final,
            "speaker_final": self.speaker_final,
            "approved": "true" if self.approved else "false",
            "notes": self.notes,
        }

    @classmethod
    def from_csv_row(cls, row: Mapping[str, Any]) -> "TranscriptReviewRow":
        return cls(
            segment_id=_normalize_text(row.get("segment_id")),
            speaker=_normalize_text(row.get("speaker")),
            start_sec=_to_float(row.get("start_sec")),
            end_sec=_to_float(row.get("end_sec")),
            text_ru_raw=_normalize_text(row.get("text_ru_raw")),
            text_ru_final=_normalize_text(row.get("text_ru_final")),
            speaker_final=_normalize_text(row.get("speaker_final")),
            approved=_to_bool(row.get("approved")),
            notes=_normalize_text(row.get("notes")),
        )


@dataclass
class TranslationReviewRow:
    segment_id: str
    speaker: str
    start_sec: float
    end_sec: float
    text_ru_final: str = ""
    text_en_draft: str = ""
    text_en_final: str = ""
    duration_sec: float = 0.0
    syllables_per_sec: float = 0.0
    overflow: bool = False
    approved: bool = False
    notes: str = ""

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "start_sec": f"{self.start_sec:.6f}",
            "end_sec": f"{self.end_sec:.6f}",
            "text_ru_final": self.text_ru_final,
            "text_en_draft": self.text_en_draft,
            "text_en_final": self.text_en_final,
            "duration_sec": f"{self.duration_sec:.6f}",
            "syllables_per_sec": f"{self.syllables_per_sec:.6f}",
            "overflow": "true" if self.overflow else "false",
            "approved": "true" if self.approved else "false",
            "notes": self.notes,
        }

    @classmethod
    def from_csv_row(cls, row: Mapping[str, Any]) -> "TranslationReviewRow":
        return cls(
            segment_id=_normalize_text(row.get("segment_id")),
            speaker=_normalize_text(row.get("speaker")),
            start_sec=_to_float(row.get("start_sec")),
            end_sec=_to_float(row.get("end_sec")),
            text_ru_final=_normalize_text(row.get("text_ru_final")),
            text_en_draft=_normalize_text(row.get("text_en_draft")),
            text_en_final=_normalize_text(row.get("text_en_final")),
            duration_sec=_to_float(row.get("duration_sec")),
            syllables_per_sec=_to_float(row.get("syllables_per_sec")),
            overflow=_to_bool(row.get("overflow")),
            approved=_to_bool(row.get("approved")),
            notes=_normalize_text(row.get("notes")),
        )


def _write_rows(path: PathLike, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    output_path = Path(path)
    _ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" for key in fieldnames} | dict(row))
    return output_path


def _read_rows(path: PathLike) -> List[Dict[str, str]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_transcript_review_csv(rows: Iterable[TranscriptReviewRow], path: PathLike) -> Path:
    return _write_rows(path, TRANSCRIPT_REVIEW_FIELDS, (row.to_csv_row() for row in rows))


def read_transcript_review_csv(path: PathLike) -> List[TranscriptReviewRow]:
    return [TranscriptReviewRow.from_csv_row(row) for row in _read_rows(path)]


def write_translation_review_csv(rows: Iterable[TranslationReviewRow], path: PathLike) -> Path:
    return _write_rows(path, TRANSLATION_REVIEW_FIELDS, (row.to_csv_row() for row in rows))


def read_translation_review_csv(path: PathLike) -> List[TranslationReviewRow]:
    return [TranslationReviewRow.from_csv_row(row) for row in _read_rows(path)]


def rows_to_dicts(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert dataclass rows or dictionaries into plain dictionaries."""

    converted: List[Dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "to_csv_row"):
            converted.append(dict(row.to_csv_row()))
        elif hasattr(row, "__dataclass_fields__"):
            converted.append(asdict(row))
        elif isinstance(row, Mapping):
            converted.append(dict(row))
        else:
            raise TypeError(f"unsupported row type: {type(row)!r}")
    return converted

