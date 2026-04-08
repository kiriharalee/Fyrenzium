"""Manifest and job-state helpers for the dubbing pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    JobManifest,
    PipelineSettings,
    SourceMedia,
    StageName,
    StageRecord,
    StageStatus,
)

MANIFEST_FILENAME = "manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "job"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(job_dir: Path) -> Path:
    return job_dir / MANIFEST_FILENAME


def default_stage_records() -> List[StageRecord]:
    return [StageRecord(name=stage) for stage in StageName]


def build_manifest(
    job_name: str,
    job_dir: Path,
    source_media: SourceMedia,
    settings: PipelineSettings,
    notes: str = "",
) -> JobManifest:
    timestamp = utc_now_iso()
    return JobManifest(
        job_name=job_name,
        job_dir=job_dir,
        source_media=source_media,
        settings=settings,
        stage_records=default_stage_records(),
        created_at=timestamp,
        updated_at=timestamp,
        notes=notes,
    )


def stage_lookup(manifest: JobManifest) -> Dict[StageName, StageRecord]:
    return {record.name: record for record in manifest.stage_records}


def update_stage_status(
    manifest: JobManifest,
    stage_name: StageName,
    status: StageStatus,
    message: str = "",
    artifacts: Optional[Dict[str, str]] = None,
) -> JobManifest:
    timestamp = utc_now_iso()
    updated_records: List[StageRecord] = []
    for record in manifest.stage_records:
        if record.name == stage_name:
            started_at = record.started_at
            completed_at = record.completed_at
            if status == StageStatus.IN_PROGRESS and not started_at:
                started_at = timestamp
            if status in {
                StageStatus.COMPLETED,
                StageStatus.NEEDS_REVIEW,
                StageStatus.FAILED,
                StageStatus.SKIPPED,
            }:
                completed_at = timestamp
            updated_records.append(
                replace(
                    record,
                    status=status,
                    started_at=started_at,
                    completed_at=completed_at,
                    message=message or record.message,
                    artifacts=artifacts or record.artifacts,
                )
            )
        else:
            updated_records.append(record)
    return replace(manifest, stage_records=updated_records, updated_at=timestamp)


def to_json_dict(manifest: JobManifest) -> Dict[str, Any]:
    return manifest.to_dict()


def from_json_dict(data: Dict[str, Any]) -> JobManifest:
    source_media_data = data["source_media"]
    settings_data = data["settings"]
    stage_records_data = data.get("stage_records", [])

    source_media = SourceMedia(
        video_path=Path(source_media_data["video_path"]),
        audio_path=Path(source_media_data["audio_path"])
        if source_media_data.get("audio_path")
        else None,
        language=source_media_data.get("language", "ru"),
        title=source_media_data.get("title"),
    )
    settings = PipelineSettings(
        simple_mode=bool(settings_data.get("simple_mode", False)),
        source_language=settings_data.get("source_language", "ru"),
        target_language=settings_data.get("target_language", "en"),
        source_separation_runner=settings_data.get("source_separation_runner", "auto"),
        source_separation_command=settings_data.get("source_separation_command", ""),
        elevenlabs_api_key_env=settings_data.get("elevenlabs_api_key_env", "ELEVENLABS_API_KEY"),
        elevenlabs_scribe_model=settings_data.get("elevenlabs_scribe_model", "scribe_v2"),
        elevenlabs_tts_model=settings_data.get("elevenlabs_tts_model", "eleven_multilingual_v2"),
        openrouter_api_key_env=settings_data.get("openrouter_api_key_env", "OPENROUTER_API_KEY"),
        translation_model=settings_data.get(
            "translation_model", "minimax/minimax-m2.5:free"
        ),
        scribe_keyterms=list(settings_data.get("scribe_keyterms", [])),
        translation_glossary=list(settings_data.get("translation_glossary", [])),
        speaker_voice_map=dict(settings_data.get("speaker_voice_map", {})),
        estimated_speakers=settings_data.get("estimated_speakers"),
        syllables_per_second=float(settings_data.get("syllables_per_second", 4.5)),
        max_duration_stretch=float(settings_data.get("max_duration_stretch", 0.15)),
        low_confidence_threshold=float(settings_data.get("low_confidence_threshold", 0.75)),
        segment_gap_seconds=float(settings_data.get("segment_gap_seconds", 0.8)),
        max_segment_seconds=float(settings_data.get("max_segment_seconds", 8.0)),
    )
    stage_records: List[StageRecord] = []
    for record_data in stage_records_data:
        stage_records.append(
            StageRecord(
                name=StageName(record_data["name"]),
                status=StageStatus(record_data.get("status", StageStatus.PENDING.value)),
                started_at=record_data.get("started_at"),
                completed_at=record_data.get("completed_at"),
                message=record_data.get("message", ""),
                artifacts=dict(record_data.get("artifacts", {})),
            )
        )
    if not stage_records:
        stage_records = default_stage_records()
    return JobManifest(
        job_name=data["job_name"],
        job_dir=Path(data["job_dir"]),
        source_media=source_media,
        settings=settings,
        stage_records=stage_records,
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        notes=data.get("notes", ""),
    )


def save_manifest(manifest: JobManifest) -> Path:
    ensure_directory(manifest.job_dir)
    target = manifest_path(manifest.job_dir)
    payload = json.dumps(to_json_dict(manifest), indent=2, sort_keys=True)
    temp_target = target.with_suffix(".json.tmp")
    temp_target.write_text(payload + "\n", encoding="utf-8")
    temp_target.replace(target)
    return target


def load_manifest(job_dir: Path) -> JobManifest:
    payload = json.loads(manifest_path(job_dir).read_text(encoding="utf-8"))
    return from_json_dict(payload)


def jobs_root(default: Optional[Path] = None) -> Path:
    return ensure_directory(default or (Path.cwd() / "jobs"))


def create_job_dir(job_name: str, root: Optional[Path] = None) -> Path:
    root_dir = jobs_root(root)
    job_dir = root_dir / slugify(job_name)
    return ensure_directory(job_dir)


def list_existing_jobs(root: Optional[Path] = None) -> List[Path]:
    root_dir = jobs_root(root)
    return sorted(
        [path for path in root_dir.iterdir() if path.is_dir() and manifest_path(path).exists()]
    )
