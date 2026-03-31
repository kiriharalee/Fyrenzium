"""Interactive CLI for the modular dubbing pipeline."""

from __future__ import annotations

import getpass
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .models import PipelineSettings, SourceMedia
from .pipeline import build_context, load_runner
from .state import (
    build_manifest,
    create_job_dir,
    ensure_directory,
    list_existing_jobs,
    load_manifest,
    save_manifest,
    stage_lookup,
)


def prompt_text(prompt: str, default: Optional[str] = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("A value is required.")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        value = input(f"{prompt}{suffix}: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_int(prompt: str, default: Optional[int] = None) -> Optional[int]:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter a whole number.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def prompt_csv(prompt: str, default: Optional[List[str]] = None) -> List[str]:
    default = default or []
    default_text = ",".join(default)
    raw = prompt_text(prompt, default_text, required=False)
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def prompt_path(prompt: str, default: Optional[Path] = None, must_exist: bool = False) -> Path:
    while True:
        value = prompt_text(prompt, str(default) if default else None)
        path = Path(value).expanduser()
        if must_exist and not path.exists():
            print(f"Path does not exist: {path}")
            continue
        return path


def prompt_secret(label: str, env_name: str) -> str:
    import os

    existing = os.environ.get(env_name, "").strip()
    if existing:
        return existing
    while True:
        value = getpass.getpass(f"{label} ({env_name}): ").strip()
        if value:
            return value
        print("A value is required.")


def choose_jobs_root() -> Path:
    root = prompt_path("Jobs root directory", Path.cwd() / "jobs")
    return ensure_directory(root)


def choose_new_job_dir() -> Path:
    jobs_root = choose_jobs_root()
    job_name = prompt_text("Job name", "russian-to-english-dub")
    return create_job_dir(job_name, jobs_root)


def choose_existing_job_dir() -> Path:
    jobs_root = choose_jobs_root()
    jobs = list_existing_jobs(jobs_root)
    if not jobs:
        raise SystemExit("No existing jobs were found.")
    print("Available jobs:")
    for index, job in enumerate(jobs, start=1):
        print(f"{index}. {job.name}")
    while True:
        selection = prompt_int("Choose a job number", 1)
        if selection is not None and 1 <= selection <= len(jobs):
            return jobs[selection - 1]
        print("Select a valid job number.")


def collect_source_media() -> SourceMedia:
    print("\nSource media")
    video_path = prompt_path("Source video path", must_exist=True)
    audio_path = None
    if prompt_yes_no("Do you already have a separate source audio file?", default=False):
        audio_path = prompt_path("Source audio path", must_exist=True)
    language = prompt_text("Source spoken language code", "ru")
    title = prompt_text("Project title", video_path.stem)
    return SourceMedia(video_path=video_path, audio_path=audio_path, language=language, title=title)


def collect_pipeline_settings() -> PipelineSettings:
    print("\nPipeline settings")
    return PipelineSettings(
        source_language=prompt_text("Source language code", "ru"),
        target_language=prompt_text("Target language code", "en"),
        source_separation_runner=prompt_text("Source separation runner label", "uvr5"),
        source_separation_command=prompt_text(
            "Source separation command template", "", required=False
        ),
        elevenlabs_api_key_env=prompt_text("ElevenLabs API key environment variable", "ELEVENLABS_API_KEY"),
        elevenlabs_scribe_model=prompt_text("ElevenLabs Scribe model", "scribe_v2"),
        elevenlabs_tts_model=prompt_text("ElevenLabs TTS model", "eleven_multilingual_v2"),
        openrouter_api_key_env=prompt_text("OpenRouter API key environment variable", "OPENROUTER_API_KEY"),
        translation_model=prompt_text(
            "OpenRouter translation model", "qwen/qwen3.6-plus-preview:free"
        ),
        scribe_keyterms=prompt_csv("Scribe keyterms (comma separated)", []),
        translation_glossary=prompt_csv("Translation glossary terms (comma separated)", []),
        estimated_speakers=prompt_int("Estimated speaker count", None),
        syllables_per_second=prompt_float("Target syllables per second", 4.5),
        max_duration_stretch=prompt_float("Maximum allowed stretch ratio", 0.15),
        low_confidence_threshold=prompt_float("Low confidence threshold", 0.75),
        segment_gap_seconds=prompt_float("Segment gap threshold in seconds", 0.8),
        max_segment_seconds=prompt_float("Maximum segment length in seconds", 8.0),
    )


def build_new_job_manifest() -> Path:
    job_dir = choose_new_job_dir()
    source_media = collect_source_media()
    settings = collect_pipeline_settings()
    manifest = build_manifest(job_dir.name, job_dir, source_media, settings)
    save_manifest(manifest)
    print(f"\nCreated manifest at {job_dir / 'manifest.json'}")
    return job_dir


def update_voice_mappings(job_dir: Path) -> None:
    manifest = load_manifest(job_dir)
    existing = dict(manifest.settings.speaker_voice_map)
    approved_path = job_dir / "02_transcript_review" / "transcript_segments_approved.json"
    if approved_path.exists():
        import json

        payload = json.loads(approved_path.read_text(encoding="utf-8"))
        speakers = sorted({str(item.get("speaker") or "") for item in payload if item.get("speaker")})
        for speaker in speakers:
            current = existing.get(speaker, "")
            value = prompt_text(
                f"ElevenLabs voice_id for {speaker}",
                current or None,
                required=False,
            )
            if value:
                existing[speaker] = value
    manifest.settings.speaker_voice_map = existing
    save_manifest(manifest)


def collect_api_keys(settings: PipelineSettings) -> Dict[str, str]:
    return {
        "elevenlabs": prompt_secret("ElevenLabs API key", settings.elevenlabs_api_key_env),
        "openrouter": prompt_secret("OpenRouter API key", settings.openrouter_api_key_env),
    }


def run_job(job_dir: Path) -> None:
    manifest = load_manifest(job_dir)
    print(f"\nLoaded job: {manifest.job_name}")
    if prompt_yes_no("Update speaker to voice mappings before running?", default=False):
        update_voice_mappings(job_dir)
        manifest = load_manifest(job_dir)

    api_keys = collect_api_keys(manifest.settings)
    runner = load_runner(job_dir, api_keys)
    final_manifest = runner.run(resume=True)

    lookup = stage_lookup(final_manifest)
    print("\nStage summary")
    for stage_name, record in lookup.items():
        print(f"- {stage_name.value}: {record.status.value}")
        if record.message:
            print(f"  {record.message}")


def print_help() -> None:
    print(
        "Interactive dubbing pipeline wizard.\n"
        "Run without flags to create or resume a job and execute the next stages."
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print_help()
        return 0

    print("AI Video Dubbing Pipeline Wizard")
    print("---------------------------------")
    if prompt_yes_no("Resume an existing job?", default=False):
        job_dir = choose_existing_job_dir()
    else:
        job_dir = build_new_job_manifest()

    run_job(job_dir)
    return 0
