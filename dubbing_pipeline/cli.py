"""Interactive CLI for the modular dubbing pipeline."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .models import PipelineSettings, SourceMedia
from .pipeline import load_runner
from .providers import ElevenLabsClient, ElevenLabsError
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
    existing = os.environ.get(env_name, "").strip()
    if existing:
        return existing
    while True:
        value = getpass.getpass(f"{label} ({env_name}): ").strip()
        if value:
            return value
        print("A value is required.")


def build_settings(
    *,
    simple_mode: bool,
    source_language: str,
    target_language: str,
    elevenlabs_api_key_env: str,
    openrouter_api_key_env: str,
    elevenlabs_scribe_model: str,
    elevenlabs_tts_model: str,
    translation_model: str,
    source_separation_command: str = "",
    scribe_keyterms: Optional[Sequence[str]] = None,
    translation_glossary: Optional[Sequence[str]] = None,
    auto_approve_transcript_review: bool = False,
    auto_approve_translation_review: bool = False,
    auto_voice_id: str = "",
    auto_clone_voices: bool = True,
    voice_clone_prefix: str = "fyrenzium",
    voice_clone_min_seconds: float = 60.0,
    voice_clone_target_seconds: float = 120.0,
    voice_clone_max_seconds: float = 300.0,
    voice_clone_remove_background_noise: bool = False,
    allow_alignment_overflow: bool = False,
    estimated_speakers: Optional[int] = None,
    syllables_per_second: float = 4.5,
    max_duration_stretch: float = 0.15,
    low_confidence_threshold: float = 0.75,
    segment_gap_seconds: float = 0.8,
    max_segment_seconds: float = 8.0,
) -> PipelineSettings:
    return PipelineSettings(
        simple_mode=simple_mode,
        source_language=source_language,
        target_language=target_language,
        source_separation_runner="auto",
        source_separation_command=source_separation_command,
        elevenlabs_api_key_env=elevenlabs_api_key_env,
        elevenlabs_scribe_model=elevenlabs_scribe_model,
        elevenlabs_tts_model=elevenlabs_tts_model,
        openrouter_api_key_env=openrouter_api_key_env,
        translation_model=translation_model,
        scribe_keyterms=list(scribe_keyterms or []),
        translation_glossary=list(translation_glossary or []),
        auto_approve_transcript_review=auto_approve_transcript_review,
        auto_approve_translation_review=auto_approve_translation_review,
        auto_voice_id=auto_voice_id,
        auto_clone_voices=auto_clone_voices,
        voice_clone_prefix=voice_clone_prefix,
        voice_clone_min_seconds=voice_clone_min_seconds,
        voice_clone_target_seconds=voice_clone_target_seconds,
        voice_clone_max_seconds=voice_clone_max_seconds,
        voice_clone_remove_background_noise=voice_clone_remove_background_noise,
        allow_alignment_overflow=allow_alignment_overflow,
        estimated_speakers=estimated_speakers,
        syllables_per_second=syllables_per_second,
        max_duration_stretch=max_duration_stretch,
        low_confidence_threshold=low_confidence_threshold,
        segment_gap_seconds=segment_gap_seconds,
        max_segment_seconds=max_segment_seconds,
    )


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


def collect_source_media(source_language: str) -> SourceMedia:
    print("\nSource media")
    video_path = prompt_path("Source video path", must_exist=True)
    audio_path = None
    if prompt_yes_no("Do you already have a separate source audio file?", default=False):
        audio_path = prompt_path("Source audio path", must_exist=True)
    title = prompt_text("Project title", video_path.stem)
    return SourceMedia(video_path=video_path, audio_path=audio_path, language=source_language, title=title)


def collect_pipeline_settings() -> PipelineSettings:
    print("\nPipeline settings")
    return build_settings(
        simple_mode=False,
        source_language=prompt_text("Source language code", "ru"),
        target_language=prompt_text("Target language code", "en"),
        elevenlabs_api_key_env=prompt_text("ElevenLabs API key environment variable", "ELEVENLABS_API_KEY"),
        openrouter_api_key_env=prompt_text("OpenRouter API key environment variable", "OPENROUTER_API_KEY"),
        elevenlabs_scribe_model=prompt_text("ElevenLabs Scribe model", "scribe_v2"),
        elevenlabs_tts_model=prompt_text("ElevenLabs TTS model", "eleven_multilingual_v2"),
        translation_model=prompt_text("OpenRouter translation model", "minimax/minimax-m2.5:free"),
        scribe_keyterms=prompt_csv("Scribe keyterms (comma separated)", []),
        translation_glossary=prompt_csv("Translation glossary terms (comma separated)", []),
        estimated_speakers=prompt_int("Estimated speaker count", None),
        syllables_per_second=prompt_float("Target syllables per second", 4.5),
        max_duration_stretch=prompt_float("Maximum allowed stretch ratio", 0.15),
        low_confidence_threshold=prompt_float("Low confidence threshold", 0.75),
        segment_gap_seconds=prompt_float("Segment gap threshold in seconds", 0.8),
        max_segment_seconds=prompt_float("Maximum segment length in seconds", 8.0),
    )


def collect_simple_pipeline_settings() -> PipelineSettings:
    print("\nSimple test mode")
    print("Basic functionality check with one suggested speaker and fewer settings.")
    return build_settings(
        simple_mode=True,
        source_language=prompt_text("Source language code", "ru"),
        target_language=prompt_text("Target language code", "en"),
        elevenlabs_api_key_env=prompt_text("ElevenLabs API key environment variable", "ELEVENLABS_API_KEY"),
        openrouter_api_key_env=prompt_text("OpenRouter API key environment variable", "OPENROUTER_API_KEY"),
        elevenlabs_scribe_model="scribe_v2",
        elevenlabs_tts_model="eleven_multilingual_v2",
        translation_model="minimax/minimax-m2.5:free",
        estimated_speakers=1,
        syllables_per_second=4.5,
        max_duration_stretch=0.15,
        low_confidence_threshold=0.75,
        segment_gap_seconds=0.8,
        max_segment_seconds=8.0,
    )


def build_new_job_manifest() -> Path:
    job_dir = choose_new_job_dir()
    use_simple_mode = prompt_yes_no("Use simple test mode?", default=False)
    mode_label = "Simple Test Mode" if use_simple_mode else "Full Mode"
    print(f"\nSelected mode: {mode_label}")
    settings = collect_simple_pipeline_settings() if use_simple_mode else collect_pipeline_settings()
    source_media = collect_source_media(settings.source_language)
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


def collect_api_keys(settings: PipelineSettings, *, non_interactive: bool = False) -> Dict[str, str]:
    api_keys: Dict[str, str] = {}
    required = {
        "elevenlabs": ("ElevenLabs API key", settings.elevenlabs_api_key_env),
        "openrouter": ("OpenRouter API key", settings.openrouter_api_key_env),
    }
    for key, (label, env_name) in required.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            api_keys[key] = value
            continue
        if non_interactive:
            raise SystemExit(f"Missing {label} in environment variable {env_name}.")
        api_keys[key] = prompt_secret(label, env_name)
    return api_keys


def maybe_assign_auto_voice(job_dir: Path, api_keys: Dict[str, str], *, auto_select_voice: bool) -> None:
    manifest = load_manifest(job_dir)
    if not auto_select_voice:
        return
    if manifest.settings.auto_voice_id.strip():
        return
    if manifest.settings.speaker_voice_map:
        return
    try:
        payload = ElevenLabsClient(api_keys["elevenlabs"]).list_voices()
    except ElevenLabsError as exc:
        raise SystemExit(f"Could not auto-select an ElevenLabs voice: {exc}") from exc
    voices = payload.get("voices") or []
    for voice in voices:
        voice_id = str(voice.get("voice_id") or "").strip()
        if not voice_id:
            continue
        manifest.settings.auto_voice_id = voice_id
        save_manifest(manifest)
        voice_name = str(voice.get("name") or voice_id)
        print(f"Automatically selected ElevenLabs voice: {voice_name} ({voice_id})")
        return
    raise SystemExit("No ElevenLabs voices were available for automatic selection.")


def print_stage_summary(final_manifest) -> None:
    lookup = stage_lookup(final_manifest)
    print("\nStage summary")
    for stage_name, record in lookup.items():
        print(f"- {stage_name.value}: {record.status.value}")
        if record.message:
            print(f"  {record.message}")


def run_job(
    job_dir: Path,
    *,
    non_interactive: bool = False,
    prompt_for_voice_mappings: bool = True,
    auto_select_voice: bool = False,
):
    manifest = load_manifest(job_dir)
    print(f"\nLoaded job: {manifest.job_name}")
    if prompt_for_voice_mappings and not non_interactive and prompt_yes_no(
        "Update speaker to voice mappings before running?", default=False
    ):
        update_voice_mappings(job_dir)
        manifest = load_manifest(job_dir)

    api_keys = collect_api_keys(manifest.settings, non_interactive=non_interactive)
    maybe_assign_auto_voice(job_dir, api_keys, auto_select_voice=auto_select_voice)
    runner = load_runner(job_dir, api_keys)
    final_manifest = runner.run(resume=True)
    print_stage_summary(final_manifest)
    return final_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive dubbing pipeline wizard. Run without arguments to use the wizard, "
            "or pass arguments for a single non-interactive run."
        )
    )
    parser.add_argument("--resume-job", help="Resume an existing job directory non-interactively.")
    parser.add_argument("--jobs-root", help="Jobs root directory for new jobs.")
    parser.add_argument("--job-name", default="russian-to-english-dub", help="Job name for a new run.")
    parser.add_argument("--input-video", help="Source video path for a new run.")
    parser.add_argument("--input-audio", help="Optional separate source audio path for a new run.")
    parser.add_argument("--project-title", help="Optional project title stored in the manifest.")
    parser.add_argument("--source-language", default="ru", help="Source language code.")
    parser.add_argument("--target-language", default="en", help="Target language code.")
    parser.add_argument("--simple-mode", action="store_true", help="Use simple mode defaults.")
    parser.add_argument(
        "--one-run",
        action="store_true",
        help=(
            "Enable a single-run path: auto-approve review checkpoints, reuse one voice for all speakers, "
            "and allow alignment overflow stretching so the job can finish in one execution."
        ),
    )
    parser.add_argument("--voice-id", help="ElevenLabs voice_id to use for every speaker in one-run mode.")
    parser.add_argument(
        "--disable-auto-clone-voices",
        action="store_true",
        help="Skip automatic per-speaker voice cloning and require manual voice IDs or auto-selected voices.",
    )
    parser.add_argument(
        "--voice-clone-prefix",
        default="fyrenzium",
        help="Prefix used when naming automatically created ElevenLabs voice clones.",
    )
    parser.add_argument(
        "--voice-clone-min-seconds",
        type=float,
        default=60.0,
        help="Minimum combined clean sample duration per speaker for automatic cloning.",
    )
    parser.add_argument(
        "--voice-clone-target-seconds",
        type=float,
        default=120.0,
        help="Target combined clean sample duration per speaker for automatic cloning.",
    )
    parser.add_argument(
        "--voice-clone-max-seconds",
        type=float,
        default=300.0,
        help="Maximum combined clean sample duration per speaker for automatic cloning.",
    )
    parser.add_argument(
        "--voice-clone-remove-background-noise",
        action="store_true",
        help="Enable ElevenLabs background-noise removal when creating automatic voice clones.",
    )
    parser.add_argument(
        "--auto-select-voice",
        action="store_true",
        help="Automatically choose the first available ElevenLabs voice when needed.",
    )
    parser.add_argument(
        "--elevenlabs-api-key-env",
        default="ELEVENLABS_API_KEY",
        help="Environment variable name for the ElevenLabs API key.",
    )
    parser.add_argument(
        "--openrouter-api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable name for the OpenRouter API key.",
    )
    parser.add_argument(
        "--translation-model",
        default="minimax/minimax-m2.5:free",
        help="OpenRouter model used for translation.",
    )
    parser.add_argument(
        "--source-separation-command",
        default="",
        help="Optional custom source separation command template.",
    )
    parser.add_argument("--estimated-speakers", type=int, help="Estimated number of speakers in full mode.")
    parser.add_argument("--syllables-per-second", type=float, default=4.5, help="Target syllables per second.")
    parser.add_argument(
        "--max-duration-stretch",
        type=float,
        help="Maximum allowed stretch ratio. One-run mode defaults to 1.0 if omitted.",
    )
    parser.add_argument(
        "--allow-alignment-overflow",
        action="store_true",
        help="Stretch overflowing lines instead of stopping for manual review.",
    )
    return parser


def build_settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    simple_mode = bool(args.simple_mode or args.one_run)
    if simple_mode:
        elevenlabs_scribe_model = "scribe_v2"
        elevenlabs_tts_model = "eleven_multilingual_v2"
        estimated_speakers = 1
        low_confidence_threshold = 0.75
        segment_gap_seconds = 0.8
        max_segment_seconds = 8.0
    else:
        elevenlabs_scribe_model = "scribe_v2"
        elevenlabs_tts_model = "eleven_multilingual_v2"
        estimated_speakers = args.estimated_speakers
        low_confidence_threshold = 0.75
        segment_gap_seconds = 0.8
        max_segment_seconds = 8.0

    max_duration_stretch = args.max_duration_stretch
    if max_duration_stretch is None:
        max_duration_stretch = 1.0 if args.one_run else 0.15

    return build_settings(
        simple_mode=simple_mode,
        source_language=args.source_language,
        target_language=args.target_language,
        elevenlabs_api_key_env=args.elevenlabs_api_key_env,
        openrouter_api_key_env=args.openrouter_api_key_env,
        elevenlabs_scribe_model=elevenlabs_scribe_model,
        elevenlabs_tts_model=elevenlabs_tts_model,
        translation_model=args.translation_model,
        source_separation_command=args.source_separation_command,
        auto_approve_transcript_review=args.one_run,
        auto_approve_translation_review=args.one_run,
        auto_voice_id=args.voice_id or "",
        auto_clone_voices=not args.disable_auto_clone_voices,
        voice_clone_prefix=args.voice_clone_prefix,
        voice_clone_min_seconds=args.voice_clone_min_seconds,
        voice_clone_target_seconds=args.voice_clone_target_seconds,
        voice_clone_max_seconds=args.voice_clone_max_seconds,
        voice_clone_remove_background_noise=bool(args.voice_clone_remove_background_noise),
        allow_alignment_overflow=bool(args.allow_alignment_overflow or args.one_run),
        estimated_speakers=estimated_speakers,
        syllables_per_second=args.syllables_per_second,
        max_duration_stretch=max_duration_stretch,
        low_confidence_threshold=low_confidence_threshold,
        segment_gap_seconds=segment_gap_seconds,
        max_segment_seconds=max_segment_seconds,
    )


def apply_noninteractive_overrides(job_dir: Path, args: argparse.Namespace) -> None:
    manifest = load_manifest(job_dir)
    changed = False
    if args.one_run:
        if not manifest.settings.auto_approve_transcript_review:
            manifest.settings.auto_approve_transcript_review = True
            changed = True
        if not manifest.settings.auto_approve_translation_review:
            manifest.settings.auto_approve_translation_review = True
            changed = True
        if not manifest.settings.allow_alignment_overflow:
            manifest.settings.allow_alignment_overflow = True
            changed = True
        if manifest.settings.max_duration_stretch < 1.0 and args.max_duration_stretch is None:
            manifest.settings.max_duration_stretch = 1.0
            changed = True
    if args.voice_id and manifest.settings.auto_voice_id != args.voice_id:
        manifest.settings.auto_voice_id = args.voice_id
        changed = True
    if args.disable_auto_clone_voices and manifest.settings.auto_clone_voices:
        manifest.settings.auto_clone_voices = False
        changed = True
    if args.voice_clone_prefix and manifest.settings.voice_clone_prefix != args.voice_clone_prefix:
        manifest.settings.voice_clone_prefix = args.voice_clone_prefix
        changed = True
    if manifest.settings.voice_clone_min_seconds != args.voice_clone_min_seconds:
        manifest.settings.voice_clone_min_seconds = args.voice_clone_min_seconds
        changed = True
    if manifest.settings.voice_clone_target_seconds != args.voice_clone_target_seconds:
        manifest.settings.voice_clone_target_seconds = args.voice_clone_target_seconds
        changed = True
    if manifest.settings.voice_clone_max_seconds != args.voice_clone_max_seconds:
        manifest.settings.voice_clone_max_seconds = args.voice_clone_max_seconds
        changed = True
    if args.voice_clone_remove_background_noise and not manifest.settings.voice_clone_remove_background_noise:
        manifest.settings.voice_clone_remove_background_noise = True
        changed = True
    if args.translation_model and manifest.settings.translation_model != args.translation_model:
        manifest.settings.translation_model = args.translation_model
        changed = True
    if args.max_duration_stretch is not None and manifest.settings.max_duration_stretch != args.max_duration_stretch:
        manifest.settings.max_duration_stretch = args.max_duration_stretch
        changed = True
    if args.allow_alignment_overflow and not manifest.settings.allow_alignment_overflow:
        manifest.settings.allow_alignment_overflow = True
        changed = True
    if changed:
        save_manifest(manifest)


def create_job_from_args(args: argparse.Namespace) -> Path:
    if not args.input_video:
        raise SystemExit("--input-video is required for a new non-interactive run.")
    video_path = Path(args.input_video).expanduser()
    if not video_path.exists():
        raise SystemExit(f"Source video does not exist: {video_path}")
    audio_path = Path(args.input_audio).expanduser() if args.input_audio else None
    if audio_path is not None and not audio_path.exists():
        raise SystemExit(f"Source audio does not exist: {audio_path}")

    jobs_root = Path(args.jobs_root).expanduser() if args.jobs_root else None
    job_dir = create_job_dir(args.job_name, jobs_root)
    settings = build_settings_from_args(args)
    source_media = SourceMedia(
        video_path=video_path,
        audio_path=audio_path,
        language=settings.source_language,
        title=args.project_title or video_path.stem,
    )
    manifest = build_manifest(job_dir.name, job_dir, source_media, settings)
    save_manifest(manifest)
    print(f"Created manifest at {job_dir / 'manifest.json'}")
    return job_dir


def run_non_interactive(args: argparse.Namespace) -> int:
    if args.resume_job:
        job_dir = Path(args.resume_job).expanduser()
        if not (job_dir / "manifest.json").exists():
            raise SystemExit(f"Could not find manifest.json in {job_dir}")
        apply_noninteractive_overrides(job_dir, args)
    else:
        job_dir = create_job_from_args(args)
    run_job(
        job_dir,
        non_interactive=True,
        prompt_for_voice_mappings=False,
        auto_select_voice=bool(args.auto_select_voice or args.one_run),
    )
    return 0


def run_interactive() -> int:
    print("AI Video Dubbing Pipeline Wizard")
    print("---------------------------------")
    if prompt_yes_no("Resume an existing job?", default=False):
        job_dir = choose_existing_job_dir()
    else:
        job_dir = build_new_job_manifest()

    run_job(job_dir)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return run_interactive()
    parser = build_parser()
    return run_non_interactive(parser.parse_args(args))
