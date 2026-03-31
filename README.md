# Fyrenzium

Interactive, modular Python pipeline for Russian-to-English video dubbing.

## What It Does

The pipeline is split into maintainable modules for:

- interactive job setup and resume
- manifest/state persistence
- source separation orchestration
- ElevenLabs transcription and synthesis
- OpenRouter translation with `qwen/qwen3.6-plus-preview:free`
- CSV-based transcript and translation review checkpoints
- timing alignment and final audio/video mix

Lip sync is intentionally excluded.

## Project Layout

- `dubbing_pipeline.py`: root launcher
- `dubbing_pipeline/`: package modules for CLI, pipeline runner, stages, providers, review helpers, and media helpers
- `jobs/`: created at runtime for manifests and stage artifacts

## Usage

Run the wizard:

```bash
python3 dubbing_pipeline.py
```

The script will ask for:

- source video and optional source audio
- source-separation command template
- ElevenLabs and OpenRouter settings
- glossary/keyterms
- speaker-to-voice mappings when you resume after PVC creation

## Prerequisites

- Python 3.9+
- FFmpeg on `PATH`
- Optional: Rubber Band on `PATH` for higher-quality time stretching
- An external UVR or compatible Mel-RoFormer runner available through the command template you provide
- ElevenLabs API key
- OpenRouter API key

## Review Checkpoints

The pipeline pauses at two manual checkpoints:

- `02_transcript_review/transcript_review.csv`
- `03_translation/translation_review.csv`

Approve every row, then rerun the wizard to continue.

## Notes

- The translation model defaults to `qwen/qwen3.6-plus-preview:free`.
- If OpenRouter rate-limits or rejects that model, the translation stage will prompt you to retry, switch models, or stop.
- Voice prep generates per-speaker sample packs and pauses until you supply ElevenLabs `voice_id` mappings.
