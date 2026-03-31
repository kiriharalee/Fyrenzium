# Fyrenzium

**Interactive, modular Python pipeline for Russian-to-English video dubbing.**

No external Python dependencies — just stdlib, your API keys, and FFmpeg.

---

## Pipeline at a Glance

```
Source Video
  │
  ├─ 1. Source Separation ──── Demucs isolates vocals from instrumentals
  ├─ 2. Transcription ──────── ElevenLabs Scribe v2 with speaker diarization
  ├─ 3. Transcript Review ──── ✏️  CSV checkpoint — correct speakers & text
  ├─ 4. Translation ─────────── OpenRouter (Qwen) with glossary & timing hints
  ├─ 5. Translation Review ─── ✏️  CSV checkpoint — verify syllable fit & meaning
  ├─ 6. Voice Prep ──────────── Speaker sample packs + ElevenLabs voice mapping
  ├─ 7. Synthesis ──────────── ElevenLabs TTS per segment
  ├─ 8. Alignment ──────────── Time-stretch to match original timing
  └─ 9. Final Mix ──────────── Combine English voice + instrumental → output.mp4
```

> Lip sync is intentionally excluded.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.9+** | No pip packages needed |
| **FFmpeg** | Must be on `PATH` |
| **Demucs** (or other backend) | Auto-detected for source separation |
| **Rubber Band** *(optional)* | Higher-quality time-stretching; falls back to FFmpeg |
| **ElevenLabs API key** | Transcription + synthesis |
| **OpenRouter API key** | Translation (defaults to `qwen/qwen3.6-plus-preview:free`) |

---

## Quick Start

```bash
# Set your API keys
export ELEVENLABS_API_KEY="..."
export OPENROUTER_API_KEY="..."

# Launch the interactive wizard
python3 dubbing_pipeline.py
```

The wizard walks you through:

1. **New job** — provide a source video, configure languages, glossary, and timing parameters
2. **Stage execution** — the pipeline runs automatically, pausing at review checkpoints
3. **Resume** — rerun the wizard at any time to pick up where you left off

---

## Review Checkpoints

The pipeline pauses at two points for manual quality control:

### Transcript Review
**`{job}/02_transcript_review/transcript_review.csv`**

Review and correct the Russian transcription and speaker labels. Mark every row `approved=true`, then rerun the wizard.

### Translation Review
**`{job}/03_translation/translation_review.csv`**

Check English translations for accuracy, syllable count, and timing fit. Fix any overflow flagged rows, approve all, and resume.

---

## Project Layout

```
dubbing_pipeline.py              ← entry point
dubbing_pipeline/
├── cli.py                       ← interactive wizard & prompts
├── pipeline.py                  ← pipeline runner & context
├── stages.py                    ← all 9 stage implementations
├── models.py                    ← data classes & enums
├── state.py                     ← manifest persistence
├── media.py                     ← FFmpeg / Rubber Band wrappers
├── review.py                    ← CSV checkpoint helpers
└── providers/
    ├── elevenlabs.py            ← transcription & synthesis client
    └── openrouter.py            ← translation client
tests/
├── test_stages.py
├── test_state.py
└── test_review.py
jobs/                            ← created at runtime
└── {job_name}/
    ├── manifest.json
    ├── 01_source/
    ├── 02_transcript_review/
    ├── 03_translation/
    ├── 04_voice_prep/
    ├── 04_synthesis/
    ├── 05_alignment/
    └── 06_mix/
        └── output.mp4           ← final result
```

---

## How It Works

### Job Persistence

Every job is backed by a `manifest.json` that tracks source media, settings, stage status, timestamps, and artifact hashes. You can stop and resume at any point — the pipeline validates previous outputs (including SHA1 checksums) before reusing them.

### Voice Mapping

After the voice prep stage, rerun the wizard and choose to update speaker-to-voice mappings. For each detected speaker, provide their ElevenLabs `voice_id`. These are saved in the manifest and used during synthesis.

### Translation Model

The default translation model is `qwen/qwen3.6-plus-preview:free` via OpenRouter. If the model is rate-limited or unavailable, the pipeline will prompt you to retry, switch models, or stop.

### Source Separation

The pipeline auto-detects supported CLI backends (e.g. `demucs`). You can also provide a custom separation command in the manifest for advanced setups.

---

## Notes

- Zero external Python dependencies — all HTTP clients use `urllib` from stdlib
- Multipart uploads, JSON parsing, and form data are handled inline
- Configurable timing parameters: syllables/sec target, max stretch ratio, segment gap threshold, and more
- Comprehensive error handling with clear messages and recovery hints
