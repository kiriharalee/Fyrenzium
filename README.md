# Fyrenzium

**Interactive, resumable Python CLI pipeline for Russian-to-English video dubbing.**

The repository’s own code uses Python stdlib only, while media-heavy stages rely on external tools and APIs such as FFmpeg, Demucs, Rubber Band, ElevenLabs, and OpenRouter.

---

## Pipeline at a Glance

```text
Source Video
  │
  ├─ 1. Source Separation ──── Demucs auto-detect or custom separation command
  ├─ 2. Transcription ──────── ElevenLabs Scribe v2 with optional diarization
  ├─ 3. Transcript Review ──── CSV checkpoint — correct text and speaker labels
  ├─ 4. Translation ────────── OpenRouter translation model with timing hints
  ├─ 5. Translation Review ─── CSV checkpoint — verify meaning and timing fit
  ├─ 6. Voice Prep ─────────── Automatic per-speaker sample aggregation + ElevenLabs instant voice cloning
  ├─ 7. Synthesis ──────────── ElevenLabs TTS per approved segment
  ├─ 8. Alignment ──────────── Time-stretch to match original timing windows
  └─ 9. Final Mix ──────────── English voice + instrumental → output.mp4
```

> Lip sync is intentionally excluded.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.14+** | Core repo code uses stdlib only |
| **FFmpeg** | Must be on `PATH` |
| **Demucs** *(optional)* | Auto-detected for source separation if installed |
| **Custom separation backend** *(optional)* | Supported via `source_separation_command` if you do not use Demucs |
| **Rubber Band** *(optional)* | Higher-quality time-stretching; otherwise FFmpeg `atempo` is used |
| **ElevenLabs API key** | Required for transcription, automatic voice cloning, and synthesis |
| **OpenRouter API key** | Required for translation; default model is `minimax/minimax-m2.5:free` |
| **Source video** | Always required because the final stage remuxes audio back into the original video |

---

## Quick Start

```bash
export ELEVENLABS_API_KEY="..."
export OPENROUTER_API_KEY="..."
python3.14 dubbing_pipeline.py
```

The wizard walks you through:

1. Choosing a jobs root and creating or resuming a job
2. Optionally enabling **simple test mode**
3. Providing a source video and, if needed, a separate source-audio file
4. Running stages until the pipeline reaches a manual stop point
5. Rerunning the wizard later to resume from the next pending stage

---

## Manual Intervention Points

The pipeline is resumable, but several stages intentionally stop for review or operator input.

### 1. Transcript Review
**`{job}/02_transcript_review/transcript_review.csv`**

Review the Russian transcript, fix text and speaker labels as needed, and mark every row `approved=true`. When all rows are approved, rerun the wizard to continue.

### 2. Translation Review
**`{job}/03_translation/translation_review.csv`**

Review the English draft for accuracy, phrasing, and timing fit. Rows with `overflow=true` need to be shortened or rephrased before approval. When all rows are approved, rerun the wizard to continue.

### 3. Voice Prep and Automatic Voice Cloning
**`{job}/04_voice_prep/speaker_samples/`**
**`{job}/04_voice_prep/voice_prep_manifest.json`**

The pipeline now automatically prepares per-speaker clone audio from the separated vocals stem. For each detected speaker it:

1. Selects diarized transcript segments that appear to contain a persistent single speaker
2. Trims those ranges from the vocals stem
3. Keeps only non-silent spans
4. Merges multiple spans together until the configured cloning duration is reached
5. Normalizes the merged sample audio
6. Creates an ElevenLabs instant voice clone automatically when auto-cloning is enabled

Default clone duration settings are 60 seconds minimum, 120 seconds target, and 300 seconds maximum. If a speaker does not have enough clean material, automatic cloning is disabled, or ElevenLabs rejects the clone, the stage stops for review. Inspect `voice_prep_manifest.json`, create or assign a `voice_id` manually if needed, then rerun.

### 4. Alignment Overflow
**`{job}/05_alignment/alignment_issues.json`**

If a synthesized segment would need more stretching than the configured `max_duration_stretch` allows, alignment stops and records the issue. Shorten the translation or adjust timing settings, then rerun.

### 5. Source Separation Setup Problems
If Demucs is not available, or a configured separation command fails, the pipeline stops with instructions. Install/fix Demucs or update `source_separation_command` and rerun.

---

## Project Layout

```text
dubbing_pipeline.py              ← entry point
dubbing_pipeline/
├── cli.py                       ← interactive wizard and prompts
├── media.py                     ← FFmpeg / Rubber Band wrappers
├── models.py                    ← data classes and enums
├── pipeline.py                  ← runtime context and sequential stage runner
├── review.py                    ← CSV review schemas and parsing helpers
├── stages.py                    ← all 9 pipeline stage implementations
├── state.py                     ← manifest persistence and job discovery
└── providers/
    ├── elevenlabs.py            ← transcription and synthesis client
    └── openrouter.py            ← translation client
tests/
├── test_review.py
├── test_stages.py
└── test_state.py
{jobs_root}/                     ← user-chosen root, not fixed to the repo
└── {job_name}/
    ├── manifest.json
    ├── 01_source/
    │   ├── input_audio.wav
    │   ├── vocals.wav
    │   └── instrumental.wav
    ├── 02_transcript/
    │   └── transcript_raw.json
    ├── 02_transcript_review/
    │   ├── transcript_segments.json
    │   ├── transcript_review.csv
    │   └── transcript_segments_approved.json
    ├── 03_translation/
    │   ├── translation_review.csv
    │   └── translation_segments.json
    ├── 03_translation_review/
    │   └── translation_segments_approved.json
    ├── 04_voice_prep/
    │   ├── speaker_samples/
    │   │   └── {speaker}/
    │   │       ├── raw_segments/
    │   │       ├── clone_source.wav
    │   │       └── clone_source_normalized.wav
    │   └── voice_prep_manifest.json
    ├── 04_synthesis/
    │   └── synthesis_manifest.json
    ├── 05_alignment/
    │   ├── alignment_manifest.json
    │   └── alignment_issues.json
    └── 06_mix/
        ├── final_manifest.json
        └── output.mp4
```

---

## How It Works

### Job Persistence

Every job is backed by a `manifest.json` that stores source media, settings, stage status, timestamps, messages, and artifact paths. You can stop and resume later; completed stages are skipped only when their outputs are still considered reusable.

Resume validation is strongest for source separation, synthesis, alignment, and final mix. Earlier stages rely more on recorded status plus expected artifact files.

### Source Media Requirements

The source video path is always required because the final stage remuxes the finished dub onto the original video stream.

If you provide a separate audio file, the pipeline copies it directly into `01_source/input_audio.wav` and uses it as the source audio for separation/transcription. It is not transcoded on ingest.

### Source Separation

If `source_separation_command` is not configured, the pipeline auto-detects **Demucs** only. Other backends are still possible, but they must be invoked through a custom command template.

Available placeholders for a custom command are:

- `{input_video}`
- `{input_audio}`
- `{output_dir}`
- `{vocals_path}`
- `{instrumental_path}`

### Simple Test Mode

Simple mode is intended for lightweight testing. It reduces the number of prompts, disables diarization, and forces a single-speaker workflow so the rest of the pipeline can be exercised more quickly.

### Translation

Translation is performed through OpenRouter chat completions. The default model is `minimax/minimax-m2.5:free`, and the pipeline can prompt you to retry, switch models, or stop if the current model is rate-limited or unavailable.

The workflow is designed for Russian-to-English dubbing. The wizard stores source and target language codes, but the current translation prompt is English-oriented in practice.

### Voice Mapping and Automatic Cloning

Voice prep no longer stops at raw speaker sample extraction alone. It now builds clone-ready sample audio per speaker from the vocals stem, using diarized transcript timing plus silence detection to merge only usable speech spans.

When `auto_clone_voices` is enabled, the pipeline sends the normalized merged sample to ElevenLabs instant voice cloning and stores the returned `voice_id` in `speaker_voice_map` automatically. Existing speaker mappings are reused as-is, so reruns do not reclone speakers that already have a mapped voice.

Clone preparation is controlled by these settings:

- `voice_clone_prefix`
- `voice_clone_min_seconds`
- `voice_clone_target_seconds`
- `voice_clone_max_seconds`
- `voice_clone_remove_background_noise`

If automatic cloning cannot complete for a speaker, `04_voice_prep/voice_prep_manifest.json` records the status for that speaker, including cases such as `insufficient_audio`, `clone_ready_manual_voice_id_needed`, or `clone_failed`.

### Non-Interactive Voice Cloning Flags

For non-interactive runs, the CLI supports:

- `--disable-auto-clone-voices`
- `--voice-clone-prefix`
- `--voice-clone-min-seconds`
- `--voice-clone-target-seconds`
- `--voice-clone-max-seconds`
- `--voice-clone-remove-background-noise`
- `--voice-id` to force one voice for every speaker when desired

---

## Review CSV Schemas

### Transcript Review CSV
**`{job}/02_transcript_review/transcript_review.csv`**

Columns:

- `segment_id`
- `speaker`
- `start_sec`
- `end_sec`
- `text_ru_raw`
- `text_ru_final`
- `speaker_final`
- `approved`
- `notes`

### Translation Review CSV
**`{job}/03_translation/translation_review.csv`**

Columns:

- `segment_id`
- `speaker`
- `start_sec`
- `end_sec`
- `text_ru_final`
- `text_en_draft`
- `text_en_final`
- `duration_sec`
- `syllables_per_sec`
- `overflow`
- `approved`
- `notes`

---

## Tests

Run the unit test suite with:

```bash
python3.14 -m unittest discover -s tests
```

The repository currently includes tests for stage behavior, manifest persistence, and review CSV round-tripping.

---

## Notes

- Core HTTP/API calls use `urllib` from the Python standard library
- Timing behavior is configurable through settings such as syllables-per-second, max duration stretch, segment gap threshold, and segment duration limits
- Voice prep can automatically clone each detected speaker from merged clean sample audio, but manual intervention is still possible when clone creation needs review
- External tools and APIs are wrapped with explicit error messages intended to make reruns and recovery easier
