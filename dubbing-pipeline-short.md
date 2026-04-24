# Fyrenzium Pipeline Summary

This file is a short, implementation-aligned overview of the current repository.
For operational details, file locations, and workflow notes, use `README.md` as the primary reference.

---

## Stage Summary

### 1. Source Separation
- Auto-detects **Demucs** if available
- Otherwise uses a custom `source_separation_command`
- Produces `vocals.wav` and `instrumental.wav`

### 2. Transcription
- Uses **ElevenLabs Scribe v2**
- Accepts source language, optional diarization, estimated speaker count, and keyterms
- Writes raw transcript JSON for later review

### 3. Transcript Review
- Creates `02_transcript_review/transcript_review.csv`
- Lets you correct Russian text and speaker labels
- Requires every row to be approved before continuing

### 4. Translation
- Uses an **OpenRouter-configured chat-completions model**
- Default model: `minimax/minimax-m2.5:free`
- Builds prompts with timing, neighboring context, and glossary hints

### 5. Translation Review
- Uses `03_translation/translation_review.csv`
- Flags rows that likely overflow the available timing window
- Requires approval before synthesis can begin

### 6. Voice Prep
- Extracts per-speaker sample clips into `04_voice_prep/speaker_samples/`
- Can create ElevenLabs instant voice clones automatically when enabled
- Writes `voice_prep_manifest.json`
- Stops for manual voice assignment when cloning is disabled or cannot complete

### 7. Synthesis
- Uses **ElevenLabs TTS** for each approved segment
- Requires a `voice_id` mapping for every speaker
- Writes per-speaker segment audio plus `synthesis_manifest.json`

### 8. Alignment
- Uses **Rubber Band** when available, otherwise FFmpeg `atempo`
- Time-stretches synthesized segments to match original timing
- Stops and records `alignment_issues.json` when stretch exceeds the configured limit

### 9. Final Mix
- Positions aligned speech on the timeline
- Mixes English speech with the instrumental stem
- Normalizes output and remuxes it onto the original video as `output.mp4`

---

## Quick Reference

| Stage | Implementation |
|---|---|
| 1. Source separation | Demucs auto-detect or custom command |
| 2. Transcription | ElevenLabs Scribe v2 |
| 3. Transcript review | Manual CSV approval |
| 4. Translation | OpenRouter model |
| 5. Translation review | Manual CSV approval |
| 6. Voice prep | Speaker sample extraction + optional voice cloning |
| 7. Synthesis | ElevenLabs TTS |
| 8. Alignment | Rubber Band or FFmpeg |
| 9. Final mix | FFmpeg-based render |

---

## Scope Notes

- The implemented workflow is designed for **Russian-to-English dubbing**
- Lip sync is **not** part of the current repository
- The jobs root is chosen at runtime; outputs are tracked through each job’s `manifest.json`
