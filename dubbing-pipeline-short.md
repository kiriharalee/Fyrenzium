# AI Video Dubbing Pipeline: Russian → English

**Updated March 2026 — All models verified as top-of-benchmark for their task.**

---

## Stage 1 — Source Separation

**Tool:** Mel-RoFormer (via UVR5)

Extracts all human voices into one clean vocal track and everything else (music, SFX, ambience) into a separate instrumental track. Mel-RoFormer is the current leader for vocal isolation, outperforming Demucs by 2+ dB SDR on standard benchmarks.

Run via UVR5 (GUI) or MVSEP. Input: full video audio. Output: `vocals.wav` + `instrumental.wav`.

---

## Stage 2 — Transcription + Speaker Diarization

**Tool:** ElevenLabs Scribe v2 (batch mode)

Feed the isolated vocal track from Stage 1 into Scribe v2. A single API call returns a structured JSON with the full Russian transcript, word-level timestamps, speaker IDs, and confidence scores. Supports 90+ languages with the lowest WER on industry benchmarks.

Use **keyterm prompting** (up to 1,000 terms) to pre-load Russian names, places, and technical vocabulary for accurate recognition.

---

## Stage 3 — Human Review of Transcription

Fix proper nouns, verify speaker labels are consistent, and check that timestamps align with the audio. Prioritize low-confidence segments from the Scribe output.

---

## Stage 4 — Translation with Timing Awareness

**Tool:** Claude

Translate segment by segment with a prompt that includes original timestamps and instructs Claude to match approximate syllable count / duration of the Russian. Preserve speaker labels. Russian is typically longer than English, which works in your favor.

---

## Stage 5 — Translation Review

Read each English line aloud against the original timing. If a line overflows its segment duration, rephrase shorter. Rule of thumb: ~4.5 syllables per second of available time.

---

## Stage 6 — Voice Cloning + Speech Synthesis

**Tool:** ElevenLabs Dubbing Studio

Two approaches:

- **Full upload:** Upload the original video and let Dubbing Studio handle separation, transcription, translation, voice cloning, and synthesis end-to-end. Review and tweak in the editor.
- **Segment-by-segment API:** Extract 30–60s clean vocal samples per speaker from Stage 1, create Professional Voice Clones, then synthesize each translated segment individually for full control over speed/stability/similarity.

For a 2-hour video, expect to need a Scale or Business plan.

---

## Stage 7 — Timing Alignment

**Tools:** FFmpeg (`atempo`) or Rubberband

Time-stretch synthesized segments to fit original timing windows. Stay within ±15% — beyond that, go back and rephrase the translation shorter.

```bash
ffmpeg -i segment.wav -filter:a "atempo=1.1" segment_stretched.wav
```

---

## Stage 8 — Lip Sync (Optional)

**Tool:** LatentSync 1.6 (quality) or MuseTalk 1.5 (speed)

Only needed for close-up talking-head footage. Skip for B-roll, presentations, or distant/turned faces.

- **LatentSync 1.6** — Best benchmark results, built on Stable Diffusion, excellent temporal consistency. ~8–10 GB VRAM.
- **MuseTalk 1.5** — Real-time (30+ fps), MIT licensed, sharp output. ~6–8 GB VRAM.

---

## Stage 9 — Final Mix + Render

**Tool:** FFmpeg

Combine the English voice track with the original instrumental stem from Stage 1. Target -16 LUFS for voice, duck music ~6 dB below.

```bash
ffmpeg -i video.mp4 -i english_voice.wav -i instrumental.wav \
  -filter_complex "[1:a]volume=0dB[v];[2:a]volume=-6dB[m];[v][m]amix=inputs=2[mixed];[mixed]loudnorm=I=-16:TP=-1.5[out]" \
  -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 192k output.mp4
```

---

## Quick Reference

| Stage | Tool | Type |
|-------|------|------|
| 1. Source separation | Mel-RoFormer (UVR5) | Local |
| 2. Transcription + diarization | ElevenLabs Scribe v2 | API |
| 3. Transcript review | Human | Manual |
| 4. Translation | Claude | API |
| 5. Translation review | Human | Manual |
| 6. Voice cloning + TTS | ElevenLabs Dubbing Studio | API |
| 7. Timing alignment | FFmpeg / Rubberband | Local |
| 8. Lip sync | LatentSync 1.6 / MuseTalk 1.5 | Local |
| 9. Final mix | FFmpeg | Local |

**GPU:** A single RTX 3090/4090 (24 GB) handles all local stages. ElevenLabs stages run in the cloud.
