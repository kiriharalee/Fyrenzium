"""Pipeline stages for the interactive dubbing workflow."""

from __future__ import annotations

import json
import re
import shlex
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

from .media import (
    MediaToolError,
    extract_audio_from_video,
    ffprobe_duration_seconds,
    mix_audio_tracks,
    mix_voice_and_instrumental,
    mux_audio_into_video,
    normalize_audio,
    pad_audio_with_silence,
    require_executable,
    run_command,
    stretch_audio_ffmpeg,
    stretch_audio_rubberband,
    trim_audio_segment,
    which,
)
from .models import StageName, StageStatus, TranscriptSegment, TranscriptWord, TranslationSegment
from .providers import (
    ElevenLabsClient,
    ElevenLabsError,
    OpenRouterAPIError,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterRateLimitError,
)
from .review import (
    TranscriptReviewRow,
    TranslationReviewRow,
    read_transcript_review_csv,
    read_translation_review_csv,
    write_transcript_review_csv,
    write_translation_review_csv,
)

if TYPE_CHECKING:
    from .pipeline import PipelineContext, StageResult


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def extract_words(payload: Mapping[str, Any]) -> List[TranscriptWord]:
    candidates: Iterable[Any] = payload.get("words") or []
    if not candidates:
        for key in ("segments", "utterances"):
            nested = payload.get(key)
            if isinstance(nested, list):
                candidates = nested
                break

    words: List[TranscriptWord] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        if not text:
            continue
        try:
            start_sec = float(item.get("start") or item.get("start_sec") or item.get("start_time"))
            end_sec = float(item.get("end") or item.get("end_sec") or item.get("end_time"))
        except (TypeError, ValueError):
            continue
        speaker = str(item.get("speaker") or item.get("speaker_id") or item.get("speaker_label") or "speaker_1")
        confidence = item.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        words.append(
            TranscriptWord(
                word=text,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence_value,
                speaker=speaker,
            )
        )
    return words


def build_segments(words: Sequence[TranscriptWord], gap_threshold: float, max_segment_sec: float) -> List[TranscriptSegment]:
    if not words:
        return []

    segments: List[TranscriptSegment] = []
    bucket: List[TranscriptWord] = [words[0]]
    current_speaker = words[0].speaker or "speaker_1"

    def flush(index: int) -> None:
        if not bucket:
            return
        text = " ".join(word.word for word in bucket).strip()
        confidences = [word.confidence for word in bucket if word.confidence is not None]
        confidence = sum(confidences) / len(confidences) if confidences else None
        segments.append(
            TranscriptSegment(
                segment_id=f"s{index:04d}",
                speaker=current_speaker,
                start_sec=bucket[0].start_sec,
                end_sec=bucket[-1].end_sec,
                text=text,
                words=list(bucket),
                confidence=confidence,
                needs_review=confidence is None or confidence < 0.75,
            )
        )

    next_index = 1
    for word in words[1:]:
        previous = bucket[-1]
        gap = max(0.0, word.start_sec - previous.end_sec)
        segment_duration = word.end_sec - bucket[0].start_sec
        speaker_changed = (word.speaker or current_speaker) != current_speaker
        if speaker_changed or gap > gap_threshold or segment_duration > max_segment_sec:
            flush(next_index)
            next_index += 1
            bucket = [word]
            current_speaker = word.speaker or "speaker_1"
            continue
        bucket.append(word)

    flush(next_index)
    return segments


def estimate_syllables(text: str) -> int:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0
    total = 0
    for token in tokens:
        count = len(re.findall(r"[aeiouy]+", token))
        if token.endswith("e") and count > 1:
            count -= 1
        total += max(1, count)
    return total


def parse_translation_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = stripped.rstrip("`").strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise MediaToolError("Translation response was not valid JSON.")
        payload = json.loads(match.group(0))
    translation = str(payload.get("translation") or payload.get("text") or "").strip()
    if not translation:
        raise MediaToolError("Translation response did not include a translation field.")
    return translation


def translation_prompt(segment: TranscriptSegment, neighbors_before: Sequence[TranscriptSegment], neighbors_after: Sequence[TranscriptSegment], glossary: Sequence[str], syllables_per_second: float) -> str:
    target_syllables = max(1, round((segment.end_sec - segment.start_sec) * syllables_per_second))
    lines = [
        "Translate the current Russian segment into natural English.",
        "Return strict JSON with exactly one key: translation.",
        "Do not merge or split segments.",
        f"Current segment speaker: {segment.speaker}",
        f"Current segment timing: {segment.start_sec:.3f}-{segment.end_sec:.3f}",
        f"Target syllables: {target_syllables}",
    ]
    if glossary:
        lines.append("Glossary: " + ", ".join(glossary))
    for neighbor in neighbors_before:
        lines.append(f'Previous [{neighbor.segment_id}] {neighbor.speaker}: "{neighbor.text}"')
    lines.append(f'Current [{segment.segment_id}] {segment.speaker}: "{segment.text}"')
    for neighbor in neighbors_after:
        lines.append(f'Next [{neighbor.segment_id}] {neighbor.speaker}: "{neighbor.text}"')
    return "\n".join(lines)


def review_rows_complete(rows: Sequence[Any]) -> bool:
    return bool(rows) and all(bool(getattr(row, "approved", False)) for row in rows)


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    stage_name: StageName

    @abstractmethod
    def run(self, context: "PipelineContext") -> "StageResult":
        raise NotImplementedError


class SourceSeparationStage(PipelineStage):
    stage_name = StageName.SOURCE_SEPARATION

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        input_audio = stage_dir / "input_audio.wav"
        source_audio = context.source_media.audio_path
        if source_audio:
            input_audio.parent.mkdir(parents=True, exist_ok=True)
            if input_audio != source_audio:
                input_audio.write_bytes(source_audio.read_bytes())
        else:
            extract_audio_from_video(context.source_media.video_path, input_audio)

        vocals_path = stage_dir / "vocals.wav"
        instrumental_path = stage_dir / "instrumental.wav"
        command_template = context.settings.source_separation_command.strip()
        if not command_template:
            return StageResult(
                self.stage_name,
                StageStatus.NEEDS_REVIEW,
                "Add a source separation command template, then rerun.",
                {"input_audio": str(input_audio)},
            )

        command = command_template.format(
            input_video=str(context.source_media.video_path),
            input_audio=str(input_audio),
            output_dir=str(stage_dir),
            vocals_path=str(vocals_path),
            instrumental_path=str(instrumental_path),
        )
        run_command(shlex.split(command))
        if not vocals_path.exists() or not instrumental_path.exists():
            raise MediaToolError(
                "Source separation completed but did not create vocals.wav and instrumental.wav."
            )
        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={
                "input_audio": str(input_audio),
                "vocals": str(vocals_path),
                "instrumental": str(instrumental_path),
            },
        )


class TranscriptionStage(PipelineStage):
    stage_name = StageName.TRANSCRIPTION

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        vocals_path = context.stage_artifact_path(StageName.SOURCE_SEPARATION, "vocals")
        if vocals_path is None:
            raise MediaToolError("Missing vocals stem from source separation.")
        api_key = context.api_keys.get("elevenlabs")
        if not api_key:
            raise MediaToolError("Missing ElevenLabs API key.")
        client = ElevenLabsClient(api_key)
        payload = client.transcribe_audio(
            vocals_path,
            model_id=context.settings.elevenlabs_scribe_model,
            language_code=context.settings.source_language,
            diarize=True,
            num_speakers=context.settings.estimated_speakers,
            keyterms=context.settings.scribe_keyterms,
        )
        raw_path = context.stage_dir(self.stage_name) / "transcript_raw.json"
        write_json(raw_path, payload)
        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={"transcript_raw": str(raw_path)},
        )


class TranscriptReviewStage(PipelineStage):
    stage_name = StageName.TRANSCRIPT_REVIEW

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        review_csv = stage_dir / "transcript_review.csv"
        approved_json = stage_dir / "transcript_segments_approved.json"
        if review_csv.exists():
            rows = read_transcript_review_csv(review_csv)
            if review_rows_complete(rows):
                segments = [
                    TranscriptSegment(
                        segment_id=row.segment_id,
                        speaker=row.speaker_final or row.speaker,
                        start_sec=row.start_sec,
                        end_sec=row.end_sec,
                        text=row.text_ru_final or row.text_ru_raw,
                        confidence=None,
                        needs_review=not row.approved,
                    )
                    for row in rows
                ]
                write_json(approved_json, [segment.to_dict() for segment in segments])
                return StageResult(
                    self.stage_name,
                    StageStatus.COMPLETED,
                    artifacts={
                        "transcript_review_csv": str(review_csv),
                        "approved_segments": str(approved_json),
                    },
                )
            return StageResult(
                self.stage_name,
                StageStatus.NEEDS_REVIEW,
                "Review transcript_review.csv and approve all rows before rerunning.",
                {"transcript_review_csv": str(review_csv)},
            )

        raw_path = context.stage_artifact_path(StageName.TRANSCRIPTION, "transcript_raw")
        if raw_path is None:
            raise MediaToolError("Missing raw transcript JSON.")
        payload = read_json(raw_path, default={}) or {}
        words = extract_words(payload)
        segments = build_segments(
            words,
            context.settings.segment_gap_seconds,
            context.settings.max_segment_seconds,
        )
        write_json(stage_dir / "transcript_segments.json", [segment.to_dict() for segment in segments])
        rows = [
            TranscriptReviewRow(
                segment_id=segment.segment_id,
                speaker=segment.speaker,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text_ru_raw=segment.text,
                text_ru_final=segment.text,
                speaker_final=segment.speaker,
                approved=False,
                notes="",
            )
            for segment in segments
        ]
        write_transcript_review_csv(rows, review_csv)
        return StageResult(
            self.stage_name,
            StageStatus.NEEDS_REVIEW,
            "Transcript review CSV created. Approve it and rerun the wizard to continue.",
            {"transcript_review_csv": str(review_csv)},
        )


class TranslationStage(PipelineStage):
    stage_name = StageName.TRANSLATION

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        review_csv = stage_dir / "translation_review.csv"
        if review_csv.exists():
            return StageResult(
                self.stage_name,
                StageStatus.COMPLETED,
                artifacts={"translation_review_csv": str(review_csv)},
            )

        transcript_path = context.stage_artifact_path(StageName.TRANSCRIPT_REVIEW, "approved_segments")
        if transcript_path is None:
            raise MediaToolError("Transcript review has not been approved yet.")
        segments = [TranscriptSegment(**item) for item in read_json(transcript_path, default=[]) or []]
        if not segments:
            raise MediaToolError("No approved transcript segments were found.")

        api_key = context.api_keys.get("openrouter")
        if not api_key:
            raise MediaToolError("Missing OpenRouter API key.")

        model = context.settings.translation_model
        client = OpenRouterClient(api_key)
        rows: List[TranslationReviewRow] = []
        for index, segment in enumerate(segments):
            while True:
                try:
                    prompt = translation_prompt(
                        segment,
                        segments[max(0, index - 2):index],
                        segments[index + 1:index + 3],
                        context.settings.translation_glossary,
                        context.settings.syllables_per_second,
                    )
                    response = client.chat_completions(
                        [
                            {"role": "system", "content": "You translate dubbing segments and return JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    translation = parse_translation_content(client.extract_message_content(response))
                    break
                except OpenRouterRateLimitError as exc:
                    print(f"OpenRouter rate limit: {exc}")
                    action = input("Retry, switch model, or stop? [r/m/s]: ").strip().lower() or "r"
                    if action == "m":
                        model = input(f"OpenRouter model [{model}]: ").strip() or model
                        context.settings.translation_model = model
                        continue
                    if action == "s":
                        return StageResult(
                            self.stage_name,
                            StageStatus.FAILED,
                            "Translation stopped after OpenRouter rate limit.",
                        )
                except (OpenRouterAPIError, OpenRouterError, MediaToolError) as exc:
                    print(f"Translation error: {exc}")
                    action = input("Retry, switch model, or stop? [r/m/s]: ").strip().lower() or "r"
                    if action == "m":
                        model = input(f"OpenRouter model [{model}]: ").strip() or model
                        context.settings.translation_model = model
                        continue
                    if action == "s":
                        return StageResult(
                            self.stage_name,
                            StageStatus.FAILED,
                            "Translation stopped before the draft CSV was complete.",
                        )

            syllables = estimate_syllables(translation)
            duration = max(0.001, segment.end_sec - segment.start_sec)
            rows.append(
                TranslationReviewRow(
                    segment_id=segment.segment_id,
                    speaker=segment.speaker,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    text_ru_final=segment.text,
                    text_en_draft=translation,
                    text_en_final=translation,
                    duration_sec=duration,
                    syllables_per_sec=syllables / duration,
                    overflow=(syllables / duration) > context.settings.syllables_per_second,
                    approved=False,
                    notes="",
                )
            )

        draft_json = stage_dir / "translation_segments.json"
        write_translation_review_csv(rows, review_csv)
        write_json(draft_json, [asdict(row) for row in rows])
        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={
                "translation_review_csv": str(review_csv),
                "translation_segments": str(draft_json),
            },
        )


class TranslationReviewStage(PipelineStage):
    stage_name = StageName.TRANSLATION_REVIEW

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        review_csv = context.stage_artifact_path(StageName.TRANSLATION, "translation_review_csv")
        approved_json = stage_dir / "translation_segments_approved.json"
        if review_csv is None or not review_csv.exists():
            raise MediaToolError("Missing translation review CSV.")

        rows = read_translation_review_csv(review_csv)
        if review_rows_complete(rows):
            segments = [
                TranslationSegment(
                    segment_id=row.segment_id,
                    speaker=row.speaker,
                    start_sec=row.start_sec,
                    end_sec=row.end_sec,
                    source_text=row.text_ru_final,
                    translated_text=row.text_en_final or row.text_en_draft,
                    overflow=row.overflow,
                    notes=row.notes,
                )
                for row in rows
            ]
            write_json(approved_json, [segment.to_dict() for segment in segments])
            return StageResult(
                self.stage_name,
                StageStatus.COMPLETED,
                artifacts={
                    "translation_review_csv": str(review_csv),
                    "approved_translations": str(approved_json),
                },
            )

        return StageResult(
            self.stage_name,
            StageStatus.NEEDS_REVIEW,
            "Review translation_review.csv and approve all rows before synthesis.",
            {"translation_review_csv": str(review_csv)},
        )


class VoicePrepStage(PipelineStage):
    stage_name = StageName.VOICE_PREP

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        transcript_path = context.stage_artifact_path(StageName.TRANSCRIPT_REVIEW, "approved_segments")
        vocals_path = context.stage_artifact_path(StageName.SOURCE_SEPARATION, "vocals")
        if transcript_path is None or vocals_path is None:
            raise MediaToolError("Voice prep needs approved transcript segments and the vocals stem.")

        segments = [TranscriptSegment(**item) for item in read_json(transcript_path, default=[]) or []]
        sample_dir = stage_dir / "speaker_samples"
        manifest_entries: List[Dict[str, Any]] = []
        speakers_seen: Dict[str, int] = {}
        for segment in segments:
            count = speakers_seen.get(segment.speaker, 0)
            if count >= 5:
                continue
            speakers_seen[segment.speaker] = count + 1
            output_path = sample_dir / segment.speaker / f"{segment.segment_id}.wav"
            trim_audio_segment(
                vocals_path,
                output_path,
                start_seconds=segment.start_sec,
                duration_seconds=max(0.1, segment.end_sec - segment.start_sec),
            )
            manifest_entries.append(
                {
                    "speaker": segment.speaker,
                    "sample_path": str(output_path),
                    "segment_id": segment.segment_id,
                    "voice_id": context.settings.speaker_voice_map.get(segment.speaker, ""),
                }
            )

        prep_manifest = stage_dir / "voice_prep_manifest.json"
        write_json(prep_manifest, manifest_entries)
        missing_speakers = sorted(
            {
                entry["speaker"]
                for entry in manifest_entries
                if not entry.get("voice_id")
            }
        )
        if missing_speakers:
            message = (
                "Voice sample packs are ready. Create PVCs externally, then rerun and add voice IDs for: "
                + ", ".join(missing_speakers)
            )
            return StageResult(
                self.stage_name,
                StageStatus.NEEDS_REVIEW,
                message,
                {"voice_prep_manifest": str(prep_manifest)},
            )

        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={"voice_prep_manifest": str(prep_manifest)},
        )


class SynthesisStage(PipelineStage):
    stage_name = StageName.SYNTHESIS

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        translations_path = context.stage_artifact_path(StageName.TRANSLATION_REVIEW, "approved_translations")
        if translations_path is None:
            raise MediaToolError("Translation review must be approved before synthesis.")
        translations = [TranslationSegment(**item) for item in read_json(translations_path, default=[]) or []]
        api_key = context.api_keys.get("elevenlabs")
        if not api_key:
            raise MediaToolError("Missing ElevenLabs API key.")
        client = ElevenLabsClient(api_key, default_model_id=context.settings.elevenlabs_tts_model)
        manifest_entries: List[Dict[str, Any]] = []
        for segment in translations:
            voice_id = context.settings.speaker_voice_map.get(segment.speaker)
            if not voice_id:
                return StageResult(
                    self.stage_name,
                    StageStatus.NEEDS_REVIEW,
                    f"Missing voice_id for speaker {segment.speaker}.",
                )
            output_path = stage_dir / segment.speaker / f"{segment.segment_id}.mp3"
            client.save_speech_to_file(
                segment.translated_text,
                voice_id,
                output_path,
                model_id=context.settings.elevenlabs_tts_model,
                output_format="mp3_44100_128",
            )
            manifest_entries.append(
                {
                    "segment_id": segment.segment_id,
                    "speaker": segment.speaker,
                    "voice_id": voice_id,
                    "start_sec": segment.start_sec,
                    "target_duration_sec": max(0.001, segment.end_sec - segment.start_sec),
                    "audio_path": str(output_path),
                    "duration_sec": ffprobe_duration_seconds(output_path),
                }
            )
        synthesis_manifest = stage_dir / "synthesis_manifest.json"
        write_json(synthesis_manifest, manifest_entries)
        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={"synthesis_manifest": str(synthesis_manifest)},
        )


class AlignmentStage(PipelineStage):
    stage_name = StageName.ALIGNMENT

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        synthesis_path = context.stage_artifact_path(StageName.SYNTHESIS, "synthesis_manifest")
        if synthesis_path is None:
            raise MediaToolError("Synthesis must complete before alignment.")
        items = read_json(synthesis_path, default=[]) or []
        issues: List[Dict[str, Any]] = []
        aligned: List[Dict[str, Any]] = []
        for item in items:
            source = Path(item["audio_path"])
            actual_duration = float(item["duration_sec"])
            target_duration = float(item["target_duration_sec"])
            speed_factor = actual_duration / target_duration
            aligned_path = stage_dir / f"{item['segment_id']}_aligned.wav"
            if abs(1.0 - speed_factor) > context.settings.max_duration_stretch:
                issues.append(
                    {
                        "segment_id": item["segment_id"],
                        "speaker": item["speaker"],
                        "required_speed_factor": speed_factor,
                        "message": "Segment needs translation shortening or re-phrasing.",
                    }
                )
                continue
            if which("rubberband"):
                stretch_audio_rubberband(source, aligned_path, speed_factor)
            else:
                stretch_audio_ffmpeg(source, aligned_path, speed_factor)
            aligned.append(
                {
                    "segment_id": item["segment_id"],
                    "speaker": item["speaker"],
                    "start_sec": item["start_sec"],
                    "audio_path": str(aligned_path),
                }
            )
        if issues:
            issues_path = stage_dir / "alignment_issues.json"
            write_json(issues_path, issues)
            return StageResult(
                self.stage_name,
                StageStatus.NEEDS_REVIEW,
                "Some lines overflowed their timing window. Review alignment_issues.json.",
                {"alignment_issues": str(issues_path)},
            )
        alignment_manifest = stage_dir / "alignment_manifest.json"
        write_json(alignment_manifest, aligned)
        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={"alignment_manifest": str(alignment_manifest)},
        )


class FinalMixStage(PipelineStage):
    stage_name = StageName.FINAL_MIX

    def run(self, context: "PipelineContext") -> "StageResult":
        from .pipeline import StageResult

        stage_dir = context.stage_dir(self.stage_name)
        alignment_manifest = context.stage_artifact_path(StageName.ALIGNMENT, "alignment_manifest")
        instrumental_path = context.stage_artifact_path(StageName.SOURCE_SEPARATION, "instrumental")
        if alignment_manifest is None or instrumental_path is None:
            raise MediaToolError("Alignment and instrumental stem are required for the final mix.")
        items = read_json(alignment_manifest, default=[]) or []
        if not items:
            raise MediaToolError("No aligned segments were found.")

        positioned_tracks: List[Path] = []
        for item in items:
            positioned_path = stage_dir / "positioned" / f"{item['segment_id']}.wav"
            pad_audio_with_silence(Path(item["audio_path"]), positioned_path, float(item["start_sec"]))
            positioned_tracks.append(positioned_path)

        voice_mix = stage_dir / "english_voice_mix.wav"
        mix_audio_tracks(positioned_tracks, voice_mix)
        normalized_voice = stage_dir / "english_voice_normalized.wav"
        normalize_audio(voice_mix, normalized_voice)
        mixed_audio = stage_dir / "final_mix.wav"
        mix_voice_and_instrumental(normalized_voice, instrumental_path, mixed_audio)
        final_audio = stage_dir / "final_mix_normalized.wav"
        normalize_audio(mixed_audio, final_audio)
        final_video = stage_dir / "output.mp4"
        mux_audio_into_video(context.source_media.video_path, final_audio, final_video)
        final_manifest = stage_dir / "final_manifest.json"
        write_json(
            final_manifest,
            {
                "voice_mix": str(normalized_voice),
                "final_audio": str(final_audio),
                "final_video": str(final_video),
            },
        )
        return StageResult(
            self.stage_name,
            StageStatus.COMPLETED,
            artifacts={"final_manifest": str(final_manifest), "final_video": str(final_video)},
        )


def build_default_stages() -> List[PipelineStage]:
    return [
        SourceSeparationStage(),
        TranscriptionStage(),
        TranscriptReviewStage(),
        TranslationStage(),
        TranslationReviewStage(),
        VoicePrepStage(),
        SynthesisStage(),
        AlignmentStage(),
        FinalMixStage(),
    ]
