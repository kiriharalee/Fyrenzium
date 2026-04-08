import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dubbing_pipeline.media import MediaToolError, extract_audio_from_video
from dubbing_pipeline.models import PipelineSettings, SourceMedia, StageName, StageStatus, TranscriptWord
from dubbing_pipeline.pipeline import PipelineRunner, StageResult, build_context
from dubbing_pipeline.stages import (
    AlignmentStage,
    FinalMixStage,
    PipelineStage,
    SourceSeparationStage,
    SynthesisStage,
    TranscriptionStage,
    TranscriptReviewStage,
    build_segments,
    estimate_syllables,
    normalize_words_to_single_speaker,
    parse_translation_content,
)
from dubbing_pipeline.state import build_manifest, save_manifest, update_stage_status


class StageHelperTests(unittest.TestCase):
    def test_build_segments_splits_on_speaker_change(self) -> None:
        words = [
            TranscriptWord(word="Privet", start_sec=0.0, end_sec=0.3, speaker="speaker_1"),
            TranscriptWord(word="mir", start_sec=0.31, end_sec=0.5, speaker="speaker_1"),
            TranscriptWord(word="hello", start_sec=1.0, end_sec=1.2, speaker="speaker_2"),
        ]
        segments = build_segments(words, gap_threshold=0.4, max_segment_sec=5.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "speaker_1")
        self.assertEqual(segments[1].speaker, "speaker_2")

    def test_parse_translation_content_reads_json(self) -> None:
        self.assertEqual(parse_translation_content('{"translation":"hello"}'), "hello")

    def test_estimate_syllables_returns_positive_count(self) -> None:
        self.assertGreaterEqual(estimate_syllables("hello from fyrenzium"), 3)

    def test_normalize_words_to_single_speaker_relabels_all_words(self) -> None:
        words = [
            TranscriptWord(word="Privet", start_sec=0.0, end_sec=0.3, speaker="speaker_a"),
            TranscriptWord(word="mir", start_sec=0.31, end_sec=0.5, speaker="speaker_b"),
        ]

        normalized = normalize_words_to_single_speaker(words)

        self.assertEqual([word.speaker for word in normalized], ["speaker_1", "speaker_1"])
        self.assertEqual([word.word for word in normalized], ["Privet", "mir"])

    def test_transcription_stage_uses_simple_mode_single_speaker_settings(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(simple_mode=True, estimated_speakers=1),
            )
            manifest = update_stage_status(
                manifest,
                StageName.SOURCE_SEPARATION,
                StageStatus.COMPLETED,
                "done",
                {"vocals": str(job_dir / "01_source" / "vocals.wav")},
            )
            save_manifest(manifest)
            vocals_path = job_dir / "01_source" / "vocals.wav"
            vocals_path.parent.mkdir(parents=True, exist_ok=True)
            vocals_path.write_bytes(b"fake-vocals")
            context = build_context(job_dir, {"elevenlabs": "test-key"})

            with patch("dubbing_pipeline.stages.ElevenLabsClient.transcribe_audio", return_value={"words": []}) as transcribe_mock:
                result = TranscriptionStage().run(context)

        self.assertEqual(result.status.value, "completed")
        self.assertEqual(transcribe_mock.call_args.kwargs["diarize"], False)
        self.assertEqual(transcribe_mock.call_args.kwargs["num_speakers"], 1)

    def test_transcript_review_stage_normalizes_speakers_in_simple_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(simple_mode=True),
            )
            raw_path = job_dir / "02_transcript" / "transcript_raw.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                '[{"word":"Privet","start":0.0,"end":0.3,"speaker":"speaker_a"},'
                '{"word":"mir","start":0.31,"end":0.5,"speaker":"speaker_b"}]\n',
                encoding="utf-8",
            )
            manifest = update_stage_status(
                manifest,
                StageName.TRANSCRIPTION,
                StageStatus.COMPLETED,
                "done",
                {"transcript_raw": str(raw_path)},
            )
            save_manifest(manifest)
            context = build_context(job_dir, {})

            result = TranscriptReviewStage().run(context)

            review_csv_path = Path(result.artifacts["transcript_review_csv"])
            review_csv = review_csv_path.read_text(encoding="utf-8")

        self.assertEqual(result.status.value, "needs_review")
        self.assertIn("speaker_1", review_csv)
        self.assertNotIn("speaker_a", review_csv)
        self.assertNotIn("speaker_b", review_csv)

    def test_source_separation_reuses_existing_input_audio(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(),
            )
            save_manifest(manifest)
            input_audio = job_dir / "01_source" / "input_audio.wav"
            input_audio.parent.mkdir(parents=True, exist_ok=True)
            input_audio.write_bytes(b"fake wav data")
            context = build_context(job_dir, {})

            with patch("dubbing_pipeline.stages.extract_audio_from_video") as extract_mock, patch(
                "dubbing_pipeline.stages.which", return_value=None
            ):
                result = SourceSeparationStage().run(context)

        extract_mock.assert_not_called()
        self.assertEqual(result.status.value, "needs_review")
        self.assertEqual(result.artifacts["input_audio"], str(input_audio))

    def test_source_separation_reports_unimplemented_runner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            source_audio = Path(temp_dir) / "source.wav"
            source_audio.write_bytes(b"fake wav data")
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4"), audio_path=source_audio),
                PipelineSettings(source_separation_runner="uvr5"),
            )
            save_manifest(manifest)
            context = build_context(job_dir, {})

            with patch("dubbing_pipeline.stages.which", return_value=None):
                result = SourceSeparationStage().run(context)

        self.assertEqual(result.status.value, "needs_review")
        self.assertIn("source_separation_runner", result.message)
        self.assertIn("uvr5", result.message)

    def test_source_separation_demucs_run_records_provenance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            source_audio = Path(temp_dir) / "source.wav"
            source_audio.write_bytes(b"input-audio")
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4"), audio_path=source_audio),
                PipelineSettings(),
            )
            save_manifest(manifest)
            context = build_context(job_dir, {})

            def fake_demucs_run(args):
                output_root = job_dir / "01_source" / "demucs_output" / "htdemucs" / "input_audio"
                output_root.mkdir(parents=True, exist_ok=True)
                (output_root / "vocals.wav").write_bytes(b"isolated-vocals")
                (output_root / "no_vocals.wav").write_bytes(b"isolated-music")

            with patch("dubbing_pipeline.stages.which", return_value="/usr/bin/demucs"), patch(
                "dubbing_pipeline.stages.run_command", side_effect=fake_demucs_run
            ), patch.object(
                SourceSeparationStage, "_probe_audio", return_value={"codec_name": "pcm_s16le", "channels": 2, "sample_rate": 44100, "duration": "1.0", "size": "16"}
            ), patch.object(SourceSeparationStage, "_audio_peak_dbfs", return_value=-12.0):
                result = SourceSeparationStage().run(context)

        self.assertEqual(result.status.value, "completed")
        self.assertEqual(result.artifacts["source_separation_backend"], "demucs")
        self.assertIn("/usr/bin/demucs --two-stems vocals", result.artifacts["source_separation_command"])
        self.assertTrue(result.artifacts["source_separation_raw_output_dir"].endswith("/01_source/demucs_output"))
        self.assertIn("vocals_stream_info", result.artifacts)

    def test_source_separation_demucs_backend_error_needs_review(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            source_audio = Path(temp_dir) / "source.wav"
            source_audio.write_bytes(b"input-audio")
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4"), audio_path=source_audio),
                PipelineSettings(),
            )
            save_manifest(manifest)
            context = build_context(job_dir, {})

            error = MediaToolError(
                "Command failed with exit code 1: /usr/bin/demucs --two-stems vocals "
                "-o /tmp/out /tmp/input.wav\n"
                "RuntimeError: Couldn't find appropriate backend to handle uri "
                "/tmp/out/vocals.wav and format None.\n"
                "torchaudio backend unavailable"
            )

            with patch("dubbing_pipeline.stages.which", return_value="/usr/bin/demucs"), patch(
                "dubbing_pipeline.stages.run_command", side_effect=error
            ):
                result = SourceSeparationStage().run(context)

        self.assertEqual(result.status, StageStatus.NEEDS_REVIEW)
        self.assertEqual(result.artifacts["input_audio"], str(job_dir / "01_source" / "input_audio.wav"))
        self.assertIn("Demucs was detected", result.message)
        self.assertIn("source_separation_command", result.message)
        self.assertIn("/usr/bin/demucs --two-stems vocals", result.message)

    def test_source_separation_rejects_identical_vocals_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(),
            )
            save_manifest(manifest)
            input_audio = job_dir / "01_source" / "input_audio.wav"
            vocals = job_dir / "01_source" / "vocals.wav"
            instrumental = job_dir / "01_source" / "instrumental.wav"
            input_audio.parent.mkdir(parents=True, exist_ok=True)
            input_audio.write_bytes(b"same-audio")
            vocals.write_bytes(b"same-audio")
            instrumental.write_bytes(b"music")
            artifacts = {
                "input_audio": str(input_audio),
                "vocals": str(vocals),
                "instrumental": str(instrumental),
                "source_separation_backend": "demucs",
                "source_separation_command": "/usr/bin/demucs",
                "source_separation_validation_version": "1",
                "input_audio_sha1": SourceSeparationStage()._sha1(input_audio),
                "input_audio_stream_info": "{}",
                "vocals_stream_info": "{}",
                "instrumental_stream_info": "{}",
                "source_separation_raw_output_dir": str(job_dir / "01_source" / "demucs_output"),
            }
            (job_dir / "01_source" / "demucs_output").mkdir(parents=True, exist_ok=True)
            updated = update_stage_status(
                manifest,
                StageName.SOURCE_SEPARATION,
                StageStatus.COMPLETED,
                "done",
                artifacts,
            )
            save_manifest(updated)
            context = build_context(job_dir, {})

            with patch.object(SourceSeparationStage, "_audio_peak_dbfs", return_value=-12.0):
                reusable = SourceSeparationStage().completed_result_is_reusable(context)

        self.assertFalse(reusable)

    def test_source_separation_rejects_silent_instrumental_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(),
            )
            save_manifest(manifest)
            input_audio = job_dir / "01_source" / "input_audio.wav"
            vocals = job_dir / "01_source" / "vocals.wav"
            instrumental = job_dir / "01_source" / "instrumental.wav"
            input_audio.parent.mkdir(parents=True, exist_ok=True)
            input_audio.write_bytes(b"input-audio")
            vocals.write_bytes(b"vocals")
            instrumental.write_bytes(b"silence")
            artifacts = {
                "input_audio": str(input_audio),
                "vocals": str(vocals),
                "instrumental": str(instrumental),
                "source_separation_backend": "demucs",
                "source_separation_command": "/usr/bin/demucs",
                "source_separation_validation_version": "1",
                "input_audio_sha1": SourceSeparationStage()._sha1(input_audio),
                "input_audio_stream_info": "{}",
                "vocals_stream_info": "{}",
                "instrumental_stream_info": "{}",
                "source_separation_raw_output_dir": str(job_dir / "01_source" / "demucs_output"),
            }
            (job_dir / "01_source" / "demucs_output").mkdir(parents=True, exist_ok=True)
            updated = update_stage_status(
                manifest,
                StageName.SOURCE_SEPARATION,
                StageStatus.COMPLETED,
                "done",
                artifacts,
            )
            save_manifest(updated)
            context = build_context(job_dir, {})

            with patch.object(SourceSeparationStage, "_audio_peak_dbfs", return_value=-100.0):
                reusable = SourceSeparationStage().completed_result_is_reusable(context)

        self.assertFalse(reusable)

    def test_pipeline_does_not_skip_invalid_completed_source_separation(self) -> None:
        class RejectingCompletedStage(PipelineStage):
            stage_name = StageName.SOURCE_SEPARATION

            def __init__(self) -> None:
                self.run_calls = 0

            def completed_result_is_reusable(self, context) -> bool:
                return False

            def run(self, context) -> StageResult:
                self.run_calls += 1
                return StageResult(self.stage_name, StageStatus.NEEDS_REVIEW, "invalid outputs")

        class FollowupStage(PipelineStage):
            stage_name = StageName.TRANSCRIPTION

            def __init__(self) -> None:
                self.run_calls = 0

            def run(self, context) -> StageResult:
                self.run_calls += 1
                return StageResult(self.stage_name, StageStatus.COMPLETED)

        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(),
            )
            manifest = update_stage_status(
                manifest,
                StageName.SOURCE_SEPARATION,
                StageStatus.COMPLETED,
                "done",
                {"input_audio": str(job_dir / "01_source" / "input_audio.wav")},
            )
            save_manifest(manifest)
            runner_stage = RejectingCompletedStage()
            followup_stage = FollowupStage()
            runner = PipelineRunner(build_context(job_dir, {}), [runner_stage, followup_stage])

            final_manifest = runner.run(resume=True)

        self.assertEqual(runner_stage.run_calls, 1)
        self.assertEqual(followup_stage.run_calls, 0)
        source_record = next(
            item for item in final_manifest.stage_records if item.name == StageName.SOURCE_SEPARATION
        )
        self.assertEqual(source_record.status, StageStatus.NEEDS_REVIEW)

    def test_pipeline_marks_media_tool_error_as_failed(self) -> None:
        class FailingStage(PipelineStage):
            stage_name = StageName.SOURCE_SEPARATION

            def run(self, context) -> StageResult:
                raise MediaToolError("ffmpeg failed")

        class FollowupStage(PipelineStage):
            stage_name = StageName.TRANSCRIPTION

            def __init__(self) -> None:
                self.run_calls = 0

            def run(self, context) -> StageResult:
                self.run_calls += 1
                return StageResult(self.stage_name, StageStatus.COMPLETED)

        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(),
            )
            save_manifest(manifest)
            followup_stage = FollowupStage()
            runner = PipelineRunner(build_context(job_dir, {}), [FailingStage(), followup_stage])

            final_manifest = runner.run(resume=True)

        source_record = next(
            item for item in final_manifest.stage_records if item.name == StageName.SOURCE_SEPARATION
        )
        self.assertEqual(source_record.status, StageStatus.FAILED)
        self.assertEqual(source_record.message, "ffmpeg failed")
        self.assertEqual(followup_stage.run_calls, 0)

    def test_pipeline_persists_demucs_backend_error_as_needs_review(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            source_audio = Path(temp_dir) / "source.wav"
            source_audio.write_bytes(b"input-audio")
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4"), audio_path=source_audio),
                PipelineSettings(),
            )
            save_manifest(manifest)

            error = MediaToolError(
                "Command failed with exit code 1: /usr/bin/demucs --two-stems vocals "
                "-o /tmp/out /tmp/input.wav\n"
                "RuntimeError: Couldn't find appropriate backend to handle uri "
                "/tmp/out/vocals.wav and format None."
            )

            with patch("dubbing_pipeline.stages.which", return_value="/usr/bin/demucs"), patch(
                "dubbing_pipeline.stages.run_command", side_effect=error
            ):
                runner = PipelineRunner(build_context(job_dir, {}), [SourceSeparationStage()])
                final_manifest = runner.run(resume=True)

        source_record = next(
            item for item in final_manifest.stage_records if item.name == StageName.SOURCE_SEPARATION
        )
        self.assertEqual(source_record.status, StageStatus.NEEDS_REVIEW)
        self.assertNotEqual(source_record.status, StageStatus.IN_PROGRESS)
        self.assertEqual(
            source_record.artifacts["input_audio"],
            str(job_dir / "01_source" / "input_audio.wav"),
        )
        self.assertIn("Demucs was detected", source_record.message)

    def test_extract_audio_from_video_preserves_stereo_by_default(self) -> None:
        with patch("dubbing_pipeline.media.require_executable", return_value="ffmpeg"), patch(
            "dubbing_pipeline.media.run_command"
        ) as run_mock:
            extract_audio_from_video(Path("/tmp/input.mp4"), Path("/tmp/output.wav"))

        command = run_mock.call_args.args[0]
        self.assertIn("-ac", command)
        self.assertEqual(command[command.index("-ac") + 1], "2")

    def test_synthesis_reuse_rejects_changed_voice_mapping(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            approved_translations = job_dir / "03_translation_review" / "translation_segments_approved.json"
            approved_translations.parent.mkdir(parents=True, exist_ok=True)
            approved_translations.write_text(
                '[{"segment_id":"s0001","speaker":"speaker_1","start_sec":0.0,"end_sec":1.0,"source_text":"privet","translated_text":"hello"}]\n',
                encoding="utf-8",
            )
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(speaker_voice_map={"speaker_1": "voice-old"}),
            )
            manifest = update_stage_status(
                manifest,
                StageName.TRANSLATION_REVIEW,
                StageStatus.COMPLETED,
                "done",
                {"approved_translations": str(approved_translations)},
            )
            save_manifest(manifest)
            context = build_context(job_dir, {"elevenlabs": "test-key"})

            def fake_save_speech_to_file(text, voice_id, output_path, **kwargs):
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{voice_id}:{text}".encode("utf-8"))
                return path

            with patch("dubbing_pipeline.stages.ffprobe_duration_seconds", return_value=0.8), patch.object(
                SynthesisStage, "completed_result_is_reusable", return_value=False
            ), patch(
                "dubbing_pipeline.stages.ElevenLabsClient.save_speech_to_file",
                side_effect=fake_save_speech_to_file,
            ):
                result = SynthesisStage().run(context)

            manifest = update_stage_status(
                context.manifest,
                StageName.SYNTHESIS,
                result.status,
                result.message,
                result.artifacts,
            )
            save_manifest(manifest)
            reusable_context = build_context(job_dir, {"elevenlabs": "test-key"})
            self.assertTrue(SynthesisStage().completed_result_is_reusable(reusable_context))

            reusable_context.manifest.settings.speaker_voice_map["speaker_1"] = "voice-new"
            self.assertFalse(SynthesisStage().completed_result_is_reusable(reusable_context))

    def test_alignment_reuse_rejects_changed_synthesis_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            synthesis_manifest = job_dir / "04_synthesis" / "synthesis_manifest.json"
            audio_path = job_dir / "04_synthesis" / "speaker_1" / "s0001.mp3"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"audio")
            synthesis_manifest.write_text(
                f'[{{"segment_id":"s0001","speaker":"speaker_1","voice_id":"voice-old","start_sec":0.0,"target_duration_sec":1.0,"audio_path":"{audio_path}","duration_sec":0.8}}]\n',
                encoding="utf-8",
            )
            alignment_manifest = job_dir / "05_alignment" / "alignment_manifest.json"
            aligned_path = job_dir / "05_alignment" / "s0001_aligned.wav"
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_path.write_bytes(b"aligned")
            alignment_manifest.write_text(
                f'[{{"segment_id":"s0001","speaker":"speaker_1","start_sec":0.0,"audio_path":"{aligned_path}"}}]\n',
                encoding="utf-8",
            )
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(max_duration_stretch=0.4),
            )
            manifest = update_stage_status(
                manifest,
                StageName.SYNTHESIS,
                StageStatus.COMPLETED,
                "done",
                {"synthesis_manifest": str(synthesis_manifest)},
            )
            save_manifest(manifest)
            context = build_context(job_dir, {})
            alignment_stage = AlignmentStage()
            manifest = update_stage_status(
                manifest,
                StageName.ALIGNMENT,
                StageStatus.COMPLETED,
                "done",
                {
                    "alignment_manifest": str(alignment_manifest),
                    "alignment_inputs_fingerprint": alignment_stage._inputs_fingerprint(context, synthesis_manifest),
                },
            )
            save_manifest(manifest)

            reusable_context = build_context(job_dir, {})
            self.assertTrue(alignment_stage.completed_result_is_reusable(reusable_context))

            synthesis_manifest.write_text(
                f'[{{"segment_id":"s0001","speaker":"speaker_1","voice_id":"voice-new","start_sec":0.0,"target_duration_sec":1.0,"audio_path":"{audio_path}","duration_sec":0.8}}]\n',
                encoding="utf-8",
            )
            self.assertFalse(alignment_stage.completed_result_is_reusable(reusable_context))

    def test_final_mix_reuse_rejects_changed_alignment_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job1"
            alignment_manifest = job_dir / "05_alignment" / "alignment_manifest.json"
            aligned_path = job_dir / "05_alignment" / "s0001_aligned.wav"
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_path.write_bytes(b"aligned")
            alignment_manifest.write_text(
                f'[{{"segment_id":"s0001","speaker":"speaker_1","start_sec":0.0,"audio_path":"{aligned_path}"}}]\n',
                encoding="utf-8",
            )
            instrumental_path = job_dir / "01_source" / "instrumental.wav"
            instrumental_path.parent.mkdir(parents=True, exist_ok=True)
            instrumental_path.write_bytes(b"instrumental")
            final_manifest = job_dir / "06_mix" / "final_manifest.json"
            voice_mix = job_dir / "06_mix" / "english_voice_normalized.wav"
            final_audio = job_dir / "06_mix" / "final_mix_normalized.wav"
            final_video = job_dir / "06_mix" / "output.mp4"
            final_manifest.parent.mkdir(parents=True, exist_ok=True)
            for path in (voice_mix, final_audio, final_video):
                path.write_bytes(b"output")
            final_manifest.write_text(
                (
                    "{\n"
                    f'  "voice_mix": "{voice_mix}",\n'
                    f'  "final_audio": "{final_audio}",\n'
                    f'  "final_video": "{final_video}"\n'
                    "}\n"
                ),
                encoding="utf-8",
            )
            manifest = build_manifest(
                "job1",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(),
            )
            manifest = update_stage_status(
                manifest,
                StageName.ALIGNMENT,
                StageStatus.COMPLETED,
                "done",
                {"alignment_manifest": str(alignment_manifest)},
            )
            manifest = update_stage_status(
                manifest,
                StageName.SOURCE_SEPARATION,
                StageStatus.COMPLETED,
                "done",
                {"instrumental": str(instrumental_path)},
            )
            final_mix_stage = FinalMixStage()
            manifest = update_stage_status(
                manifest,
                StageName.FINAL_MIX,
                StageStatus.COMPLETED,
                "done",
                {
                    "final_manifest": str(final_manifest),
                    "final_video": str(final_video),
                    "final_mix_inputs_fingerprint": final_mix_stage._inputs_fingerprint(
                        alignment_manifest,
                        instrumental_path,
                    ),
                },
            )
            save_manifest(manifest)

            reusable_context = build_context(job_dir, {})
            self.assertTrue(final_mix_stage.completed_result_is_reusable(reusable_context))

            alignment_manifest.write_text("[]\n", encoding="utf-8")
            self.assertFalse(final_mix_stage.completed_result_is_reusable(reusable_context))


if __name__ == "__main__":
    unittest.main()
