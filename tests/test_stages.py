import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dubbing_pipeline.media import extract_audio_from_video
from dubbing_pipeline.models import PipelineSettings, SourceMedia, StageName, StageStatus, TranscriptWord
from dubbing_pipeline.pipeline import PipelineRunner, StageResult, build_context
from dubbing_pipeline.stages import (
    PipelineStage,
    SourceSeparationStage,
    build_segments,
    estimate_syllables,
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

    def test_extract_audio_from_video_preserves_stereo_by_default(self) -> None:
        with patch("dubbing_pipeline.media.require_executable", return_value="ffmpeg"), patch(
            "dubbing_pipeline.media.run_command"
        ) as run_mock:
            extract_audio_from_video(Path("/tmp/input.mp4"), Path("/tmp/output.wav"))

        command = run_mock.call_args.args[0]
        self.assertIn("-ac", command)
        self.assertEqual(command[command.index("-ac") + 1], "2")


if __name__ == "__main__":
    unittest.main()
