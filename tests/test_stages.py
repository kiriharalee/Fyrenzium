import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dubbing_pipeline.models import PipelineSettings, SourceMedia, TranscriptWord
from dubbing_pipeline.pipeline import build_context
from dubbing_pipeline.stages import (
    SourceSeparationStage,
    build_segments,
    estimate_syllables,
    parse_translation_content,
)
from dubbing_pipeline.state import build_manifest, save_manifest


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


if __name__ == "__main__":
    unittest.main()
