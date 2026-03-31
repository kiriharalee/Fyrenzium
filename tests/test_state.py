import tempfile
import unittest
from pathlib import Path

from dubbing_pipeline.models import PipelineSettings, SourceMedia, StageName, StageStatus
from dubbing_pipeline.state import build_manifest, load_manifest, save_manifest, update_stage_status


class StateTests(unittest.TestCase):
    def test_manifest_persists_settings_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job"
            manifest = build_manifest(
                "demo",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(
                    translation_glossary=["term-a"],
                    speaker_voice_map={"speaker_1": "voice-123"},
                ),
            )
            save_manifest(manifest)
            loaded = load_manifest(job_dir)
        self.assertEqual(loaded.settings.translation_glossary, ["term-a"])
        self.assertEqual(loaded.settings.speaker_voice_map["speaker_1"], "voice-123")

    def test_stage_status_update_sets_completion(self) -> None:
        manifest = build_manifest(
            "demo",
            Path("/tmp/demo-job"),
            SourceMedia(video_path=Path("/tmp/video.mp4")),
            PipelineSettings(),
        )
        updated = update_stage_status(
            manifest,
            StageName.TRANSLATION,
            StageStatus.COMPLETED,
            "done",
            {"artifact": "value"},
        )
        record = next(item for item in updated.stage_records if item.name == StageName.TRANSLATION)
        self.assertEqual(record.status, StageStatus.COMPLETED)
        self.assertEqual(record.artifacts["artifact"], "value")
        self.assertIsNotNone(record.completed_at)


if __name__ == "__main__":
    unittest.main()
