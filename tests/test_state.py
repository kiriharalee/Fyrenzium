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
        self.assertFalse(loaded.settings.simple_mode)

    def test_manifest_persists_simple_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job"
            manifest = build_manifest(
                "demo",
                job_dir,
                SourceMedia(video_path=Path("/tmp/video.mp4")),
                PipelineSettings(simple_mode=True, estimated_speakers=1),
            )
            save_manifest(manifest)
            loaded = load_manifest(job_dir)
        self.assertTrue(loaded.settings.simple_mode)
        self.assertEqual(loaded.settings.estimated_speakers, 1)

    def test_state_loader_defaults_simple_mode_to_false(self) -> None:
        payload = {
            "job_name": "demo",
            "job_dir": "/tmp/demo-job",
            "source_media": {
                "video_path": "/tmp/video.mp4",
                "audio_path": None,
                "language": "ru",
                "title": None,
            },
            "settings": {
                "source_language": "ru",
                "target_language": "en",
            },
            "stage_records": [],
            "created_at": "2026-04-07T00:00:00+00:00",
            "updated_at": "2026-04-07T00:00:00+00:00",
            "notes": "",
        }

        manifest = load_manifest.__globals__["from_json_dict"](payload)

        self.assertFalse(manifest.settings.simple_mode)

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
