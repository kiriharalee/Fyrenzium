import tempfile
import unittest
from pathlib import Path

from dubbing_pipeline.review import (
    TranscriptReviewRow,
    TranslationReviewRow,
    read_transcript_review_csv,
    read_translation_review_csv,
    write_transcript_review_csv,
    write_translation_review_csv,
)


class ReviewCsvTests(unittest.TestCase):
    def test_transcript_review_round_trip(self) -> None:
        rows = [
            TranscriptReviewRow(
                segment_id="s0001",
                speaker="speaker_1",
                start_sec=0.0,
                end_sec=1.2,
                text_ru_raw="privet",
                text_ru_final="Privet",
                speaker_final="speaker_1",
                approved=True,
                notes="ok",
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.csv"
            write_transcript_review_csv(rows, path)
            loaded = read_transcript_review_csv(path)
        self.assertEqual(loaded[0].segment_id, "s0001")
        self.assertTrue(loaded[0].approved)

    def test_translation_review_round_trip(self) -> None:
        rows = [
            TranslationReviewRow(
                segment_id="s0001",
                speaker="speaker_1",
                start_sec=0.0,
                end_sec=1.2,
                text_ru_final="privet",
                text_en_draft="hello",
                text_en_final="hello",
                duration_sec=1.2,
                syllables_per_sec=1.0,
                overflow=False,
                approved=True,
                notes="ok",
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "translation.csv"
            write_translation_review_csv(rows, path)
            loaded = read_translation_review_csv(path)
        self.assertEqual(loaded[0].text_en_final, "hello")
        self.assertTrue(loaded[0].approved)


if __name__ == "__main__":
    unittest.main()
