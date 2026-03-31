import unittest

from dubbing_pipeline.models import TranscriptWord
from dubbing_pipeline.stages import build_segments, estimate_syllables, parse_translation_content


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


if __name__ == "__main__":
    unittest.main()
