import unittest

from very_good_service import VeryGoodVideoProcessor, validate_video_constraints


class VeryGoodVideoConstraintsTests(unittest.TestCase):
    def test_rejects_videos_longer_than_ten_minutes(self):
        with self.assertRaises(ValueError):
            validate_video_constraints(file_size_bytes=100, duration_seconds=600.01)

    def test_rejects_videos_larger_than_two_gb(self):
        with self.assertRaises(ValueError):
            validate_video_constraints(file_size_bytes=2 * 1024 * 1024 * 1024 + 1, duration_seconds=30)

    def test_very_good_processor_requires_api_key(self):
        with self.assertRaises(ValueError):
            VeryGoodVideoProcessor(api_key="")


if __name__ == "__main__":
    unittest.main()