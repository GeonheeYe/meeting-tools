import unittest

import pipeline


class CorrectionTest(unittest.TestCase):
    def test_normalize_terms_uses_alias_map_only(self):
        transcript = "이지스 AP와 VoC 기능을 본다"
        term_metadata = {"alias_map": {"이지스 AP": "AEGIS-AP", "VoC": "VOC"}}
        corrected = pipeline.normalize_terms(transcript, term_metadata)
        self.assertEqual(corrected, "AEGIS-AP와 VOC 기능을 본다")


if __name__ == "__main__":
    unittest.main()
