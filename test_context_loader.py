import tempfile
import unittest
from pathlib import Path

from context_loader import load


class ContextLoaderTest(unittest.TestCase):
    def test_load_returns_term_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text(
                "AEGIS-AP WiNG VOC 기반 AI 분석 기능 개발 목표합의서",
                encoding="utf-8",
            )

            key_terms, doc_content, term_metadata = load([str(path)])

        self.assertTrue(key_terms)
        self.assertIn("AEGIS-AP", doc_content)
        self.assertIn("canonical_terms", term_metadata)
        self.assertIn("alias_map", term_metadata)
        self.assertIn("priority_terms", term_metadata)

    def test_priority_terms_put_product_names_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text(
                "AEGIS-AP WiNG VOC 기반 AI 분석 기능 개발",
                encoding="utf-8",
            )

            _, _, term_metadata = load([str(path)])

        self.assertIn("AEGIS-AP", term_metadata["priority_terms"])


class ExtractAgendaItemsTest(unittest.TestCase):
    def test_numbered_items_extracted(self):
        from context_loader import extract_agenda_items
        text = (
            "1. VQML: ITU-T 표준화 및 라이선스 사업화\n"
            "   ● 시청자 MOS 수집 : 60% (04/03)\n"
            "2. XTelLM: Base Private SLM 확보\n"
            "3. NDR\n"
        )
        result = extract_agenda_items(text)
        self.assertEqual(result, [
            "VQML: ITU-T 표준화 및 라이선스 사업화",
            "XTelLM: Base Private SLM 확보",
            "NDR",
        ])

    def test_circle_number_pattern(self):
        from context_loader import extract_agenda_items
        text = "① 항목 A\n② 항목 B\n"
        result = extract_agenda_items(text)
        self.assertEqual(result, ["항목 A", "항목 B"])

    def test_paren_number_pattern(self):
        from context_loader import extract_agenda_items
        text = "(1) 항목 가\n(2) 항목 나\n"
        result = extract_agenda_items(text)
        self.assertEqual(result, ["항목 가", "항목 나"])

    def test_fallback_empty_when_one_or_zero(self):
        from context_loader import extract_agenda_items
        self.assertEqual(extract_agenda_items("1. 유일한 항목"), [])
        self.assertEqual(extract_agenda_items(""), [])

    def test_sub_bullets_excluded(self):
        from context_loader import extract_agenda_items
        text = (
            "1. 메인 항목\n"
            "   - 하위 내용\n"
            "   ● 세부 항목\n"
            "2. 다른 항목\n"
        )
        result = extract_agenda_items(text)
        self.assertEqual(result, ["메인 항목", "다른 항목"])


if __name__ == "__main__":
    unittest.main()
