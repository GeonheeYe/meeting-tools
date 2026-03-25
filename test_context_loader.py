import tempfile
import unittest
from pathlib import Path

import pipeline
from context_loader import load


class ContextLoaderTest(unittest.TestCase):
    def test_load_returns_term_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text(
                "AEGIS-AP WiNG VOC 기반 AI 분석 기능 개발 목표합의서",
                encoding="utf-8",
            )
            key_terms, doc_content, term_metadata, agenda_items = load([str(path)])

        self.assertTrue(key_terms)
        self.assertIn("AEGIS-AP", doc_content)
        self.assertIn("canonical_terms", term_metadata)
        self.assertIn("alias_map", term_metadata)
        self.assertIn("priority_terms", term_metadata)
        self.assertIsInstance(agenda_items, list)

    def test_priority_terms_put_product_names_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text(
                "AEGIS-AP WiNG VOC 기반 AI 분석 기능 개발",
                encoding="utf-8",
            )
            _, _, term_metadata, _ = load([str(path)])

        self.assertIn("AEGIS-AP", term_metadata["priority_terms"])

    def test_load_extracts_agenda_from_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agenda.txt"
            path.write_text(
                "1. VQML 표준화\n2. XTelLM 확보\n3. NDR 검토\n",
                encoding="utf-8",
            )
            _, _, _, agenda_items = load([str(path)])

        self.assertEqual(agenda_items, ["VQML 표준화", "XTelLM 확보", "NDR 검토"])

    def test_load_returns_empty_agenda_when_no_agenda_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("일반 텍스트 내용입니다.", encoding="utf-8")
            _, _, _, agenda_items = load([str(path)])

        self.assertEqual(agenda_items, [])

    def test_alias_map_includes_segmented_product_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terms.txt"
            path.write_text("XTelLM VQML MOS", encoding="utf-8")
            _, _, term_metadata, _ = load([str(path)])

        self.assertEqual(term_metadata["alias_map"]["X Tel LM"], "XTelLM")
        self.assertEqual(term_metadata["alias_map"]["V Q M L"], "VQML")
        self.assertEqual(term_metadata["alias_map"]["M O S"], "MOS")

    def test_normalize_terms_uses_segmented_aliases_from_docs(self):
        transcript = "X Tel LM과 V Q M L, M O S를 같이 검토합니다."
        normalized = pipeline.normalize_terms(
            transcript,
            {
                "alias_map": {
                    "X Tel LM": "XTelLM",
                    "V Q M L": "VQML",
                    "M O S": "MOS",
                }
            },
        )

        self.assertEqual(normalized, "XTelLM과 VQML, MOS를 같이 검토합니다.")


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
