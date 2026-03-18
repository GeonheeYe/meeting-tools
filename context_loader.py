# ~/meeting_tools/context_loader.py
"""
참고 문서에서 키워드와 본문을 추출한다.
- key_terms: Whisper initial_prompt에 삽입할 쉼표 구분 핵심 용어
- doc_content: Claude 교정 단계에 전달할 문서 본문
"""
import re
from collections import Counter
from pathlib import Path
from typing import Optional


def extract_text(path: Path) -> str:
    """파일 포맷에 맞게 텍스트 추출."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".docx":
        return _extract_docx(path)
    elif suffix in {".xlsx", ".xls"}:
        return _extract_xlsx(path)
    else:
        print(f"[context_loader] 지원하지 않는 포맷: {suffix}, 건너뜀")
        return ""


def _extract_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_xlsx(path: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    rows = []
    for sheet in wb.worksheets:
        rows.append(f"[시트: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                rows.append(" | ".join(cells))
    wb.close()
    return "\n".join(rows)


def extract_key_terms(text: str, max_terms: int = 30) -> str:
    """
    텍스트에서 핵심 용어 추출 (규칙 기반).
    - 영문 대문자로 시작하는 단어/약어 (BIRD, XQBot, GRPO 등)
    - 한글 명사구 (2자 이상 연속 한글)
    - 중복 제거, max_terms 개까지
    """
    en_terms = re.findall(r'\b[A-Z][a-zA-Z0-9\-]+\b|\b[A-Z]{2,}\b', text)
    ko_terms = re.findall(r'[가-힣]{2,}', text)

    all_terms = en_terms + ko_terms
    counted = Counter(all_terms)

    _STOPWORDS = {
        "있다", "없다", "하다", "된다", "이다", "합니다", "입니다",
        "있는", "없는", "하는", "되는", "이런", "그런", "저런",
        "The", "This", "That", "For", "With", "And", "Are",
        "활용합니다", "프레임워크", "벤치마크",
    }
    terms = [t for t, _ in counted.most_common(max_terms * 2)
             if t not in _STOPWORDS and len(t) >= 2]

    return ", ".join(terms[:max_terms])


def load(doc_paths: list) -> tuple:
    """
    문서 목록을 로드해서 (key_terms, doc_content) 반환.
    - key_terms: initial_prompt에 삽입
    - doc_content: Claude 교정용 전문
    """
    texts = []
    for p in doc_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            print(f"[context_loader] 파일 없음: {path}, 건너뜀")
            continue
        text = extract_text(path)
        if text:
            texts.append(f"=== {path.name} ===\n{text}")

    if not texts:
        return "", ""

    doc_content = "\n\n".join(texts)
    key_terms = extract_key_terms(doc_content)
    print(f"[context_loader] 키워드 추출 완료: {key_terms[:80]}...")
    return key_terms, doc_content


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python context_loader.py <파일1> [파일2] ...")
        sys.exit(1)
    terms, content = load(sys.argv[1:])
    print(f"\n=== 키워드 ===\n{terms}")
    print(f"\n=== 본문 (앞 500자) ===\n{content[:500]}")
