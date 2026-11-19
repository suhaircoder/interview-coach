from pathlib import Path

import pytest

from src.resume import MAX_RESUME_CHARS, ResumeError, extract_resume_text


def test_no_resume_returns_empty_text() -> None:
    assert extract_resume_text(None) == ""


def test_text_resume_is_cleaned(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("  Python   Developer  \n\n Built APIs  ", encoding="utf-8")

    assert extract_resume_text(str(resume)) == "Python Developer\nBuilt APIs"


def test_resume_text_is_limited(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("a" * (MAX_RESUME_CHARS + 100), encoding="utf-8")

    assert len(extract_resume_text(str(resume))) == MAX_RESUME_CHARS


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    resume = tmp_path / "resume.exe"
    resume.write_text("not a resume", encoding="utf-8")

    with pytest.raises(ResumeError, match="PDF, DOCX, or TXT"):
        extract_resume_text(str(resume))


def test_empty_resume_is_rejected(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("  \n", encoding="utf-8")

    with pytest.raises(ResumeError, match="No readable text"):
        extract_resume_text(str(resume))
