from pathlib import Path

from docx import Document
from pypdf import PdfReader

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_RESUME_CHARS = 10_000
MAX_PDF_PAGES = 50
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class ResumeError(ValueError):
    """Raised when a résumé cannot be safely read."""


def extract_resume_text(file_path: str | None) -> str:
    if not file_path:
        return ""

    path = Path(file_path)
    if not path.is_file():
        raise ResumeError("The uploaded CV is no longer available. Upload it again.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ResumeError("Upload a PDF, DOCX, or TXT file.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ResumeError("The CV must be smaller than 5 MB.")

    try:
        text = _read_file(path)
    except ResumeError:
        raise
    except Exception as error:
        raise ResumeError("The CV could not be read. Try another file.") from error

    cleaned_lines = (" ".join(line.split()) for line in text.splitlines())
    cleaned = "\n".join(line for line in cleaned_lines if line)
    if not cleaned:
        raise ResumeError(
            "No readable text was found. Scanned PDFs need OCR before upload."
        )
    return cleaned[:MAX_RESUME_CHARS]


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ResumeError(f"The CV must have at most {MAX_PDF_PAGES} pages.")
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    raise ResumeError("Unsupported CV file type.")
