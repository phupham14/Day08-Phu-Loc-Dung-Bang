"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import html
import json
import re
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc"}
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | {".json"} | TEXT_EXTENSIONS | HTML_EXTENSIONS


def _get_markitdown():
    """Lazy import để module vẫn import được khi môi trường chưa cài MarkItDown."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise ImportError(
            "MarkItDown chưa được cài. Chạy: pip install markitdown"
        ) from exc

    return MarkItDown()


def _output_path_for(filepath: Path) -> Path:
    """Tạo output path trong data/standardized/ và giữ nguyên cấu trúc thư mục."""
    relative_path = filepath.relative_to(LANDING_DIR)
    return (OUTPUT_DIR / relative_path).with_suffix(".md")


def _write_markdown(filepath: Path, content: str) -> Path:
    """Lưu Markdown, tạo thư mục cha nếu cần."""
    output_path = _output_path_for(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  Saved: {output_path}")
    return output_path


def _metadata_header(data: dict) -> str:
    """Tạo metadata header cho bài báo JSON."""
    title = data.get("title") or data.get("headline") or "Unknown"
    url = data.get("url") or data.get("source") or "N/A"
    crawled = data.get("date_crawled") or data.get("crawled_at") or "N/A"

    return (
        f"# {title}\n\n"
        f"**Source:** {url}\n"
        f"**Crawled:** {crawled}\n\n"
        "---\n\n"
    )


def _json_to_markdown(filepath: Path) -> str:
    """Đọc JSON crawler output và chuẩn hoá sang Markdown."""
    data = json.loads(filepath.read_text(encoding="utf-8"))

    if isinstance(data, list):
        sections = []
        for index, item in enumerate(data, 1):
            if not isinstance(item, dict):
                sections.append(str(item))
                continue
            content = (
                item.get("content_markdown")
                or item.get("markdown")
                or item.get("content")
                or item.get("text")
                or ""
            )
            sections.append(f"## Article {index}\n\n{_metadata_header(item)}{content}")
        return "\n\n".join(sections)

    if not isinstance(data, dict):
        return str(data)

    content = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or data.get("text")
        or ""
    )
    return _metadata_header(data) + content


def _html_to_markdown(filepath: Path) -> str:
    """Fallback đơn giản cho HTML nếu raw crawler output là .html."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    content = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", content)
    content = re.sub(r"(?i)<br\s*/?>", "\n", content)
    content = re.sub(r"(?i)</p\s*>", "\n\n", content)
    content = re.sub(r"(?i)</h([1-6])\s*>", "\n\n", content)
    content = re.sub(r"(?i)<h([1-6])[^>]*>", lambda match: "\n" + "#" * int(match.group(1)) + " ", content)
    content = re.sub(r"<[^>]+>", " ", content)
    content = html.unescape(content)
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n\s+\n", "\n\n", content)
    return content.strip()


def convert_file(filepath: Path, md_converter=None) -> Path | None:
    """Convert một file trong data/landing/ sang Markdown."""
    if filepath.name.startswith(".") or not filepath.is_file():
        return None

    suffix = filepath.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        print(f"Skipping unsupported file: {filepath}")
        return None

    print(f"Converting: {filepath.relative_to(LANDING_DIR)}")

    if suffix in DOCUMENT_EXTENSIONS:
        converter = md_converter or _get_markitdown()
        result = converter.convert(str(filepath))
        content = getattr(result, "text_content", "") or ""
    elif suffix == ".json":
        content = _json_to_markdown(filepath)
    elif suffix in TEXT_EXTENSIONS:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    else:
        content = _html_to_markdown(filepath)

    if not content.strip():
        print(f"  Skipped empty output: {filepath}")
        return None

    return _write_markdown(filepath, content)


def iter_landing_files(subdir: str | None = None):
    """Yield supported files dưới data/landing/, có thể giới hạn theo thư mục con."""
    base_dir = LANDING_DIR / subdir if subdir else LANDING_DIR
    if not base_dir.exists():
        return

    for filepath in sorted(base_dir.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield filepath


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    md = _get_markitdown()
    converted = []

    for filepath in iter_landing_files("legal"):
        if filepath.suffix.lower() in DOCUMENT_EXTENSIONS:
            output_path = convert_file(filepath, md)
            if output_path:
                converted.append(output_path)

    print(f"Legal converted: {len(converted)} files")
    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    converted = []

    for filepath in iter_landing_files("news"):
        output_path = convert_file(filepath)
        if output_path:
            converted.append(output_path)

    print(f"News converted: {len(converted)} files")
    return converted


def convert_all_landing_files():
    """Convert mọi file được hỗ trợ trong data/landing/."""
    md = None
    converted = []

    for filepath in iter_landing_files():
        if filepath.suffix.lower() in DOCUMENT_EXTENSIONS:
            md = md or _get_markitdown()
            output_path = convert_file(filepath, md)
        else:
            output_path = convert_file(filepath)

        if output_path:
            converted.append(output_path)

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    converted = convert_all_landing_files()

    print(f"\nDone! Converted {len(converted)} files. Output tai:", OUTPUT_DIR)
    return converted


if __name__ == "__main__":
    convert_all()
