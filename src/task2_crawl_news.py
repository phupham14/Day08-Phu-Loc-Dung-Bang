"""
Task 2 - Crawl bai bao tu cac trang tin tuc Viet Nam.

Huong dan:
    1. Crawl toi thieu 5 bai bao tu cac trang tin tuc Viet Nam.
    2. Su dung Crawl4AI hoac thu vien crawling tuong tu.
    3. Luu output vao data/landing/news/
    4. Moi bai luu 1 file JSON voi metadata (url, title, date_crawled, content).

Cai dat:
    pip install crawl4ai
"""

import asyncio
import html
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


ARTICLE_URLS = [
    "https://tuoitre.vn/nghe-si-viet-anh-binh-luan-sai-vu-huu-tin-choi-ma-tuy-20220614145042023.htm",
    "https://thanhnien.vn/nhieu-nghe-si-ten-tuoi-trung-quoc-bi-phong-sat-vi-dinh-toi-ma-tuy-1851391860.htm",
    "https://tuoitre.vn/bao-han-tiet-lo-ly-do-g-dragon-va-lee-sun-kyun-bi-dieu-tra-ve-ma-tuy-20231123123654534.htm",
    "https://tuoitre.vn/g-dragon-bi-khoi-to-anh-long-nghien-ma-20231026000038474.htm",
    "https://vnexpress.net/dien-vien-phim-tu-than-ky-bi-bat-vi-hut-ma-tuy-1903539.html",
    "https://vnexpress.net/nam-dien-vien-bi-bat-tai-o-lac-2091480.html",
]


def setup_directory() -> None:
    """Tao thu muc output neu chua ton tai."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _extract_title(raw_html: str) -> str:
    og_title = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        raw_html,
        flags=re.IGNORECASE,
    )
    if og_title:
        return _clean_text(og_title.group(1))

    title = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if title:
        return _clean_text(title.group(1))

    return "Unknown"


def _html_to_text(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style|noscript|svg|iframe).*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?i)</(p|div|h[1-6]|li|br|tr)>", "\n", raw_html)
    raw_html = re.sub(r"(?s)<[^>]+>", " ", raw_html)
    return _clean_text(raw_html)


def crawl_article_with_urllib(url: str) -> dict:
    """
    Fallback crawler bang thu vien chuan cua Python.

    Ham nay giup script van tao du lieu khi Crawl4AI chua duoc cai dat
    hoac bi loi Playwright/browser trong moi truong local.
    """
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw_html = response.read().decode(charset, errors="replace")

    title = _extract_title(raw_html)
    content = _html_to_text(raw_html)
    return build_article(url=url, title=title, content=content)


def build_article(url: str, title: str, content: str) -> dict:
    """Chuan hoa metadata cho moi bai bao."""
    return {
        "url": url,
        "title": title or "Unknown",
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content": content,
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl mot bai bao va tra ve dict chua metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content": str,
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            metadata = getattr(result, "metadata", {}) or {}
            markdown = getattr(result, "markdown", "") or ""
            title = metadata.get("title") or "Unknown"
            content = _clean_text(str(markdown))
            if len(content) < 500:
                raise ValueError("Crawl4AI returned too little content")
            return build_article(url=url, title=title, content=content)
    except Exception as exc:
        print(f"  ! Crawl4AI fallback for {url}: {exc}")
        return await asyncio.to_thread(crawl_article_with_urllib, url)


async def crawl_all() -> None:
    """Crawl toan bo bai bao trong ARTICLE_URLS."""
    setup_directory()

    saved_count = 0
    for url in ARTICLE_URLS:
        print(f"[{saved_count + 1}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ! Skipped: {exc}")
            continue

        saved_count += 1
        filename = f"article_{saved_count:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  - Saved: {filepath}")

    if saved_count < 5:
        raise RuntimeError(f"Only crawled {saved_count} articles; need at least 5.")


if __name__ == "__main__":
    asyncio.run(crawl_all())
