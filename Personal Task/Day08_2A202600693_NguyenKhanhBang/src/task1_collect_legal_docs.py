"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path
from urllib.parse import urlparse

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str | None = None) -> Path:
    """Tải file PDF/DOCX từ direct link và lưu vào DATA_DIR."""
    setup_directory()
    parsed_name = Path(urlparse(url).path).name
    output_name = filename or parsed_name

    if not output_name:
        raise ValueError("Không xác định được tên file. Hãy truyền filename.")

    filepath = DATA_DIR / output_name
    suffix = filepath.suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ValueError("Chỉ hỗ trợ tải file PDF hoặc DOCX.")

    if filepath.exists() and filepath.stat().st_size > 0:
        print(f"✓ File đã tồn tại: {filepath}")
        return filepath

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with filepath.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print(f"✓ Đã tải: {filepath}")
    return filepath


if __name__ == "__main__":
    download_file("https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf", "luat-phong-chong-ma-tuy-2021.pdf")
    download_file("https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf", "nghi-dinh-105-2021.pdf")
    download_file("https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf", "bo-luat-hinh-su-2015.pdf")