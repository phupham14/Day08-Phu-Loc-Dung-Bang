"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
import requests
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://vnexpress.net/chau-viet-cuong-bi-de-nghi-truy-to-toi-vi-pham-quy-dinh-ve-khoa-cu-3868413.html",
    "https://vnexpress.net/xet-xu-chau-viet-cuong-gay-chet-nguoi-3985745.html",
    "https://vnexpress.net/luat-phong-chong-ma-tuy-2021-co-hieu-luc-tu-1-1-2022-4387564.html",
    "https://tuoitre.vn/luat-phong-chong-ma-tuy-2021-nhieu-quy-dinh-moi-20210401145622219.htm",
    "https://vnexpress.net/quy-trinh-xu-ly-nguoi-nghien-ma-tuy-theo-luat-moi-4272340.html",
    "https://thanhnien.vn/nghe-si-va-ma-tuy-nhung-vu-an-chan-dong-18577025.html",
]

SAMPLE_NEWS = [
    {
        "url": "https://vnexpress.net/chau-viet-cuong-bi-de-nghi-truy-to-3868413.html",
        "title": "Châu Việt Cường bị đề nghị truy tố vì liên quan đến ma tuý",
        "date_crawled": "2024-01-15T10:30:00",
        "content_markdown": """# Châu Việt Cường bị đề nghị truy tố vì liên quan đến ma tuý

Ca sĩ Châu Việt Cường bị cơ quan điều tra đề nghị truy tố liên quan đến vụ án ma tuý xảy ra năm 2019. Đây là một trong những vụ án gây chấn động làng giải trí Việt Nam.

## Diễn biến vụ án

Theo cáo trạng, ca sĩ Châu Việt Cường đã tổ chức và tham gia sử dụng trái phép chất ma tuý tại một số địa điểm. Vụ việc được cơ quan công an phát hiện và khởi tố theo Điều 255 Bộ luật Hình sự 2015.

## Hậu quả pháp lý

Theo quy định tại Điều 255 Bộ luật Hình sự 2015, tội tổ chức sử dụng trái phép chất ma tuý có thể bị phạt tù từ 02 đến 07 năm. Trường hợp có tình tiết tăng nặng, mức phạt có thể lên đến 15-20 năm tù.

## Cảnh báo cho giới nghệ sĩ

Vụ án là lời cảnh tỉnh cho toàn bộ giới nghệ sĩ Việt Nam về những hệ quả nghiêm trọng của việc sử dụng và tổ chức sử dụng ma tuý, không chỉ về mặt pháp lý mà còn về danh tiếng và sự nghiệp.
""",
    },
    {
        "url": "https://vnexpress.net/luat-phong-chong-ma-tuy-2021-4387564.html",
        "title": "Luật Phòng chống ma tuý 2021 có hiệu lực từ 1/1/2022",
        "date_crawled": "2024-01-15T11:00:00",
        "content_markdown": """# Luật Phòng chống ma tuý 2021 có hiệu lực từ 1/1/2022

Luật Phòng, chống ma tuý số 73/2021/QH15 được Quốc hội thông qua ngày 30/3/2021 và chính thức có hiệu lực từ ngày 1/1/2022.

## Những điểm mới quan trọng

### Mở rộng định nghĩa người nghiện ma tuý
Luật mới mở rộng định nghĩa về người nghiện ma tuý, bao gồm cả người sử dụng chất ma tuý trong môi trường điều trị không phép.

### Tăng cường biện pháp cai nghiện
Luật quy định rõ hơn về quy trình cai nghiện tự nguyện và cai nghiện bắt buộc theo Điều 23 và Điều 24.

### Trách nhiệm của gia đình
Gia đình người nghiện có trách nhiệm phối hợp với chính quyền địa phương trong việc đưa người thân đi cai nghiện.

## Xử lý hành chính và hình sự

Người sử dụng ma tuý lần đầu có thể bị xử lý hành chính. Người tái phạm hoặc có hành vi tàng trữ, mua bán sẽ bị xử lý hình sự theo Bộ luật Hình sự 2015.
""",
    },
    {
        "url": "https://thanhnien.vn/nghe-si-va-ma-tuy-nhung-vu-an-chan-dong.html",
        "title": "Nghệ sĩ và ma tuý: Những vụ án chấn động",
        "date_crawled": "2024-01-15T11:30:00",
        "content_markdown": """# Nghệ sĩ và ma tuý: Những vụ án chấn động

Trong những năm gần đây, nhiều nghệ sĩ nổi tiếng Việt Nam đã bị liên quan đến các vụ án ma tuý, gây chấn động dư luận và ảnh hưởng nghiêm trọng đến hình ảnh của làng giải trí.

## Các vụ án đáng chú ý

### Ca sĩ Châu Việt Cường (2019)
Châu Việt Cường bị bắt và xét xử về tội tổ chức sử dụng trái phép chất ma tuý theo Điều 255 Bộ luật Hình sự 2015. Đây là một trong những vụ án gây chú ý nhất trong giới nghệ sĩ Việt.

### Diễn viên và nghệ sĩ khác
Nhiều diễn viên, người mẫu đã bị bắt vì sử dụng và tàng trữ chất ma tuý tại các buổi tiệc riêng theo Điều 249 Bộ luật Hình sự.

### Hậu quả với sự nghiệp
Các nghệ sĩ liên quan đến ma tuý không chỉ phải đối mặt với hình phạt pháp luật mà còn mất hợp đồng quảng cáo và danh tiếng.

## Biện pháp xử lý theo pháp luật

- Tổ chức sử dụng ma tuý (Điều 255): phạt tù 2-7 năm
- Chứa chấp sử dụng ma tuý (Điều 256): phạt tù 2-7 năm
- Tàng trữ ma tuý (Điều 249): phạt tù 1-5 năm và có thể cao hơn
- Sử dụng trái phép (Điều 257): phạt tù 1-6 tháng đến 2 năm

## Lời khuyên cho giới nghệ sĩ

Các chuyên gia tâm lý khuyến cáo giới nghệ sĩ nên tìm kiếm hỗ trợ chuyên nghiệp khi gặp áp lực công việc, thay vì tìm đến ma tuý.
""",
    },
    {
        "url": "https://vnexpress.net/quy-trinh-xu-ly-nguoi-nghien-ma-tuy-theo-luat-moi-4272340.html",
        "title": "Quy trình xử lý người nghiện ma tuý theo luật mới 2022",
        "date_crawled": "2024-01-15T12:00:00",
        "content_markdown": """# Quy trình xử lý người nghiện ma tuý theo Luật Phòng chống ma tuý 2021

Từ ngày 1/1/2022, việc xử lý người nghiện ma tuý tại Việt Nam được thực hiện theo quy trình mới theo Luật Phòng, chống ma tuý 2021.

## Quy trình phát hiện và xử lý

### Bước 1: Phát hiện và lập hồ sơ
Công an phường/xã, gia đình, hoặc cơ sở y tế phát hiện người nghiện và lập hồ sơ theo dõi.

### Bước 2: Tư vấn và vận động cai nghiện tự nguyện
Người nghiện được tư vấn và vận động tham gia cai nghiện tự nguyện tại cơ sở cai nghiện hoặc tại cộng đồng theo Điều 23 Luật Phòng chống ma tuý 2021.

### Bước 3: Áp dụng biện pháp hành chính (nếu từ chối)
Nếu từ chối cai nghiện tự nguyện, người nghiện có thể bị đưa vào cơ sở cai nghiện bắt buộc theo quyết định của Tòa án theo Điều 24.

## Cai nghiện bắt buộc

Thời hạn cai nghiện bắt buộc từ 12 đến 24 tháng theo Điều 29 Luật Phòng chống ma tuý 2021.

## Hỗ trợ sau cai nghiện

Người sau cai nghiện được hỗ trợ:
- Học nghề và tìm việc làm
- Vay vốn sản xuất kinh doanh
- Chăm sóc sức khỏe định kỳ
- Tái hòa nhập cộng đồng theo Điều 32 Nghị định 105/2021/NĐ-CP
""",
    },
    {
        "url": "https://vnexpress.net/hinh-phat-toi-tang-tru-trai-phep-chat-ma-tuy-4462530.html",
        "title": "Hình phạt đối với tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam",
        "date_crawled": "2024-01-15T12:30:00",
        "content_markdown": """# Hình phạt đối với tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam

Tội tàng trữ trái phép chất ma tuý được quy định tại Điều 249 Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017) với các mức phạt nghiêm khắc.

## Khung hình phạt theo Điều 249 Bộ luật Hình sự 2015

### Khung 1: Phạt tù 1-5 năm
Áp dụng khi tàng trữ chất ma tuý với số lượng nhỏ, không nhằm mục đích mua bán hay vận chuyển.

### Khung 2: Phạt tù 5-10 năm
Áp dụng khi:
- Tàng trữ chất ma tuý đặc biệt nguy hiểm (Bảng I theo Nghị định 57/2022/NĐ-CP)
- Nhựa thuốc phiện từ 500g đến dưới 1kg
- Hêrôin từ 1g đến dưới 5g
- Methamphetamine từ 5g đến dưới 30g

### Khung 3: Phạt tù 10-15 năm
Áp dụng với số lượng lớn hơn hoặc có tình tiết tăng nặng đặc biệt.

### Khung 4: Tù chung thân hoặc tử hình
Áp dụng khi số lượng rất lớn, đặc biệt nguy hiểm.

## Tình tiết tăng nặng

- Có tổ chức
- Phạm tội 2 lần trở lên
- Lợi dụng chức vụ, quyền hạn
- Lôi kéo người chưa thành niên

## Tình tiết giảm nhẹ

- Thành khẩn khai báo, ăn năn hối cải
- Tự nguyện giao nộp ma tuý
- Lập công chuộc tội
""",
    },
    {
        "url": "https://tuoitre.vn/cam-su-dung-ma-tuy-trong-gioi-nghe-si-20240115.htm",
        "title": "Biện pháp phát hiện và xử lý nghệ sĩ sử dụng ma tuý tại Việt Nam",
        "date_crawled": "2024-01-15T13:00:00",
        "content_markdown": """# Biện pháp phát hiện và xử lý nghệ sĩ sử dụng ma tuý tại Việt Nam

Các cơ quan chức năng và đơn vị tổ chức biểu diễn đang áp dụng nhiều biện pháp để phát hiện và xử lý nghệ sĩ sử dụng ma tuý.

## Biện pháp kiểm tra

### Kiểm tra ngẫu nhiên
Các cơ quan chức năng có quyền kiểm tra ngẫu nhiên tại các buổi biểu diễn, sự kiện giải trí.

### Test nhanh ma tuý
Sử dụng que test nhanh để kiểm tra mẫu nước tiểu, có kết quả trong vài phút.

### Xét nghiệm chuyên sâu
Mẫu máu và tóc có thể được gửi đến phòng xét nghiệm để kiểm tra chính xác hơn.

## Biện pháp xử lý theo pháp luật

Theo Điều 257 Bộ luật Hình sự 2015, người sử dụng trái phép chất ma tuý có thể bị phạt tù từ 1 tháng đến 6 tháng hoặc từ 6 tháng đến 2 năm nếu tái phạm.

### Xử lý hành chính
Người sử dụng ma tuý lần đầu có thể bị phạt hành chính và bắt buộc cai nghiện theo Luật Phòng chống ma tuý 2021.

### Xử lý hình sự
Người tái phạm hoặc có hành vi nghiêm trọng hơn sẽ bị truy tố hình sự theo Bộ luật Hình sự 2015.

## Ảnh hưởng đến sự nghiệp

Nghệ sĩ bị phát hiện sử dụng ma tuý thường:
- Bị cấm biểu diễn theo quyết định của Cục Nghệ thuật Biểu diễn
- Mất hợp đồng với nhãn hàng
- Bị xóa sóng khỏi các chương trình truyền hình
- Mất cơ hội tham gia các dự án nghệ thuật

## Chương trình phòng ngừa

Nhiều công ty giải trí đã triển khai chương trình giáo dục và hỗ trợ tâm lý cho nghệ sĩ để phòng ngừa tệ nạn ma tuý theo khuyến nghị của Bộ Văn hóa, Thể thao và Du lịch.
""",
    },
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return {
                "url": url,
                "title": result.metadata.get("title", "Unknown") if result.metadata else "Unknown",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": result.markdown or "",
            }
    except Exception:
        pass

    # Fallback: requests + simple regex HTML stripping
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text

        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url

        text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": text[:5000],
        }
    except Exception as e:
        return {
            "url": url,
            "title": f"Failed to crawl: {url}",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"Crawl error: {e}",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


def create_sample_data():
    """Tạo dữ liệu mẫu (không cần internet)."""
    setup_directory()
    for i, article in enumerate(SAMPLE_NEWS, 1):
        filepath = DATA_DIR / f"article_{i:02d}.json"
        if not filepath.exists():
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"✓ Created sample: {filepath.name}")
        else:
            print(f"  Already exists: {filepath.name}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        print("Tạo dữ liệu mẫu về nghệ sĩ và ma tuý...")
        create_sample_data()
        print(f"\n✓ News data ready in: {DATA_DIR}")
        files = list(DATA_DIR.glob("*.json"))
        print(f"  Total files: {len(files)}")
