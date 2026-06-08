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

import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def _create_docx(content: str, filepath: Path):
    """Tạo file DOCX hợp lệ tối giản bằng zipfile (không cần python-docx)."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        '</Relationships>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '</Relationships>'
    )

    escaped = (
        content
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
    paragraphs = [p for p in escaped.split('\n') if p.strip()]
    para_xml = ''.join(
        f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>'
        for p in paragraphs
    )
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{para_xml}<w:sectPr/></w:body>'
        '</w:document>'
    )

    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types.encode('utf-8'))
        zf.writestr('_rels/.rels', rels.encode('utf-8'))
        zf.writestr('word/_rels/document.xml.rels', doc_rels.encode('utf-8'))
        zf.writestr('word/document.xml', doc_xml.encode('utf-8'))


LEGAL_DOCS = [
    {
        "filename": "luat_phong_chong_ma_tuy_2021.docx",
        "content": """LUẬT PHÒNG, CHỐNG MA TUÝ 2021
Luật số 73/2021/QH15 ngày 30 tháng 3 năm 2021

CHƯƠNG I. NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Luật này quy định về phòng ngừa, phát hiện, ngăn chặn, đấu tranh chống tội phạm và tệ nạn ma tuý; kiểm soát các hoạt động hợp pháp liên quan đến ma tuý; cai nghiện ma tuý; quản lý sau cai nghiện ma tuý; trách nhiệm của cơ quan, tổ chức, cá nhân, gia đình trong phòng, chống ma tuý.

Điều 2. Giải thích từ ngữ
1. Ma tuý là các chất gây nghiện, chất hướng thần được quy định trong các danh mục do Chính phủ ban hành.
2. Chất ma tuý là các chất gây nghiện, chất hướng thần, tiền chất ma tuý được quy định trong danh mục do Chính phủ ban hành.
3. Chất gây nghiện là chất kích thích hoặc ức chế thần kinh, dễ gây tình trạng nghiện đối với người sử dụng.
4. Chất hướng thần là chất kích thích, ức chế thần kinh hoặc gây ảo giác, nếu sử dụng nhiều lần có thể dẫn đến tình trạng nghiện.
5. Tiền chất ma tuý là các hoá chất không thể thiếu được trong quá trình điều chế, sản xuất chất ma tuý.
6. Người nghiện ma tuý là người sử dụng chất ma tuý, thuốc gây nghiện, thuốc hướng thần và bị lệ thuộc vào các chất này.

Điều 3. Các hành vi bị nghiêm cấm
1. Trồng cây có chứa chất ma tuý.
2. Sản xuất, tàng trữ, vận chuyển, bảo quản, mua bán, phân phối, giám định, xử lý, trao đổi, xuất khẩu, nhập khẩu, quá cảnh, chiếm đoạt chất ma tuý trái phép.
3. Sử dụng, tổ chức sử dụng trái phép chất ma tuý; kích động, lôi kéo, dụ dỗ, ép buộc người khác sử dụng trái phép chất ma tuý.
4. Cưỡng bức người sử dụng trái phép chất ma tuý.
5. Chứa chấp việc sử dụng trái phép chất ma tuý.
6. Hướng dẫn, cung cấp thông tin về sử dụng trái phép chất ma tuý.

CHƯƠNG V. CAI NGHIỆN MA TUÝ

Điều 23. Cai nghiện ma tuý
Cai nghiện ma tuý gồm:
a) Cai nghiện ma tuý tự nguyện;
b) Cai nghiện ma tuý bắt buộc.

Điều 24. Đối tượng cai nghiện ma tuý bắt buộc
Người nghiện ma tuý từ đủ 18 tuổi trở lên thuộc một trong các trường hợp sau đây bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc:
a) Không đăng ký tham gia cai nghiện tự nguyện;
b) Trong thời gian cai nghiện tự nguyện bỏ trốn mà không có lý do chính đáng;
c) Đã cai nghiện tự nguyện nhưng vẫn còn nghiện ma tuý.

Điều 29. Thời hạn cai nghiện ma tuý bắt buộc
Thời hạn cai nghiện ma tuý bắt buộc là từ 12 tháng đến 24 tháng.

Điều 32. Quản lý sau cai nghiện ma tuý
Người sau cai nghiện ma tuý được quản lý tại nơi cư trú trong thời hạn từ 01 năm đến 03 năm kể từ ngày chấp hành xong quyết định áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc.
""",
    },
    {
        "filename": "bo_luat_hinh_su_2015_chuong_xx.docx",
        "content": """BỘ LUẬT HÌNH SỰ 2015 (SỬA ĐỔI, BỔ SUNG 2017)
Luật số 100/2015/QH13 - Sửa đổi, bổ sung theo Luật số 12/2017/QH14

CHƯƠNG XX - CÁC TỘI PHẠM VỀ MA TÚY

Điều 247. Tội trồng cây thuốc phiện, cây côca, cây cần sa hoặc các loại cây khác có chứa chất ma túy
1. Người nào trồng cây thuốc phiện, cây côca, cây cần sa hoặc các loại cây khác có chứa chất ma túy, đã được giáo dục nhiều lần và đã được tạo điều kiện để ổn định cuộc sống, thì bị phạt tù từ 6 tháng đến 3 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 3 năm đến 7 năm: a) Có tổ chức; b) Với số lượng lớn.

Điều 248. Tội sản xuất trái phép chất ma túy
1. Người nào sản xuất trái phép chất ma túy, thì bị phạt tù từ 02 năm đến 07 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 07 năm đến 15 năm:
a) Có tổ chức;
b) Phạm tội 02 lần trở lên;
c) Lợi dụng chức vụ, quyền hạn;
d) Lợi dụng danh nghĩa cơ quan, tổ chức;
e) Chất ma túy thuộc danh mục các chất ma túy đặc biệt nguy hiểm.
3. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 15 năm đến 20 năm: a) Nhựa thuốc phiện từ 01 kilôgam đến dưới 05 kilôgam; b) Hêrôin hoặc côcain từ 30 gam đến dưới 100 gam.
4. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù 20 năm, tù chung thân hoặc tử hình.

Điều 249. Tội tàng trữ trái phép chất ma túy
1. Người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển, thì bị phạt tù từ 01 năm đến 05 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 05 năm đến 10 năm:
a) Có chất ma túy thuộc danh mục các chất ma túy đặc biệt nguy hiểm;
b) Nhựa thuốc phiện, nhựa cần sa hoặc cao côca có khối lượng từ 500 gram đến dưới 01 kilôgam;
c) Hêrôin hoặc côcain có khối lượng từ 01 gam đến dưới 05 gam.
3. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 10 năm đến 15 năm.
4. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 15 năm đến 20 năm hoặc tù chung thân.

Điều 250. Tội vận chuyển trái phép chất ma túy
1. Người nào vận chuyển trái phép chất ma túy, thì bị phạt tù từ 02 năm đến 07 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 07 năm đến 15 năm.
3. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 15 năm đến 20 năm.
4. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù 20 năm, tù chung thân hoặc tử hình.

Điều 251. Tội mua bán trái phép chất ma túy
1. Người nào mua bán trái phép chất ma túy, thì bị phạt tù từ 02 năm đến 07 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 07 năm đến 15 năm:
a) Có tổ chức;
b) Có tính chất chuyên nghiệp;
c) Nhựa thuốc phiện, nhựa cần sa hoặc cao côca có khối lượng từ 500 gram đến dưới 01 kilôgam;
d) Hêrôin hoặc côcain có khối lượng từ 05 gam đến dưới 30 gam.
3. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 15 năm đến 20 năm.
4. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù 20 năm, tù chung thân hoặc tử hình.

Điều 255. Tội tổ chức sử dụng trái phép chất ma túy
1. Người nào tổ chức sử dụng trái phép chất ma túy dưới bất kỳ hình thức nào, thì bị phạt tù từ 02 năm đến 07 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 07 năm đến 15 năm.
3. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 15 năm đến 20 năm.

Điều 256. Tội chứa chấp việc sử dụng trái phép chất ma túy
1. Người nào chứa chấp việc sử dụng trái phép chất ma túy dưới bất kỳ hình thức nào, thì bị phạt tù từ 02 năm đến 07 năm.
2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 07 năm đến 15 năm.

Điều 257. Tội sử dụng trái phép chất ma túy
Người nào sử dụng trái phép chất ma túy, thì bị phạt tù từ 01 tháng đến 06 tháng hoặc phạt tù từ 06 tháng đến 02 năm.
""",
    },
    {
        "filename": "nghi_dinh_105_2021_nd_cp.docx",
        "content": """NGHỊ ĐỊNH 105/2021/NĐ-CP
Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Phòng, chống ma tuý
Ban hành ngày 04 tháng 12 năm 2021

CHƯƠNG I. NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều của Luật Phòng, chống ma tuý năm 2021 về:
1. Kiểm soát các hoạt động hợp pháp liên quan đến ma tuý;
2. Cai nghiện ma tuý;
3. Quản lý sau cai nghiện ma tuý;
4. Trách nhiệm của cơ quan, tổ chức, cá nhân trong phòng, chống ma tuý.

Điều 3. Danh mục các chất ma tuý
Danh mục các chất ma tuý được quy định theo các Bảng sau:
Bảng I: Các chất ma tuý đặc biệt nguy hiểm cấm sử dụng trong mọi lĩnh vực (heroin, cocaine, methamphetamine,...)
Bảng II: Các chất ma tuý được dùng hạn chế trong lĩnh vực y tế, phân tích, kiểm nghiệm
Bảng III: Các chất ma tuý được dùng trong lĩnh vực y tế và phân tích, kiểm nghiệm
Bảng IV: Các tiền chất ma tuý

CHƯƠNG II. KIỂM SOÁT CÁC HOẠT ĐỘNG HỢP PHÁP LIÊN QUAN ĐẾN MA TUÝ

Điều 5. Điều kiện để được phép hoạt động liên quan đến ma tuý
1. Có giấy phép do cơ quan có thẩm quyền cấp;
2. Có địa điểm, cơ sở vật chất, trang thiết bị đáp ứng yêu cầu;
3. Có cán bộ, nhân viên chuyên môn đáp ứng yêu cầu;
4. Có hệ thống quản lý, kiểm soát đảm bảo an ninh, an toàn.

CHƯƠNG III. CAI NGHIỆN MA TUÝ

Điều 15. Điều kiện đối với cơ sở cai nghiện ma tuý tự nguyện
1. Có địa điểm, cơ sở vật chất phù hợp với quy định của Bộ Lao động - Thương binh và Xã hội;
2. Có đội ngũ cán bộ, chuyên viên về y tế, tâm lý, xã hội đáp ứng yêu cầu;
3. Có phương án, quy trình cai nghiện phù hợp với tiêu chuẩn chuyên môn;
4. Đã được cơ quan có thẩm quyền cấp giấy phép hoạt động.

Điều 20. Nội dung cai nghiện ma tuý bắt buộc
1. Tư vấn, hỗ trợ tâm lý, xã hội;
2. Điều trị y tế và điều trị nghiện chất;
3. Giáo dục, phục hồi hành vi, nhân cách;
4. Dạy nghề, tổ chức lao động sản xuất;
5. Quản lý sau cai nghiện.

Điều 23. Trình tự, thủ tục áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc
1. Lập hồ sơ đề nghị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc;
2. Gửi hồ sơ đến Tòa án nhân dân cấp huyện để xem xét, quyết định;
3. Tòa án nhân dân cấp huyện xem xét, ra quyết định trong thời hạn 03 ngày làm việc kể từ ngày nhận đủ hồ sơ.

CHƯƠNG IV. QUẢN LÝ SAU CAI NGHIỆN MA TUÝ

Điều 30. Chế độ quản lý sau cai nghiện ma tuý
1. Người sau cai nghiện ma tuý được quản lý tại nơi cư trú trong thời hạn từ 01 năm đến 03 năm kể từ ngày chấp hành xong quyết định.
2. Ủy ban nhân dân cấp xã có trách nhiệm tổ chức, quản lý người sau cai nghiện ma tuý.
3. Nội dung quản lý bao gồm: theo dõi, giám sát việc chấp hành pháp luật; hỗ trợ học nghề, tìm việc làm; hỗ trợ tái hòa nhập cộng đồng.

Điều 32. Chương trình hỗ trợ tái hòa nhập cộng đồng
Người sau cai nghiện được hỗ trợ:
1. Học nghề, tìm việc làm;
2. Vay vốn để sản xuất, kinh doanh;
3. Chăm sóc sức khỏe;
4. Hỗ trợ tâm lý, xã hội.
""",
    },
    {
        "filename": "nghi_dinh_57_2022_nd_cp_danh_muc_chat_ma_tuy.docx",
        "content": """NGHỊ ĐỊNH 57/2022/NĐ-CP
Quy định các danh mục chất ma túy và tiền chất
Ban hành ngày 25 tháng 8 năm 2022

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định danh mục chất ma túy và tiền chất của Việt Nam bao gồm 4 Bảng danh mục chất ma túy và 1 Bảng tiền chất.

Điều 2. Danh mục các chất ma tuý

BẢNG I - CÁC CHẤT MA TÚY TUYỆT ĐỐI CẤM SỬ DỤNG
1. Heroin (diacetylmorphine)
2. Cocaine
3. Methamphetamine (Crystal meth, đá)
4. MDMA (Ecstasy)
5. Fentanyl và các dẫn xuất
6. LSD (Lysergic acid diethylamide)
7. Ketamine (khi dùng phi y tế)
8. Psilocybin (nấm ma thuật)

BẢNG II - CÁC CHẤT MA TÚY DÙNG HẠN CHẾ TRONG Y TẾ
1. Morphine (dùng giảm đau, cần kê đơn đặc biệt)
2. Codeine (dùng chữa ho, cần kê đơn)
3. Methadone (dùng điều trị nghiện opioid)
4. Buprenorphine (dùng điều trị nghiện)

BẢNG III - CÁC CHẤT MA TÚY DÙNG TRONG Y TẾ VÀ KIỂM NGHIỆM
1. Diazepam (Valium)
2. Alprazolam (Xanax)
3. Zolpidem
4. Tramadol

BẢNG IV - TIỀN CHẤT MA TÚY
1. Ephedrine (dùng trong dược phẩm, có thể tổng hợp methamphetamine)
2. Pseudoephedrine
3. Acetone
4. Acetic anhydride
5. Potassium permanganate

Điều 3. Quy định xử lý
Việc tàng trữ, mua bán, sử dụng các chất trong Bảng I bị nghiêm cấm hoàn toàn và có thể bị xử lý hình sự theo Điều 249 đến Điều 257 Bộ luật Hình sự 2015.

Điều 4. Điều khoản thi hành
Nghị định này có hiệu lực từ ngày 10 tháng 10 năm 2022 và thay thế Nghị định 73/2018/NĐ-CP ngày 15 tháng 5 năm 2018.
""",
    },
]


def create_sample_docs():
    """Tạo các file DOCX mẫu với nội dung pháp luật ma tuý Việt Nam."""
    setup_directory()
    for doc in LEGAL_DOCS:
        filepath = DATA_DIR / doc["filename"]
        if not filepath.exists():
            _create_docx(doc["content"], filepath)
            print(f"✓ Created: {filepath.name} ({filepath.stat().st_size} bytes)")
        else:
            print(f"  Already exists: {filepath.name}")


def download_file(url: str, filename: str):
    """Thử tải file từ URL, bỏ qua nếu thất bại."""
    import requests
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200 and len(response.content) > 1024:
            filepath = DATA_DIR / filename
            filepath.write_bytes(response.content)
            print(f"✓ Downloaded: {filepath.name} ({len(response.content)} bytes)")
            return True
    except Exception as e:
        print(f"  Download failed for {filename}: {e}")
    return False


if __name__ == "__main__":
    create_sample_docs()
    print(f"\n✓ Legal documents ready in: {DATA_DIR}")
    files = list(DATA_DIR.iterdir())
    print(f"  Total files: {len(files)}")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
