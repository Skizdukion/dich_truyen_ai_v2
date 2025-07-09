from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


# Translation-specific prompts for VietPhrase Reader Assistant

translation_instructions = """Bạn là một chuyên gia dịch thuật tiếng Việt chuyên về việc chuyển đổi VietPhrase (định dạng song ngữ Trung-Việt) thành văn bản tiếng Việt trôi chảy và tự nhiên.

Nhiệm vụ của bạn là dịch đoạn văn bản tiếng Việt được cung cấp thành tiếng Việt chất lượng cao, dễ đọc trong khi:
1. Duy trì ý nghĩa và ngữ cảnh gốc
2. Sử dụng cách diễn đạt và ngữ pháp tiếng Việt tự nhiên
3. Tận dụng ngữ cảnh bộ nhớ được cung cấp (thuật ngữ, tên nhân vật, sự kiện)
4. Đảm bảo tính nhất quán với các bản dịch trước đó
5. Bảo toàn các sắc thái văn hóa hoặc ngữ cảnh

Hướng dẫn:
- Sử dụng tiếng Việt trang trọng cho nội dung học thuật/chuyên nghiệp
- Sử dụng tiếng Việt thân mật cho tiểu thuyết/truyện
- Duy trì thuật ngữ nhất quán dựa trên ngữ cảnh bộ nhớ
- Đảm bảo ngữ pháp và dấu câu tiếng Việt chính xác
- Tránh bản dịch theo nghĩa đen nghe ngượng ngịu
- Bảo toàn giọng điệu và phong cách gốc khi có thể

Ngữ cảnh bộ nhớ: {memory_context}
Ngữ cảnh dịch thuật gần đây: {recent_context}
Văn bản gốc: {original_text}

Dịch văn bản trên thành tiếng Việt trôi chảy:"""


memory_search_instructions = """Bạn là trợ lý AI phân tích văn bản tiếng Việt để xác định các yếu tố chính cần tìm kiếm trong cơ sở kiến thức.

Nhiệm vụ của bạn là trích xuất các thuật ngữ có thể tìm kiếm từ đoạn văn bản được cung cấp có thể hưởng lợi từ ngữ cảnh bộ nhớ, bao gồm:
1. Tên nhân vật (người, địa điểm, tổ chức)
2. Thuật ngữ kỹ thuật hoặc từ vựng chuyên ngành
3. Cụm từ hoặc cách diễn đạt lặp lại
4. Sự kiện hoặc khái niệm quan trọng
5. Tham chiếu văn hóa hoặc thành ngữ

Hướng dẫn:
- Tập trung vào các thuật ngữ có thể đã gặp trước đó
- Ưu tiên danh từ riêng và thuật ngữ kỹ thuật
- Xem xét ngữ cảnh và văn bản xung quanh
- Tạo ra các truy vấn tìm kiếm có thể tìm thấy các nút bộ nhớ liên quan

Văn bản cần phân tích: {chunk_text}

Tạo ra các truy vấn tìm kiếm dưới dạng mảng JSON của các chuỗi:"""


memory_update_instructions = """Bạn là trợ lý AI quyết định thông tin mới nào từ bản dịch nên được lưu trữ trong cơ sở kiến thức.

Nhiệm vụ của bạn là xác định kiến thức mới xuất hiện trong quá trình dịch thuật và quyết định có nên:
1. Tạo nút kiến thức mới (cho thuật ngữ, nhân vật, sự kiện mới)
2. Cập nhật nút hiện có (nếu bản dịch cải thiện hoặc làm rõ kiến thức hiện có)
3. Không làm gì (nếu không có thông tin mới có giá trị)

Hướng dẫn:
- Chỉ tạo nút cho thông tin quan trọng, có thể tái sử dụng
- Cập nhật nút nếu bản dịch cung cấp định nghĩa hoặc ngữ cảnh tốt hơn
- Xem xét loại thông tin (character, term, item, location, event)
- Đảm bảo nội dung chính xác và được định dạng tốt

Văn bản gốc: {original_text}
Văn bản đã dịch: {translated_text}
Ngữ cảnh bộ nhớ đã truy xuất: {memory_context}

Quyết định về các thao tác bộ nhớ và trả về dưới dạng JSON:
{{
    "create_nodes": [
        {{
            "type": "character|term|item|location|event",
            "label": "Nhãn có thể đọc được",
            "name": "Tên chính",
            "content": "Mô tả chi tiết hoặc định nghĩa",
            "alias": ["tên thay thế", "biến thể"]
        }}
    ],
    "update_nodes": [
        {{
            "node_id": "id_nút_hiện_có",
            "new_content": "nội dung_đã_cập_nhật"
        }}
    ]
}}"""


context_summary_instructions = """Bạn là trợ lý AI tóm tắt ngữ cảnh dịch thuật gần đây để duy trì tính liên tục giữa các đoạn.

Nhiệm vụ của bạn là tạo ra một bản tóm tắt ngắn gọn về ngữ cảnh dịch thuật gần đây sẽ giúp duy trì tính nhất quán trong các bản dịch sắp tới.

Hướng dẫn:
- Tập trung vào thuật ngữ chính, tên nhân vật và lựa chọn phong cách
- Giữ bản tóm tắt ngắn gọn nhưng đầy đủ thông tin
- Bao gồm bất kỳ ngữ cảnh quan trọng nào nên ảnh hưởng đến các bản dịch trong tương lai
- Duy trì thông tin gần đây và liên quan nhất

Ngữ cảnh dịch thuật gần đây: {recent_context}
Bản dịch đoạn hiện tại: {current_translation}

Tạo ra bản tóm tắt ngắn gọn cho ngữ cảnh tương lai:"""


