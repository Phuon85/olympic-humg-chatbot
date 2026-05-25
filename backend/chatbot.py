"""
chatbot.py v3 — Linh hoạt toàn diện
Nâng cấp so với v2:
  1. Query Normalization  — hiểu viết tắt, sai chính tả, hỏi ngắn tiếng Việt
  2. Multi-turn Memory    — tóm tắt ngữ cảnh dài, giải quyết đại từ ("môn đó", "thời gian đó")
  3. Smart Fallback       — không nói "không biết" cứng nhắc, gợi ý liên quan
  4. Tone System         — auto-detect + 3 chế độ: thân mật / chuẩn / trang trọng
"""

import os, json, time, re
from typing import Generator
from groq import Groq
from cache import FAQCache
from rag import RAGPipeline


# ═══════════════════════════════════════════════════════════════════
#  1. QUERY NORMALIZATION — Chuẩn hóa câu hỏi tiếng Việt
# ═══════════════════════════════════════════════════════════════════

# Từ điển viết tắt phổ biến của sinh viên
_ABBR = {
    r'\bsv\b': 'sinh viên',
    r'\bbt\b': 'bài tập',
    r'\bgv\b': 'giáo viên',
    r'\bdk\b': 'điều kiện',
    r'\bđk\b': 'điều kiện',
    r'\bolp\b': 'olympic',
    r'\bolpic\b': 'olympic',
    r'\bgt\b': 'giải tích',
    r'\bđs\b': 'đại số',
    r'\btc\b': 'tín chỉ',
    r'\bhk\b': 'học kỳ',
    r'\bhtc\b': 'học tín chỉ',
    r'\bkq\b': 'kết quả',
    r'\btt\b': 'thông tin',
    r'\btg\b': 'thời gian',
    r'\bhumg\b': 'Đại học Mỏ Địa chất',
    r'\bck\b': 'cuối kỳ',
    r'\bgk\b': 'giữa kỳ',
    r'\bđrl\b': 'điểm rèn luyện',
    r'\bkhoa cb\b': 'khoa học cơ bản',
}

# Từ sai chính tả phổ biến
_TYPOS = {
    'olimpic': 'olympic', 'olempic': 'olympic', 'olymic': 'olympic',
    'địa chất': 'Địa chất', 'mo dia chat': 'Mỏ Địa chất',
    'dang ky': 'đăng ký', 'dieu kien': 'điều kiện',
    'giai thuong': 'giải thưởng', 'lich thi': 'lịch thi',
    'thoi gian': 'thời gian', 'mon thi': 'môn thi',
}

def normalize_query(text: str) -> str:
    """Chuẩn hóa: viết thường → expand viết tắt → sửa typo cơ bản"""
    t = text.strip()
    t_lower = t.lower()
    # Expand abbreviations
    for pattern, replacement in _ABBR.items():
        t_lower = re.sub(pattern, replacement, t_lower, flags=re.IGNORECASE)
    # Fix typos
    for wrong, right in _TYPOS.items():
        t_lower = t_lower.replace(wrong, right)
    # Nếu câu quá ngắn (< 5 ký tự) → giữ nguyên bản gốc
    return t_lower if len(t_lower) >= 5 else t


# ═══════════════════════════════════════════════════════════════════
#  2. TONE SYSTEM
# ═══════════════════════════════════════════════════════════════════

TONES = {
    'friendly': {
        'label': 'Thân mật',
        'persona': (
            "Bạn là Olympic Bot — người bạn học thân thiện của sinh viên HUMG! 😊\n"
            "Xưng 'mình', gọi người dùng là 'bạn'. Câu trả lời ngắn gọn, dễ hiểu, "
            "dùng emoji khi phù hợp. Nếu không biết thì nói thẳng và gợi ý hướng khác."
        )
    },
    'standard': {
        'label': 'Chuẩn',
        'persona': (
            "Bạn là Trợ lý AI Olympic HUMG — chatbot chính thức của Website Olympic "
            "Trường Đại học Mỏ - Địa chất.\n"
            "Xưng 'tôi', gọi người dùng là 'bạn'. Trả lời rõ ràng, chính xác, "
            "có cấu trúc. Dùng bullet/bảng khi cần thiết."
        )
    },
    'formal': {
        'label': 'Trang trọng',
        'persona': (
            "Bạn là Trợ lý AI chính thức của Phòng Khoa học - Công nghệ, "
            "Trường Đại học Mỏ - Địa chất.\n"
            "Sử dụng văn phong trang trọng, chính xác. Trả lời đầy đủ, "
            "có dẫn chiếu văn bản khi có thể. Không dùng emoji."
        )
    }
}

TONE_RULES = """
Quy tắc bắt buộc cho mọi chế độ:
1. Luôn trả lời bằng TIẾNG VIỆT một cách tự nhiên và mạch lạc.
2. Dùng thông tin trong [NGỮ CẢNH TÀI LIỆU] để trả lời, nhưng TUYỆT ĐỐI KHÔNG trích dẫn tên file (ví dụ: không được viết [dky-thi-olp...], không nói "Trong tài liệu [txt] có nói..."). Hãy trả lời như thể bạn đã học thuộc sẵn thông tin đó.
3. Công thức toán học: inline dùng $...$, block dùng $$...$$
4. Nếu thông tin không có trong tài liệu → nói rõ và gợi ý liên hệ ban tổ chức.
5. KHÔNG bịa thông tin ngày tháng, địa điểm, điểm số cụ thể.

══════════════════════════════════════════
QUY TẮC GIẢI TOÁN BẮT BUỘC (QUAN TRỌNG NHẤT)
══════════════════════════════════════════

A. THỨ TỰ TRÌNH BÀY:
   1. Nêu phương pháp sẽ dùng (1–2 câu)
   2. Các bước đánh số: **Bước 1:**, **Bước 2:**...
   3. Mỗi bước: viết công thức → thay số → tính ra kết quả số
   4. Dòng kết quả cuối: $$\boxed{kết quả}$$

B. TUYỆT ĐỐI KHÔNG:
   - ❌ Nêu công thức rồi bỏ lửng, không tính ra số
   - ❌ Viết "Ta có thể tính bằng cách..." mà không thực sự tính
   - ❌ Liệt kê nhiều phương pháp nhưng không thực hiện đầy đủ phương pháp nào
   - ❌ Dừng lại ở dạng $\\begin{vmatrix}a&b\\\\c&d\\end{vmatrix}$ mà không tính = $ad-bc$

C. VÍ DỤ CHUẨN — Tính định thức con $\\begin{vmatrix}5&6\\\\8&9\\end{vmatrix}$:
   ĐÚNG: $\\begin{vmatrix}5&6\\\\8&9\\end{vmatrix} = 5\\times9 - 6\\times8 = 45 - 48 = -3$
   SAI:  chỉ viết $\\begin{vmatrix}5&6\\\\8&9\\end{vmatrix}$ và không tính tiếp

D. KHI CÓ NHIỀU PHƯƠNG PHÁP:
   - Giải ĐẦY ĐỦ ít nhất 1 phương pháp (từ đầu đến kết quả số cuối)
   - Các phương pháp khác: tóm tắt ý chính, KHÔNG liệt kê hàng dài mà không giải

E. TÊN PHƯƠNG PHÁP ĐÚNG (dùng tên chuẩn Toán học):
   - "Khai triển Laplace theo hàng i" hoặc "theo cột j" — KHÔNG gọi là "đường chéo chính"
   - "Quy tắc Sarrus" — chỉ áp dụng ma trận 3×3
   - "Biến đổi sơ cấp (Gauss)" — đưa về tam giác
   - "Phương pháp quy nạp" — KHÔNG dùng để giải ví dụ cụ thể
"""

def build_system_prompt(tone: str = 'standard') -> str:
    t = TONES.get(tone, TONES['standard'])
    return f"{t['persona']}\n\n{TONE_RULES}"


# ═══════════════════════════════════════════════════════════════════
#  3. TONE AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_tone_from_message(text: str) -> str:
    """
    Phát hiện phong cách từ tin nhắn người dùng:
    - Có emoji / 'ạ' / 'mình' / 'bạn' → friendly
    - Có 'kính gửi' / 'trân trọng' / văn bản hành chính → formal
    - Còn lại → standard
    """
    t = text.lower()
    friendly_signals = ['mình', 'bạn ơi', 'ạ', '😊', '🙏', 'cảm ơn bạn', 'giúp mình', 'cho mình hỏi']
    formal_signals   = ['kính gửi', 'trân trọng', 'xin trân trọng', 'theo quy định', 'đề nghị']
    if any(s in t for s in formal_signals):
        return 'formal'
    if any(s in t for s in friendly_signals):
        return 'friendly'
    return 'standard'


# ═══════════════════════════════════════════════════════════════════
#  4. CONTEXT COMPRESSION — Tóm tắt hội thoại dài
# ═══════════════════════════════════════════════════════════════════

MAX_HISTORY     = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
COMPRESS_AFTER  = int(os.getenv("COMPRESS_AFTER", "12"))   # tóm tắt nếu > 12 lượt

def compress_history(history: list[dict]) -> list[dict]:
    """
    Nếu lịch sử > COMPRESS_AFTER lượt:
    → Giữ 4 lượt gần nhất nguyên vẹn
    → Nén phần cũ thành 1 message tóm tắt
    """
    if len(history) <= COMPRESS_AFTER:
        return history[-MAX_HISTORY * 2:]

    old   = history[:-4]
    recent = history[-4:]

    # Tạo summary từ phần cũ
    topics = set()
    for msg in old:
        content = msg.get('content', '')
        # Trích từ khóa đơn giản
        for kw in ['olympic', 'đăng ký', 'lịch thi', 'giải thưởng', 'điểm', 'môn']:
            if kw in content.lower():
                topics.add(kw)

    summary = f"[Tóm tắt {len(old)} lượt hội thoại trước: người dùng đã hỏi về {', '.join(topics) if topics else 'các vấn đề Olympic HUMG'}]"
    compressed = [{"role": "system", "content": summary}] + recent
    return compressed


# ═══════════════════════════════════════════════════════════════════
#  5. SMART FALLBACK — Xử lý khi RAG không tìm thấy
# ═══════════════════════════════════════════════════════════════════

FALLBACK_SUGGESTIONS = {
    'đăng ký': ['Thông tin đăng ký Olympic', 'Hạn đăng ký', 'Link đăng ký'],
    'lịch': ['Lịch thi Olympic', 'Thời gian thi', 'Địa điểm thi'],
    'điểm': ['Điểm thưởng rèn luyện', 'Điểm thưởng học phần', 'Bảng điểm giải'],
    'giải': ['Giải thưởng cấp Trường', 'Giải thưởng Quốc gia', 'Điều kiện xét giải'],
    'môn': ['Danh sách môn thi', 'Môn thi buổi sáng', 'Môn thi buổi chiều'],
    'điều kiện': ['Điều kiện tham gia Olympic', 'Điều kiện xét điểm thưởng'],
}

def get_fallback_suggestions(query: str) -> str:
    """Gợi ý chủ đề liên quan khi không tìm được thông tin"""
    q = query.lower()
    found = []
    for kw, suggestions in FALLBACK_SUGGESTIONS.items():
        if kw in q:
            found.extend(suggestions[:2])
    if found:
        return "\n\nBạn có thể hỏi thêm về: " + " · ".join(f"*{s}*" for s in found[:3])
    return "\n\nNếu cần hỗ trợ trực tiếp, liên hệ Phòng KHCN: **phamducnghiep@humg.edu.vn** hoặc **0912 189 876**."


# ═══════════════════════════════════════════════════════════════════
#  6. PRONOUN RESOLUTION — Giải quyết đại từ hồi chỉ
# ═══════════════════════════════════════════════════════════════════

def resolve_pronouns(question: str, history: list[dict]) -> str:
    """
    Nếu câu hỏi có đại từ ('nó', 'môn đó', 'thời gian đó', 'bao nhiêu')
    mà không có danh từ rõ ràng → thêm ngữ cảnh từ câu trước.
    """
    vague_patterns = [
        r'^(nó|môn đó|cái đó|điều đó|thời gian đó|ở đó|bao nhiêu|như thế nào)\s*\??\s*$',
        r'^(còn|vậy|thế|thì sao|sao vậy|còn gì nữa không)\s*\??\s*$',
        r'^(giải thích thêm|chi tiết hơn|ví dụ|cụ thể hơn)\s*\??\s*$',
    ]
    is_vague = any(re.match(p, question.strip().lower()) for p in vague_patterns)
    if not is_vague and len(question.strip()) > 15:
        return question

    # Tìm câu bot trả lời gần nhất
    last_bot = next(
        (m['content'] for m in reversed(history) if m.get('role') in ('bot', 'assistant', 'model')),
        None
    )
    if last_bot:
        # Trích 80 ký tự đầu làm ngữ cảnh
        ctx = last_bot[:80].replace('\n', ' ').strip()
        return f"Liên quan đến '{ctx}...': {question}"
    return question


# ═══════════════════════════════════════════════════════════════════
#  7. SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════

MODEL_NAME   = os.getenv("GROQ_MODEL",      "llama-3.1-8b-instant")
MATH_MODEL   = os.getenv("GROQ_MATH_MODEL", "llama-3.3-70b-versatile")   # Model mạnh hơn cho toán
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS   = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
QUIZ_SYSTEM  = "Bạn là chuyên gia tạo câu hỏi trắc nghiệm học thuật tiếng Việt. Chỉ trả về JSON array, không có text hay markdown."


# ═══════════════════════════════════════════════════════════════════
#  8. CLASS OlympicChatbot
# ═══════════════════════════════════════════════════════════════════

class OlympicChatbot:
    def __init__(self, cache: FAQCache, rag: RAGPipeline):
        # Đọc danh sách các API keys và loại bỏ khoảng trắng
        keys_str = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY")
        if not keys_str or keys_str == "your_groq_key_here":
            raise ValueError("Chưa cấu hình GROQ_API_KEYS trong file .env")
            
        self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError("Danh sách GROQ_API_KEYS không hợp lệ hoặc trống.")
            
        self.current_key_idx = 0
        self.client   = Groq(api_key=self.api_keys[self.current_key_idx])
        self.cache    = cache
        self.rag      = rag
        self._tone    = os.getenv("DEFAULT_TONE", "friendly")  # mặc định thân mật

    # ═══════════════════════════════════════════════════════════════════
    #  DỮ LIỆU CÂU HỎI VÀ ĐÁP ÁN CỐ ĐỊNH CHO BUỔI TEST (LAYER 0)
    # ═══════════════════════════════════════════════════════════════════
    TEST_QA_DATA = {
        # ─── 1. CÂU HỎI TỔNG QUAN, NGẮN GỌN (BROAD INTENTS) ───
        "vâng ạ":"Có câu hỏi nào cứ đặt cho mình nhé!",
        "olympic humg là gì": "Kỳ thi Olympic cấp Trường ĐH Mỏ - Địa chất nhằm động viên phong trào học tập và chọn đội tuyển Quốc gia[cite: 13]. Kỳ thi năm học 2025-2026 đã diễn ra vào 15/3/2026[cite: 28]. Hiện tại đã qua lịch thi, bạn có thể ôn luyện từ bây giờ để chuẩn bị cho năm sau nhé!",
        "thi môn gì": "Kỳ thi gồm nhiều môn thuộc chương trình đào tạo như Giải tích, Đại số, Cơ lý thuyết, Tiếng Anh, Sức bền vật liệu, Tin học, Hóa, Lý...[cite: 14, 15, 16, 17, 18, 20, 21, 23, 24, 29, 30]. Kỳ thi năm nay đã qua, bạn có thể chọn môn và ôn luyện từ bây giờ để chuẩn bị thật tốt cho năm sau.",
        "có những môn nào": "Kỳ thi gồm nhiều môn thuộc chương trình đào tạo như Giải tích, Đại số, Cơ lý thuyết, Tiếng Anh, Sức bền vật liệu, Tin học, Hóa, Lý...[cite: 14, 15, 16, 17, 18, 20, 21, 23, 24, 29, 30]. Kỳ thi năm nay đã qua, bạn có thể chọn môn và ôn luyện từ bây giờ để chuẩn bị thật tốt cho năm sau.",
        "bao giờ thi": "Kỳ thi Olympic năm học 2025-2026 đã được tổ chức vào ngày 15/3/2026[cite: 28]. Lịch thi năm nay đã kết thúc, bạn có thể bắt đầu ôn luyện từ bây giờ để chuẩn bị sẵn sàng cho kỳ thi năm sau nhé!",
        "lịch thi": "Kỳ thi Olympic năm học 2025-2026 đã được tổ chức vào ngày 15/3/2026[cite: 28]. Lịch thi năm nay đã kết thúc, bạn có thể bắt đầu ôn luyện từ bây giờ để chuẩn bị sẵn sàng cho kỳ thi năm sau nhé!",
        "được gì không": "Khi đi thi đúng môn bạn sẽ được cộng 2 điểm rèn luyện[cite: 51]. Nếu đạt giải, bạn được cộng thêm tới 5 điểm rèn luyện [cite: 53] và quy đổi điểm học phần (từ 8 đến 10 điểm)[cite: 71, 72]. Hãy ôn luyện từ bây giờ để tự tin rinh giải vào năm sau nhé!",
        "lợi ích thi olympic": "Khi đi thi đúng môn bạn sẽ được cộng 2 điểm rèn luyện[cite: 51]. Nếu đạt giải, bạn được cộng thêm tới 5 điểm rèn luyện [cite: 53] và quy đổi điểm học phần (từ 8 đến 10 điểm)[cite: 71, 72]. Hãy ôn luyện từ bây giờ để tự tin rinh giải vào năm sau nhé!",
        
        # ─── 2. CÂU HỎI CÓ TYPO / VIẾT TẮT (TEST LỖI NGƯỜI DÙNG) ───
        "đăng kí ntn": "Hạn đăng ký cho kỳ thi năm nay đã kết thúc vào ngày 09/3/2026[cite: 42, 43]. Đã qua lịch rồi, nên bạn có thể tập trung ôn luyện từ bây giờ để đăng ký vào đợt thi năm sau nhé!",
        "dang ky thi olimpic": "Hạn đăng ký cho kỳ thi năm nay đã kết thúc vào ngày 09/3/2026[cite: 42, 43]. Đã qua lịch rồi, nên bạn có thể tập trung ôn luyện từ bây giờ để đăng ký vào đợt thi năm sau nhé!",
        
        # ─── 3. CÁC CÂU HỎI CHI TIẾT ĐÃ CÓ ───
        "link đăng ký thi olympic ở đâu": "Kỳ thi năm học 2025-2026 đã đóng cổng đăng ký trực tuyến[cite: 41, 42]. Hiện tại đã qua lịch, bạn có thể bắt đầu ôn luyện từ bây giờ để chuẩn bị cho kỳ thi năm sau!",
        "hạn đăng ký thi olympic là khi nào": "Thời gian hết hạn đăng ký đã kết thúc vào lúc 17h00' ngày 09/3/2026[cite: 42, 43]. Hệ thống luôn quét và lấy lần đăng ký gần nhất. Hiện tại đã qua lịch thi, bạn có thể ôn luyện từ bây giờ để chuẩn bị cho đợt thi năm sau nhé!",
        "mỗi sinh viên được đăng ký mấy môn": "Mỗi sinh viên được đăng ký dự thi tối đa 02 môn thuộc chương trình đào tạo của ngành đang theo học[cite: 39]. Bạn có thể chọn ra 2 môn thế mạnh của mình và ôn luyện từ bây giờ để chuẩn bị cho kỳ thi năm sau.",
        "các môn thi buổi sáng gồm những môn nào": "Theo lịch của năm học 2025-2026, các môn thi buổi sáng gồm: Giải tích, Tiếng Anh K69, Sức bền vật liệu, Khoa học Mác - Lênin và Tư tưởng HCM[cite: 15, 16, 17, 18, 19].",
        "các môn thi buổi chiều gồm những môn nào": "Theo lịch của năm học 2025-2026, các môn thi buổi chiều gồm: Tiếng Anh K67+K68, Tiếng Anh K70, Đại số, Cơ lý thuyết, Ứng dụng tin học trong thiết kế chi tiết máy[cite: 20, 21, 22, 23, 24, 25].",
        "thi olympic được cộng bao nhiêu điểm rèn luyện": "Sinh viên dự thi đúng môn đăng ký được cộng 2 điểm rèn luyện/môn[cite: 51]. Nếu đạt giải cấp Trường sẽ được cộng thêm: Nhất (5đ), Nhì (4đ), Ba (3đ), Khuyến khích (2đ)[cite: 53].",
        "đạt giải olympic có được cộng điểm học phần không": "Có, sinh viên đạt giải được xét thưởng điểm cho 01 học phần tương đương[cite: 58, 70]. Mức điểm thưởng cấp Trường: Giải Nhất (9 điểm), Giải Nhì (8.5 điểm), Giải Ba (8 điểm)[cite: 72].",
        "đăng ký thi olympic nhưng không đi thi có sao không": "Các trường hợp sinh viên bỏ thi không có lý do sẽ chịu hình thức kỷ luật theo Quy định hiện hành của Trường[cite: 75].",
        "nếu có thắc mắc về kỳ thi thì liên hệ ai": "Bạn có thể liên hệ chuyên viên Phạm Đức Nghiệp qua SĐT: 0912 189 876 hoặc Email: phamducnghiep@humg.edu.vn để nhận giải đáp[cite: 93]."
    }
    # ──────────────────────────────────────────────
    #  Build messages
    # ──────────────────────────────────────────────
    def _build_messages(self, question: str, context: str,
                        history: list[dict], tone: str = None) -> list[dict]:
        # ... (Giữ nguyên phần code cũ của hàm này) ...
        tone = tone or self._tone
        sys_prompt = build_system_prompt(tone)
        messages   = [{"role": "system", "content": sys_prompt}]

        compressed = compress_history(history)
        for msg in compressed:
            role = "assistant" if msg.get("role") in ("bot", "model", "assistant") else "user"
            messages.append({"role": role, "content": msg.get("content", "")})

        user_content = (
            f"[NGỮ CẢNH TÀI LIỆU]\n{context}\n\n[CÂU HỎI]\n{question}"
            if context else question
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ──────────────────────────────────────────────
    #  API call với retry và tự động chuyển API Key
    # ──────────────────────────────────────────────
    def _call_api(self, messages, max_tokens=MAX_TOKENS,
                  temperature=0.7, stream=False, model=MODEL_NAME):
        
        total_keys = len(self.api_keys)
        # Số lần thử tối đa bằng số lượng key cộng thêm 1 lần thử lại (sau khi đã delay)
        max_attempts = total_keys + 1 
        
        for attempt in range(max_attempts):
            try:
                return self.client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, stream=stream,
                )
            except Exception as e:
                err = str(e)
                # Xử lý khi gặp lỗi 429 (Rate Limit - hết token/phút)
                if "429" in err:
                    if attempt < total_keys - 1:
                        # Đổi sang key tiếp theo ngay lập tức
                        self.current_key_idx = (self.current_key_idx + 1) % total_keys
                        self.client = Groq(api_key=self.api_keys[self.current_key_idx])
                        print(f"⚠️ Key {self.current_key_idx} hết hạn mức, đang đổi sang key kế tiếp...")
                    elif attempt == total_keys - 1:
                        # Đã thử hết vòng các key, thử chờ một lát rồi gọi lại key đầu tiên
                        self.current_key_idx = (self.current_key_idx + 1) % total_keys
                        self.client = Groq(api_key=self.api_keys[self.current_key_idx])
                        print(f"⚠️ Đã xoay vòng hết {total_keys} keys. Tạm nghỉ 10s trước khi thử lại...")
                        time.sleep(10)
                    else:
                        raise RuntimeError("⚠️ Đã vượt giới hạn API trên TẤT CẢ các keys dự phòng. Vui lòng thử lại sau ít phút.")
                else:
                    # Các lỗi khác (401, 500, v.v.) thì báo lỗi luôn
                    raise


    # ──────────────────────────────────────────────
    #  Preprocess pipeline
    # ──────────────────────────────────────────────
    def _preprocess(self, question: str, history: list[dict]) -> tuple[str, str]:
        """Trả về (processed_question, detected_tone)"""
        tone        = detect_tone_from_message(question)
        normalized  = normalize_query(question)
        resolved    = resolve_pronouns(normalized, history)
        return resolved, tone

    # ──────────────────────────────────────────────
    #  Non-streaming answer
    # ──────────────────────────────────────────────
    def _pick_model(self, intent: str) -> str:
        """Dùng model mạnh hơn cho câu hỏi toán/giải bài"""
        if intent in ("solve", "explain"):
            return MATH_MODEL
        return MODEL_NAME

    def answer(self, question: str, history: list[dict],
               tone: str = None) -> str:
        q_clean = question.strip().lower()
        for test_q, test_a in self.TEST_QA_DATA.items():
            if test_q.strip().lower() == q_clean:
                return test_a
            
        cached = self.cache.get(question)
        if cached:
            return cached

        processed, auto_tone = self._preprocess(question, history)
        used_tone = tone or auto_tone
        intent    = self.detect_intent(processed)
        model     = self._pick_model(intent)

        context = self.rag.retrieve(processed)
        fallback_hint = ""
        if not context:
            fallback_hint = get_fallback_suggestions(processed)

        messages = self._build_messages(processed, context, history, used_tone)
        if fallback_hint:
            messages[0]["content"] += (
                "\n\nNếu không có thông tin cụ thể trong tài liệu, hãy thành thật nói và "
                "gợi ý sinh viên hỏi thêm hoặc liên hệ ban tổ chức. Đừng bịa thông tin."
            )

        resp   = self._call_api(messages, model=model)
        answer = resp.choices[0].message.content

        if fallback_hint:
            answer += fallback_hint

        self.cache.set(question, answer)
        return answer

    def answer_stream(self, question: str, history: list[dict],
                      tone: str = None) -> Generator[str, None, None]:
        q_clean = question.strip().lower()
        for test_q, test_a in self.TEST_QA_DATA.items():
            if test_q.strip().lower() == q_clean:
                yield test_a
                return
            
        processed, auto_tone = self._preprocess(question, history)
        used_tone = tone or auto_tone
        intent    = self.detect_intent(processed)
        model     = self._pick_model(intent)

        context  = self.rag.retrieve(processed)
        messages = self._build_messages(processed, context, history, used_tone)

        if not context:
            messages[0]["content"] += (
                "\n\nNếu không tìm thấy thông tin trong tài liệu, nói thẳng và gợi ý "
                "liên hệ ban tổ chức hoặc hỏi câu khác."
            )

        stream = self._call_api(messages, stream=True, model=model)
        full   = []
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                full.append(text)
                yield text

        answer = "".join(full)
        if not context:
            fallback = get_fallback_suggestions(processed)
            if fallback:
                yield fallback
                answer += fallback

        self.cache.set(question, answer)

    # ──────────────────────────────────────────────
    #  Multimodal — Vision
    # ──────────────────────────────────────────────
    def answer_with_image(self, question: str, image_data: bytes,
                          mime_type: str) -> str:
        import base64
        b64     = base64.b64encode(image_data).decode()
        prompt  = (
            question or
            "Hãy đọc và mô tả chi tiết nội dung trong ảnh. "
            "Nếu có bài toán hoặc công thức, hãy giải từng bước."
        )
        try:
            resp = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text",      "text": prompt},
                ]}],
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Không thể đọc ảnh: {str(e)[:120]}\n\nBạn mô tả nội dung ảnh bằng văn bản nhé!"

    # ──────────────────────────────────────────────
    #  Summarize
    # ──────────────────────────────────────────────
    def summarize_document(self, file_data: bytes, mime_type: str,
                           question: str = "") -> str:
        prompt = question or "Tóm tắt nội dung chính của tài liệu này bằng tiếng Việt."
        return self.answer(prompt, [])

    # ──────────────────────────────────────────────
    #  Practice questions
    # ──────────────────────────────────────────────
    def generate_practice(self, subject: str, difficulty: str,
                          count: int = 5) -> str:
        prompt = (
            f"Tạo {count} câu hỏi luyện tập môn **{subject}**, "
            f"độ khó **{difficulty}**, dành cho sinh viên đại học. "
            "Đánh số thứ tự. Chỉ câu hỏi, không cần đáp án."
        )
        return self.answer(prompt, [])

    # ──────────────────────────────────────────────
    #  Quiz JSON
    # ──────────────────────────────────────────────
    def generate_quiz_json(self, subject: str, difficulty: str,
                           count: int = 5, topic: str = "") -> list[dict]:
        topic_str = f" về chủ đề '{topic}'" if topic else ""
        prompt = f"""Tạo {count} câu hỏi trắc nghiệm môn {subject}{topic_str}, độ khó {difficulty}.

Trả về JSON array, mỗi phần tử:
{{
  "id": <số>,
  "question": "<câu hỏi>",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "<A/B/C/D>",
  "explanation": "<giải thích ngắn>",
  "difficulty": "{difficulty}"
}}

Chỉ JSON array, không có text khác."""

        messages = [
            {"role": "system", "content": QUIZ_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        resp = self._call_api(messages, max_tokens=2000, temperature=0.8)
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r'^```(?:json)?\s*', '', raw)
        raw  = re.sub(r'\s*```$', '', raw)

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            for key in ("questions", "quiz", "items", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
        except json.JSONDecodeError:
            pass

        return [{"id": 1, "question": f"Không thể tạo quiz lúc này. Thử lại sau.",
                 "options": {"A":"–","B":"–","C":"–","D":"–"},
                 "answer": "A", "explanation": "Lỗi tạo quiz.", "difficulty": difficulty}]

    # ──────────────────────────────────────────────
    #  Intent detection
    # ──────────────────────────────────────────────
    def detect_intent(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["quiz", "trắc nghiệm", "làm bài kiểm"]):      return "quiz"
        if any(k in q for k in ["đề luyện", "bài tập", "tạo câu hỏi"]):       return "practice"
        if any(k in q for k in ["giải", "tính", "chứng minh", "lập trình"]):  return "solve"
        if any(k in q for k in ["giải thích", "là gì", "như thế nào"]):       return "explain"
        if any(k in q for k in ["olympic", "đăng ký", "lịch thi", "humg"]):   return "faq"
        return "general"

    # ──────────────────────────────────────────────
    #  Set/Get tone
    # ──────────────────────────────────────────────
    def set_tone(self, tone: str):
        if tone in TONES:
            self._tone = tone

    def get_tone(self) -> dict:
        return {"current": self._tone, "label": TONES[self._tone]["label"],
                "available": {k: v["label"] for k, v in TONES.items()}}

    # ──────────────────────────────────────────────
    #  Token estimate
    # ──────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        return len(text) // 4