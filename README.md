# 🤖 Olympic HUMG Chatbot — Backend

Hệ thống chatbot thông minh tích hợp vào Website Olympic HUMG.  
Sử dụng **Gemini 1.5 Flash** (miễn phí 1M token/ngày) + **FAISS RAG** + **Cache thông minh**.

---

## 📁 Cấu trúc thư mục

```
olympic-humg-chatbot/
├── backend/
│   ├── app.py          ← Flask API server (chạy file này)
│   ├── chatbot.py      ← Logic chatbot + tối ưu token
│   ├── rag.py          ← RAG pipeline (FAISS + Gemini Embedding)
│   ├── cache.py        ← Cache 2 lớp (FAQ tĩnh + LRU động)
│   └── data/
│       ├── faq.json        ← 10 câu hỏi FAQ mẫu (thêm tùy ý)
│       ├── quy_che.pdf     ← ⚠️ ĐẶT FILE QUY CHẾ VÀO ĐÂY
│       ├── vector_db/      ← FAISS index (tự sinh ra)
│       ├── usage_log.json  ← Log lượt hỏi (tự sinh ra)
│       └── feedback_log.json ← Log feedback (tự sinh ra)
├── frontend/           ← (Phần 2 — xây dựng sau)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt (3 bước)

### Bước 1 — Lấy Gemini API Key miễn phí

1. Truy cập: https://aistudio.google.com/app/apikey
2. Đăng nhập Google → nhấn **"Create API Key"**
3. Copy key dạng `AIza...`

### Bước 2 — Cài đặt môi trường

```bash
# Clone hoặc giải nén project
cd olympic-humg-chatbot

# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Cài đặt thư viện
pip install -r requirements.txt

# Tạo file .env từ template
cp .env.example .env

# Mở file .env và dán API key vào:
# GEMINI_API_KEY=AIza...your_key_here
```

### Bước 3 — Thêm tài liệu quy chế và chạy

```bash
# (Tùy chọn nhưng QUAN TRỌNG) Đặt file PDF quy chế vào:
# backend/data/quy_che.pdf

# Chạy server
cd backend
python app.py
```

Server khởi động tại: **http://localhost:5000**

---

## 🧪 Test API

Sau khi chạy server, test bằng curl hoặc Postman:

```bash
# Kiểm tra server
curl http://localhost:5000/api/health

# Hỏi câu hỏi thường
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Điều kiện tham gia Olympic HUMG là gì?", "history": []}'

# Build vector database từ PDF quy chế
curl -X POST http://localhost:5000/api/rag/build \
  -H "Content-Type: application/json" \
  -d '{}'

# Xem thống kê
curl http://localhost:5000/api/stats
```

---

## 📡 Danh sách API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/health` | Kiểm tra trạng thái server |
| POST | `/api/chat` | Chat văn bản (cache + RAG) |
| POST | `/api/chat/stream` | Chat streaming (SSE) |
| POST | `/api/chat/image` | Chat với hình ảnh |
| POST | `/api/summarize` | Tóm tắt tài liệu upload |
| POST | `/api/practice` | Tạo đề luyện tập |
| POST | `/api/feedback` | Ghi nhận 👍/👎 |
| GET | `/api/stats` | Dashboard thống kê |
| POST | `/api/rag/build` | Build/rebuild vector database |

---

## ⚙️ Cấu hình nâng cao (.env)

```env
GEMINI_API_KEY=AIza...           # Bắt buộc
MAX_HISTORY_MESSAGES=5           # Sliding window (5 lượt = ~500 token)
MAX_OUTPUT_TOKENS=500            # Giới hạn độ dài câu trả lời
RAG_TOP_K=3                      # Số đoạn văn bản RAG đưa vào prompt
CACHE_MAX_SIZE=100               # Dung lượng dynamic cache
```

---

## 💡 Thêm câu hỏi FAQ

Mở file `backend/data/faq.json` và thêm:

```json
{
  "question": "câu hỏi của bạn",
  "answer": "câu trả lời soạn sẵn"
}
```

FAQ được tìm kiếm bằng fuzzy matching — không cần hỏi y hệt, chỉ cần gần đúng.

---

## 🏗️ Chiến lược tối ưu Token

| Kỹ thuật | Token tiết kiệm |
|----------|----------------|
| Cache FAQ tĩnh (10 câu mẫu) | 100% lượt cache hit |
| LRU dynamic cache (100 câu) | 100% lượt cache hit |
| Sliding window 5 tin nhắn | ~500 token/lượt |
| System prompt < 100 token | ~200 token/lượt |
| RAG có chọn lọc (top 3) | Chỉ lấy đoạn liên quan nhất |

**Ước tính**: ~600 token/lượt → 1.666 lượt/ngày với free tier.  
**Với cache**: thực tế chỉ gọi API ~30-40% → ~4.000-5.000 lượt/ngày hiệu quả.

---

## 🔧 Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `503 Chatbot chưa cấu hình` | Thiếu API key | Thêm `GEMINI_API_KEY` vào `.env` |
| `ModuleNotFoundError: faiss` | Chưa cài faiss | `pip install faiss-cpu` |
| `RAG chưa sẵn sàng` | Chưa có PDF/index | Đặt `quy_che.pdf` vào `data/` rồi POST `/api/rag/build` |
| `CORS error` | Frontend domain khác | Đã cấu hình CORS, kiểm tra URL API |

---

*Phiên bản: 1.0 | NCKH 2025 — Trường Đại học Mỏ - Địa chất*
