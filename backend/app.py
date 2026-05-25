"""
app.py v2 - Flask API nâng cấp
Mới so với v1:
  - Rate limiting (flask-limiter): 60 req/min/IP
  - /api/quiz endpoint: trả về MCQ JSON có cấu trúc
  - /api/rag/build/all: build từ toàn bộ tài liệu trong data/
  - /api/admin/stats: protected bằng ADMIN_TOKEN
  - Intent detection trong response
  - Persistent cache (lưu dynamic cache ra disk)
  - Better error messages
"""

import os
import json
import base64
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Optional rate limiter
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_OK = True
except ImportError:
    LIMITER_OK = False

from cache import FAQCache
from rag import RAGPipeline
from chatbot import OlympicChatbot

# ------------------------------------------------------------------ #
#  Khởi tạo app                                                        #
# ------------------------------------------------------------------ #
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-Admin-Token"],
    }
})

# Xử lý OPTIONS preflight cho tất cả /api/* routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        from flask import make_response
        res = make_response()
        res.headers["Access-Control-Allow-Origin"]  = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Admin-Token"
        res.headers["Access-Control-Max-Age"]       = "3600"
        return res, 200

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_FILE = DATA_DIR / "usage_log.json"
CACHE_PERSIST = DATA_DIR / "dynamic_cache.pkl"

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "olympic-humg-admin-2025")

# Rate limiting
if LIMITER_OK:
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per hour", "60 per minute"],
        storage_uri="memory://",
    )
    logger.info("✅ Rate limiter đã bật")
else:
    limiter = None
    logger.warning("⚠️ flask-limiter chưa cài — không có rate limiting")

# ------------------------------------------------------------------ #
#  Khởi tạo components                                                  #
# ------------------------------------------------------------------ #
cache = FAQCache(
    faq_path=str(DATA_DIR / "faq.json"),
    max_dynamic_size=200,
    semantic_threshold=float(os.getenv("SEMANTIC_THRESHOLD", "0.88")),
    persist_path=str(CACHE_PERSIST),
)
rag = RAGPipeline(index_dir=str(DATA_DIR / "vector_db"))

# Auto-build RAG nếu chưa có index
if not rag.ready:
    logger.info("Chưa có RAG index → tự build từ thư mục data/...")
    rag.build_from_directory(str(DATA_DIR))

try:
    bot = OlympicChatbot(cache=cache, rag=rag)
    logger.info("✅ Chatbot khởi động thành công!")
except ValueError as e:
    logger.error(f"❌ {e}")
    bot = None


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def require_bot():
    if bot is None:
        return jsonify({
            "error": "Chatbot chưa được cấu hình. Vui lòng thêm GROQ_API_KEY vào file .env",
            "hint": "Lấy API key miễn phí tại: https://console.groq.com"
        }), 503
    return None


def require_admin():
    token = request.headers.get("X-Admin-Token") or request.args.get("token")
    if token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized — cần X-Admin-Token header"}), 401
    return None


def log_usage(question: str, answer: str, source: str, token_estimate: int, intent: str = ""):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question[:200],
        "source": source,
        "token_estimate": token_estimate,
        "answer_length": len(answer),
        "intent": intent,
    }
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    if len(logs) > 10000:
        logs = logs[-10000:]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #
from flask import send_from_directory

@app.route("/favicon.ico")
def favicon():
    return "", 204   # No content — tắt lỗi 404 favicon


@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return send_from_directory(static_dir, filename)


@app.route("/")
def index():
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    return send_from_directory(frontend_dir, "chat-widget.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "2.0",
        "bot_ready": bot is not None,
        "rag_ready": rag.ready,
        "rag_stats": rag.stats(),
        "cache_stats": cache.stats(),
    })


@app.route("/api/tone", methods=["GET"])
def get_tone():
    if bot is None:
        return jsonify({"error": "Bot chưa khởi động"}), 503
    return jsonify(bot.get_tone())


@app.route("/api/tone", methods=["POST"])
def set_tone():
    if bot is None:
        return jsonify({"error": "Bot chưa khởi động"}), 503
    data = request.json or {}
    tone = data.get("tone", "standard")
    bot.set_tone(tone)
    return jsonify({"status": "ok", "tone": bot.get_tone()})


# ─── Chat text ──────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    err = require_bot()
    if err:
        return err

    data     = request.json or {}
    question = (data.get("question") or "").strip()
    history  = data.get("history") or []

    if not question:
        return jsonify({"error": "Câu hỏi không được để trống"}), 400

    try:
        intent = bot.detect_intent(question)
        cached = cache.get(question)
        if cached:
            log_usage(question, cached, "cache", 0, intent)
            return jsonify({"answer": cached, "source": "cache", "intent": intent})

        answer = bot.answer(question, history)
        token_est = bot.count_tokens(question + answer)
        log_usage(question, answer, "api", token_est, intent)
        return jsonify({"answer": answer, "source": "api", "token_estimate": token_est, "intent": intent})

    except Exception as e:
        logger.error(f"Lỗi chat: {e}")
        return jsonify({"error": "Xử lý thất bại. Vui lòng thử lại sau."}), 500


# ─── Chat streaming ─────────────────────────────────────────────────
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    err = require_bot()
    if err:
        return err

    data     = request.json or {}
    question = (data.get("question") or "").strip()
    history  = data.get("history") or []

    if not question:
        return jsonify({"error": "Câu hỏi không được để trống"}), 400

    intent = bot.detect_intent(question) if bot else "general"

    def generate():
        try:
            # Gửi intent trước
            yield f"data: {json.dumps({'intent': intent}, ensure_ascii=False)}\n\n"
            for chunk in bot.answer_stream(question, history):
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Lỗi stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Chat image ─────────────────────────────────────────────────────
@app.route("/api/chat/image", methods=["POST"])
def chat_image():
    err = require_bot()
    if err:
        return err

    data       = request.json or {}
    question   = data.get("question") or ""
    image_b64  = data.get("image_base64") or ""
    mime_type  = data.get("mime_type") or "image/jpeg"

    if not image_b64:
        return jsonify({"error": "Thiếu dữ liệu hình ảnh"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
        answer = bot.answer_with_image(question, image_bytes, mime_type)
        log_usage(f"[IMAGE] {question}", answer, "api_vision", 0, "solve")
        return jsonify({"answer": answer, "source": "api_vision"})
    except Exception as e:
        logger.error(f"Lỗi ảnh: {e}")
        return jsonify({"error": "Không thể xử lý hình ảnh."}), 500


# ─── Quiz (MỚI) ─────────────────────────────────────────────────────
@app.route("/api/quiz", methods=["POST"])
def quiz():
    """
    Tạo quiz trắc nghiệm JSON.
    Body: {"subject": "Toán", "difficulty": "medium", "count": 5, "topic": "Giải tích"}
    """
    err = require_bot()
    if err:
        return err

    data       = request.json or {}
    subject    = data.get("subject") or "Toán học"
    difficulty = data.get("difficulty") or "medium"
    count      = min(int(data.get("count") or 5), 10)
    topic      = data.get("topic") or ""

    try:
        questions = bot.generate_quiz_json(subject, difficulty, count, topic)
        log_usage(f"[QUIZ] {subject} {difficulty}", f"{count} câu", "api", 0, "quiz")
        return jsonify({"questions": questions, "count": len(questions)})
    except Exception as e:
        logger.error(f"Lỗi quiz: {e}")
        return jsonify({"error": "Không thể tạo quiz. Vui lòng thử lại."}), 500


# ─── Summarize ──────────────────────────────────────────────────────
@app.route("/api/summarize", methods=["POST"])
def summarize():
    err = require_bot()
    if err:
        return err

    data     = request.json or {}
    file_b64 = data.get("file_base64") or ""
    question = data.get("question") or ""

    if not file_b64:
        return jsonify({"error": "Thiếu dữ liệu tài liệu"}), 400

    try:
        file_bytes = base64.b64decode(file_b64)
        mime_type  = data.get("mime_type") or "application/pdf"
        answer = bot.summarize_document(file_bytes, mime_type, question)
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Lỗi summarize: {e}")
        return jsonify({"error": "Không thể xử lý tài liệu."}), 500


# ─── Practice ───────────────────────────────────────────────────────
@app.route("/api/practice", methods=["POST"])
def practice():
    err = require_bot()
    if err:
        return err

    data       = request.json or {}
    subject    = data.get("subject") or "Toán học"
    difficulty = data.get("difficulty") or "trung bình"
    count      = min(int(data.get("count") or 5), 10)

    try:
        result = bot.generate_practice(subject, difficulty, count)
        return jsonify({"questions": result})
    except Exception as e:
        logger.error(f"Lỗi practice: {e}")
        return jsonify({"error": "Không thể tạo đề luyện tập."}), 500


# ─── Feedback ───────────────────────────────────────────────────────
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    fb_file = DATA_DIR / "feedback_log.json"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": (data.get("question") or "")[:200],
        "rating": data.get("rating"),
        "comment": data.get("comment") or "",
    }
    logs = []
    if fb_file.exists():
        try:
            with open(fb_file, encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(fb_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})


# ─── Stats (public) ─────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    total = len(logs)
    cache_hits = sum(1 for l in logs if l.get("source") == "cache")
    api_calls  = total - cache_hits
    total_tok  = sum(l.get("token_estimate", 0) for l in logs)

    from collections import defaultdict
    daily: dict = defaultdict(int)
    intent_counts: dict = defaultdict(int)
    for l in logs:
        daily[l.get("timestamp", "")[:10]] += 1
        intent_counts[l.get("intent", "general")] += 1

    fb_logs = []
    fb_file = DATA_DIR / "feedback_log.json"
    if fb_file.exists():
        try:
            with open(fb_file, encoding="utf-8") as f:
                fb_logs = json.load(f)
        except Exception:
            pass

    return jsonify({
        "total_requests": total,
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "cache_hit_rate": round(cache_hits / total * 100, 1) if total else 0,
        "total_tokens_estimated": total_tok,
        "avg_tokens_per_request": round(total_tok / api_calls, 1) if api_calls else 0,
        "daily_requests": [{"date": d, "count": c} for d, c in sorted(daily.items())[-7:]],
        "intent_distribution": dict(intent_counts),
        "feedback": {
            "thumbs_up":   sum(1 for f in fb_logs if f.get("rating") == 1),
            "thumbs_down": sum(1 for f in fb_logs if f.get("rating") == -1),
            "total": len(fb_logs),
        },
        "cache_stats": cache.stats(),
        "rag_stats": rag.stats(),
    })


# ─── RAG build ──────────────────────────────────────────────────────
@app.route("/api/rag/build", methods=["POST"])
def build_rag():
    data = request.json or {}
    pdf_path_str = data.get("pdf_path") or str(DATA_DIR / "quy_che.pdf")
    if not Path(pdf_path_str).exists():
        return jsonify({
            "error": f"Không tìm thấy: {pdf_path_str}",
            "hint": "Đặt file vào thư mục backend/data/"
        }), 404
    try:
        count = rag.build_from_pdf(pdf_path_str)
        return jsonify({"status": "ok", "chunks_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/build/all", methods=["POST"])
def build_rag_all():
    """Build RAG từ TẤT CẢ tài liệu trong thư mục data/"""
    try:
        count = rag.build_from_directory(str(DATA_DIR))
        return jsonify({"status": "ok", "chunks_count": count, "sources": rag.stats()["sources"]})
    except Exception as e:
        logger.error(f"Lỗi build all: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    return jsonify(rag.stats())


# ─── Admin ──────────────────────────────────────────────────────────
@app.route("/api/admin/cache/clear", methods=["POST"])
def admin_clear_cache():
    err = require_admin()
    if err:
        return err
    cache.dynamic_cache.clear()
    cache._sem_keys.clear()
    cache._sem_vals.clear()
    return jsonify({"status": "ok", "message": "Dynamic cache đã được xóa"})


@app.route("/api/admin/logs", methods=["GET"])
def admin_logs():
    err = require_admin()
    if err:
        return err
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    total = len(logs)
    start = (page - 1) * limit
    return jsonify({
        "total": total,
        "page": page,
        "logs": list(reversed(logs))[start:start + limit],
    })


# ------------------------------------------------------------------ #
#  Run                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"🚀 Olympic HUMG Chatbot v2 — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)