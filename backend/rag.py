"""
rag.py - RAG Pipeline dùng sentence-transformers (offline, không cần API)
Model: paraphrase-multilingual-MiniLM-L12-v2 (hỗ trợ tiếng Việt tốt, nhẹ ~120MB)
"""

import os
import pickle
import json
import re
import numpy as np
from pathlib import Path

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("[RAG] Chưa cài faiss-cpu. Chạy: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False
    print("[RAG] Chưa cài sentence-transformers. Chạy: pip install sentence-transformers")

try:
    from pypdf import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ------------------------------------------------------------------ #
#  Cấu hình                                                            #
# ------------------------------------------------------------------ #
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"  # ~120MB, tiếng Việt tốt
CHUNK_SIZE   = 400    # ký tự mỗi đoạn
CHUNK_OVERLAP = 60    # overlap giữa các đoạn
TOP_K        = int(os.getenv("RAG_TOP_K", "4"))
MIN_SCORE    = 0.35   # ngưỡng cosine similarity tối thiểu


# ------------------------------------------------------------------ #
#  Tiện ích                                                            #
# ------------------------------------------------------------------ #
def _clean_text(text: str) -> str:
    """Làm sạch text: bỏ khoảng trắng thừa, ký tự đặc biệt vô nghĩa"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '...', text)
    return text.strip()

def _chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Chia văn bản thành đoạn có overlap, ưu tiên cắt tại dấu câu"""
    chunks = []
    start = 0
    text = _clean_text(text)
    while start < len(text):
        end = start + size
        if end < len(text):
            # Cố gắng cắt tại dấu câu gần nhất
            for sep in ['.\n', '\n', '. ', '! ', '? ']:
                pos = text.rfind(sep, start + size//2, end)
                if pos > 0:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Bỏ đoạn quá ngắn
            chunks.append(chunk)
        start = end - overlap
    return chunks

def _extract_pdf(path: str) -> list[tuple[str, dict]]:
    """Trích xuất text từ PDF, trả về [(text, metadata), ...]"""
    if not PDF_OK:
        print(f"[RAG] Cần cài pypdf: pip install pypdf")
        return []
    results = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                results.append((text, {"source": Path(path).name, "page": i, "type": "pdf"}))
    except Exception as e:
        print(f"[RAG] Lỗi đọc PDF {path}: {e}")
    return results

def _extract_docx(path: str) -> list[tuple[str, dict]]:
    """Trích xuất text từ DOCX"""
    if not DOCX_OK:
        print(f"[RAG] Cần cài python-docx: pip install python-docx")
        return []
    results = []
    try:
        doc = DocxDocument(path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if full_text.strip():
            results.append((full_text, {"source": Path(path).name, "page": 0, "type": "docx"}))
    except Exception as e:
        print(f"[RAG] Lỗi đọc DOCX {path}: {e}")
    return results

def _extract_txt(path: str) -> list[tuple[str, dict]]:
    """Trích xuất text từ file .txt"""
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        if text.strip():
            return [(text, {"source": Path(path).name, "page": 0, "type": "txt"})]
    except Exception as e:
        print(f"[RAG] Lỗi đọc TXT {path}: {e}")
    return []


# ------------------------------------------------------------------ #
#  Class RAGPipeline                                                   #
# ------------------------------------------------------------------ #
class RAGPipeline:
    def __init__(self, index_dir: str):
        self.index_dir  = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path  = self.index_dir / "faiss.index"
        self.chunks_path = self.index_dir / "chunks.pkl"
        self.meta_path   = self.index_dir / "meta.json"

        self.index:  "faiss.Index | None" = None
        self.chunks: list[str]  = []
        self.meta:   list[dict] = []
        self.ready   = False
        self._embed_model = None  # lazy load

        self._try_load()

    # ---------------------------------------------------------------- #
    #  Lazy load embedding model                                        #
    # ---------------------------------------------------------------- #
    def _get_embed_model(self):
        if self._embed_model is None:
            if not ST_OK:
                raise RuntimeError("sentence-transformers chưa được cài đặt")
            print(f"[RAG] Đang load embedding model '{EMBED_MODEL}'...")
            print("[RAG] Lần đầu sẽ tải ~120MB, các lần sau dùng cache.")
            self._embed_model = SentenceTransformer(EMBED_MODEL)
            print("[RAG] Load model xong!")
        return self._embed_model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_embed_model()
        vecs = model.encode(texts, show_progress_bar=len(texts) > 20,
                            batch_size=32, normalize_embeddings=True)
        return np.array(vecs, dtype='float32')

    # ---------------------------------------------------------------- #
    #  Load index đã build                                              #
    # ---------------------------------------------------------------- #
    def _try_load(self):
        if not FAISS_OK:
            return
        if self.index_path.exists() and self.chunks_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                if self.meta_path.exists():
                    with open(self.meta_path, encoding='utf-8') as f:
                        self.meta = json.load(f)
                self.ready = True
                print(f"[RAG] Đã load index: {len(self.chunks)} đoạn từ {len(set(m.get('source','?') for m in self.meta))} tài liệu")
            except Exception as e:
                print(f"[RAG] Lỗi load index: {e}")

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        print(f"[RAG] Đã lưu index: {len(self.chunks)} đoạn → {self.index_dir}")

    # ---------------------------------------------------------------- #
    #  Build từ thư mục (tự động tìm PDF, DOCX, TXT)                   #
    # ---------------------------------------------------------------- #
    def build_from_directory(self, data_dir: str) -> int:
        """
        Tự động quét thư mục, đọc tất cả PDF/DOCX/TXT, build index.
        Trả về số đoạn đã index.
        """
        if not FAISS_OK:
            print("[RAG] Cần cài faiss-cpu"); return 0

        data_path = Path(data_dir)
        raw_pages: list[tuple[str, dict]] = []

        # Quét tất cả file hỗ trợ
        for f in sorted(data_path.iterdir()):
            ext = f.suffix.lower()
            if ext == '.pdf':
                pages = _extract_pdf(str(f))
                raw_pages.extend(pages)
                print(f"[RAG] PDF: {f.name} → {len(pages)} trang")
            elif ext == '.docx':
                pages = _extract_docx(str(f))
                raw_pages.extend(pages)
                print(f"[RAG] DOCX: {f.name} → {len(pages)} phần")
            elif ext == '.txt':
                pages = _extract_txt(str(f))
                raw_pages.extend(pages)
                print(f"[RAG] TXT: {f.name} → {len(pages)} phần")

        if not raw_pages:
            print("[RAG] Không tìm thấy tài liệu nào để index!")
            return 0

        # Chia thành chunks
        all_chunks, all_meta = [], []
        for text, meta in raw_pages:
            for chunk in _chunk_text(text):
                all_chunks.append(chunk)
                all_meta.append(meta.copy())

        print(f"[RAG] Tổng {len(all_chunks)} đoạn từ {len(raw_pages)} trang. Đang embed...")

        # Embed
        embeddings = self._embed(all_chunks)

        # Build FAISS index (Inner Product = cosine sau normalize)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = all_chunks
        self.meta   = all_meta
        self.ready  = True

        self._save_index()
        print(f"[RAG] ✅ Build xong! {len(all_chunks)} đoạn đã sẵn sàng.")
        return len(all_chunks)

    def build_from_pdf(self, pdf_path: str) -> int:
        """Build từ 1 file PDF"""
        return self.build_from_directory(str(Path(pdf_path).parent))

    # ---------------------------------------------------------------- #
    #  Truy xuất ngữ cảnh                                               #
    # ---------------------------------------------------------------- #
    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Tìm top_k đoạn liên quan nhất với query.
        Trả về chuỗi ngữ cảnh để đưa vào prompt.
        """
        if not self.ready or not FAISS_OK:
            return ""
        try:
            q_vec = self._embed([query])   # shape (1, dim), đã normalize
            scores, indices = self.index.search(q_vec, top_k * 2)

            results = []
            seen = set()
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < MIN_SCORE:
                    continue
                chunk = self.chunks[idx]
                # Deduplicate
                key = chunk[:80]
                if key in seen:
                    continue
                seen.add(key)
                meta = self.meta[idx] if idx < len(self.meta) else {}
                src = f"[{meta.get('source','?')}"
                if meta.get('page'):
                    src += f", tr.{meta['page']}"
                src += "]"
                results.append(f"{src}\n{chunk}")
                if len(results) >= top_k:
                    break

            return "\n\n---\n\n".join(results) if results else ""
        except Exception as e:
            print(f"[RAG] Lỗi retrieve: {e}")
            return ""

    # ---------------------------------------------------------------- #
    #  Thống kê                                                         #
    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        sources = {}
        for m in self.meta:
            s = m.get('source', '?')
            sources[s] = sources.get(s, 0) + 1
        return {
            "ready": self.ready,
            "total_chunks": len(self.chunks),
            "sources": sources,
        }