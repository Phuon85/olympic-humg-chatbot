"""
cache.py v2 - Cache nâng cấp: Semantic Similarity + Persistent Storage
Nâng cấp so với v1:
  - Semantic cache: so sánh embedding thay vì chỉ fuzzy text
  - Persistent: lưu dynamic cache ra disk → sống sót qua restart
  - Hit stats chi tiết hơn (static / semantic / exact)
"""

import json
import os
import re
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional
import numpy as np


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[?!.,;:\-]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


class FAQCache:
    """
    Cache 3 lớp:
      L1 - FAQ tĩnh: khớp chính xác / một phần với faq.json
      L2 - Dynamic exact: LRU cache cho câu hỏi đã hỏi trước
      L3 - Semantic: so sánh embedding cosine (ngưỡng 0.88)
    """

    CACHE_VERSION = 2

    def __init__(
        self,
        faq_path: str,
        max_dynamic_size: int = 200,
        semantic_threshold: float = 0.88,
        persist_path: Optional[str] = None,
    ):
        self.max_dynamic_size = max_dynamic_size
        self.semantic_threshold = semantic_threshold
        self.persist_path = persist_path  # file .pkl để lưu dynamic cache

        self.static_faq: dict[str, str] = {}
        self.dynamic_cache: OrderedDict[str, str] = OrderedDict()

        # Semantic cache: list of (embedding, question_norm, answer)
        self._sem_keys: list[np.ndarray] = []
        self._sem_vals: list[tuple[str, str]] = []   # (norm, answer)
        self._embed_model = None   # lazy load

        # Stats
        self._hits = {"static": 0, "exact": 0, "semantic": 0}
        self._miss = 0

        self._load_faq(faq_path)
        if persist_path:
            self._load_persist()

    # ---------------------------------------------------------------- #
    #  Embedding model (lazy)                                           #
    # ---------------------------------------------------------------- #
    def _get_embed(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception as e:
                print(f"[Cache] Không load được embedding model: {e}")
                self._embed_model = False   # disable semantic
        return self._embed_model if self._embed_model is not False else None

    def _embed(self, text: str) -> Optional[np.ndarray]:
        model = self._get_embed(  )
        if model is None:
            return None
        vec = model.encode([text], normalize_embeddings=True)
        return vec[0].astype("float32")

    # ---------------------------------------------------------------- #
    #  Load / persist                                                    #
    # ---------------------------------------------------------------- #
    def _load_faq(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            key = _normalize(item["question"])
            self.static_faq[key] = item["answer"]
        print(f"[Cache] Đã load {len(self.static_faq)} FAQ tĩnh")

    def _load_persist(self):
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") == self.CACHE_VERSION:
                self.dynamic_cache = data.get("dynamic", OrderedDict())
                self._sem_keys = data.get("sem_keys", [])
                self._sem_vals = data.get("sem_vals", [])
                print(f"[Cache] Đã load {len(self.dynamic_cache)} dynamic entries từ disk")
        except Exception as e:
            print(f"[Cache] Lỗi load persist: {e}")

    def _save_persist(self):
        if not self.persist_path:
            return
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "version": self.CACHE_VERSION,
                    "dynamic": self.dynamic_cache,
                    "sem_keys": self._sem_keys,
                    "sem_vals": self._sem_vals,
                }, f)
        except Exception as e:
            print(f"[Cache] Lỗi save persist: {e}")

    # ---------------------------------------------------------------- #
    #  Get                                                               #
    # ---------------------------------------------------------------- #
    def get(self, question: str) -> Optional[str]:
        norm = _normalize(question)

        # L1 — Static FAQ exact
        if norm in self.static_faq:
            self._hits["static"] += 1
            return self.static_faq[norm]

        # L1 — Static FAQ partial
        for key, answer in self.static_faq.items():
            if norm in key or key in norm:
                self._hits["static"] += 1
                return answer

        # L2 — Dynamic exact
        if norm in self.dynamic_cache:
            self.dynamic_cache.move_to_end(norm)
            self._hits["exact"] += 1
            return self.dynamic_cache[norm]

        # L3 — Semantic similarity
        q_vec = self._embed(norm)
        if q_vec is not None and self._sem_keys:
            mat = np.stack(self._sem_keys)          # (N, dim)
            scores = mat @ q_vec                     # cosine (vectors normalized)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] >= self.semantic_threshold:
                self._hits["semantic"] += 1
                return self._sem_vals[best_idx][1]

        self._miss += 1
        return None

    # ---------------------------------------------------------------- #
    #  Set                                                               #
    # ---------------------------------------------------------------- #
    def set(self, question: str, answer: str):
        norm = _normalize(question)

        # L2 exact
        if norm in self.dynamic_cache:
            self.dynamic_cache.move_to_end(norm)
        else:
            if len(self.dynamic_cache) >= self.max_dynamic_size:
                self.dynamic_cache.popitem(last=False)
        self.dynamic_cache[norm] = answer

        # L3 semantic
        q_vec = self._embed(norm)
        if q_vec is not None:
            # Kiểm tra trùng semantic trước khi thêm
            already = False
            if self._sem_keys:
                mat = np.stack(self._sem_keys)
                scores = mat @ q_vec
                if float(np.max(scores)) >= self.semantic_threshold:
                    already = True
            if not already:
                if len(self._sem_keys) >= self.max_dynamic_size:
                    self._sem_keys.pop(0)
                    self._sem_vals.pop(0)
                self._sem_keys.append(q_vec)
                self._sem_vals.append((norm, answer))

        self._save_persist()

    # ---------------------------------------------------------------- #
    #  Stats                                                             #
    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        total_hits = sum(self._hits.values())
        total = total_hits + self._miss
        return {
            "static_faq_count": len(self.static_faq),
            "dynamic_cache_count": len(self.dynamic_cache),
            "semantic_cache_count": len(self._sem_keys),
            "hits": self._hits,
            "miss": self._miss,
            "hit_rate_percent": round(total_hits / total * 100, 1) if total else 0,
        }