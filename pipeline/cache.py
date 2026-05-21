import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class BookCache:
    """
    Thread-safe, file-backed cache keyed by normalized (book_name, author_name).
    Only verified entries are stored; cache hits let the pipeline skip the full
    search → scrape → inspect → verify chain.
    """

    def __init__(self, cache_file: str = "book_cache.json"):
        self.cache_file = cache_file
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    def _key(self, book_name: str, author_name: Optional[str]) -> str:
        return f"{self._norm(book_name)}::{self._norm(author_name)}"

    def get(self, book_name: str, author_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not book_name:
            return None
        with self._lock:
            return self._data.get(self._key(book_name, author_name))

    def put(self, book_name: str, author_name: Optional[str], mention: Dict[str, Any]) -> None:
        if not book_name or not mention.get("goodreads_url"):
            return
        entry = {
            "goodreads_url": mention.get("goodreads_url"),
            "scraped_book_name": mention.get("scraped_book_name"),
            "scraped_author_name": mention.get("scraped_author_name"),
            "verified_book_name": mention.get("book_name"),
            "verified_author_name": mention.get("author_name"),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        key = self._key(book_name, author_name)
        with self._lock:
            self._data[key] = entry
            tmp = self.cache_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.cache_file)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
