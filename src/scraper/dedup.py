"""
src/scraper/dedup.py
─────────────────────────────────────────────────────────────────────────────
Two-pass deduplication:
1. Exact hash (MD5) for identical content
2. MinHash LSH for near-duplicate detection (Jaccard >= 0.85)
"""
import hashlib
from datasketch import MinHash, MinHashLSH

class ContentDeduplicator:
    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.seen_hashes: set[str] = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.threshold = threshold
        self.num_perm = num_perm

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        # exact hash
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)

        # near-duplicate via MinHash
        words = set(text.lower().split())
        if not words:
            return False

        m = MinHash(num_perm=self.num_perm)
        for word in words:
            m.update(word.encode("utf-8"))

        matches = self.lsh.query(m)
        if matches:
            return True

        try:
            self.lsh.insert(doc_id, m)
        except ValueError:
            pass
        return False
