# crmApp/image-service/store.py
"""In-memory vector store: one normalised float32 matrix, search = one matrix multiply.
Handles 100k+ images in a few ms. Mongo stays the source of truth."""
import threading
import numpy as np
from pymongo.database import Database


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.lock = threading.RLock()
        self.mat = np.zeros((0, dim), np.float32)
        self.meta: list[dict] = []
        self.pos: dict[str, int] = {}

    def __len__(self):
        return len(self.meta)

    def load(self, db: Database, model: str):
        meta, vecs = [], []
        cursor = db.image_embeddings.find(
            {"model": model, "embedding": {"$exists": True}},
            {"embedding": 1, "relative_path": 1, "filename": 1, "subfolder": 1})
        for d in cursor:
            v = np.frombuffer(d["embedding"], dtype=np.float32)
            if v.shape[0] != self.dim:
                continue
            vecs.append(v)
            meta.append({"_id": str(d["_id"]), "relative_path": d.get("relative_path", ""),
                         "filename": d.get("filename", ""), "subfolder": d.get("subfolder", "")})
        with self.lock:
            self.mat = np.vstack(vecs) if vecs else np.zeros((0, self.dim), np.float32)
            self.meta = meta
            self.pos = {m["relative_path"]: i for i, m in enumerate(meta)}

    def upsert(self, meta: dict, vec: np.ndarray):
        with self.lock:
            i = self.pos.get(meta["relative_path"])
            if i is not None:
                self.mat[i] = vec
                self.meta[i] = meta
            else:
                self.mat = np.vstack([self.mat, vec[None, :]])
                self.meta.append(meta)
                self.pos[meta["relative_path"]] = len(self.meta) - 1

    def search(self, q: np.ndarray, top_k: int, threshold: float):
        with self.lock:
            mat, meta = self.mat, self.meta
        if not len(meta):
            return []
        scores = mat @ q
        idx = np.where(scores >= threshold)[0]
        if idx.size == 0:
            return []
        if idx.size > top_k:
            idx = idx[np.argpartition(-scores[idx], top_k - 1)[:top_k]]
        idx = idx[np.argsort(-scores[idx])]
        return [(meta[i], float(scores[i])) for i in idx]
