# crmApp/image-service/indexer.py
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import Binary, ObjectId
from PIL import Image, ImageOps
from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
IO_WORKERS = 4        # threads for decoding images from disk
BATCH_SIZE = 16       # images per model forward pass


def load_image(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)           # respect phone-camera rotation
    if img.mode in ("RGBA", "LA", "P"):          # flatten transparency onto white, not black
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _doc(rel, filename, sub, vec, model):
    return {"embedding": Binary(vec.astype(np.float32).tobytes()), "model": model,
            "relative_path": rel, "filename": filename, "subfolder": sub,
            "indexed_at": datetime.now(timezone.utc)}


def index_one(db: Database, vec: np.ndarray, rel: str, filename: str, sub: str, model: str) -> ObjectId:
    db.image_embeddings.update_one({"relative_path": rel},
                                   {"$set": _doc(rel, filename, sub, vec, model)}, upsert=True)
    return db.image_embeddings.find_one({"relative_path": rel}, {"_id": 1})["_id"]


def bulk_index_folder(uploads_dir: str, db: Database, embedder, force: bool = False) -> int:
    root = Path(uploads_dir)
    if not root.exists():
        logger.error("UPLOADS_DIR not found: %s", uploads_dir)
        return 0

    done = set()
    if not force:   # only skip images already embedded with THIS model
        done = {d["relative_path"] for d in
                db.image_embeddings.find({"model": embedder.name}, {"relative_path": 1})}

    pending = [p for p in root.rglob("*")
               if p.suffix.lower() in IMAGE_EXT and p.relative_to(root).as_posix() not in done]
    logger.info("Model %s | already indexed %d | to process %d", embedder.name, len(done), len(pending))
    if not pending:
        return 0

    def safe_load(p):
        try:
            return p, load_image(p)
        except Exception as e:
            logger.warning("skip %s: %s", p, e)
            return p, None

    count = errors = 0
    with ThreadPoolExecutor(IO_WORKERS) as pool:
        for i in range(0, len(pending), BATCH_SIZE):
            loaded = [(p, im) for p, im in pool.map(safe_load, pending[i:i + BATCH_SIZE]) if im is not None]
            errors += len(pending[i:i + BATCH_SIZE]) - len(loaded)
            if not loaded:
                continue
            try:
                vecs = embedder.encode_images([im for _, im in loaded])
            except Exception as e:
                errors += len(loaded)
                logger.warning("batch failed: %s", e)
                continue
            ops = []
            for (p, _), vec in zip(loaded, vecs):
                rel = p.relative_to(root).as_posix()
                sub = p.parent.relative_to(root).as_posix()
                sub = "" if sub == "." else sub
                ops.append(UpdateOne({"relative_path": rel},
                                     {"$set": _doc(rel, p.name, sub, vec, embedder.name)}, upsert=True))
            db.image_embeddings.bulk_write(ops, ordered=False)
            count += len(ops)
            if count % 160 < BATCH_SIZE:
                logger.info("  %d / %d indexed", count, len(pending))

    logger.info("Done: indexed=%d errors=%d", count, errors)
    return count
