# crmApp/image-service/indexer.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from PIL import Image
from pymongo.database import Database

logger    = logging.getLogger(__name__)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
WORKERS   = 4


def encode_image(img: Image.Image, model, processor) -> np.ndarray:
    """Encode PIL image with CLIP → normalised 512-dim vector."""
    import torch
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features[0].cpu().numpy().astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)   # L2 normalise
    return vec


def index_one(db: Database, embedding: list, relative_path: str,
              filename: str, subfolder: str) -> ObjectId:
    result = db.image_embeddings.update_one(
        {"relative_path": relative_path},
        {"$set": {
            "embedding":     embedding,
            "relative_path": relative_path,
            "filename":      filename,
            "subfolder":     subfolder,
            "indexed_at":    datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    if result.upserted_id:
        return result.upserted_id
    return db.image_embeddings.find_one({"relative_path": relative_path}, {"_id": 1})["_id"]


def _process_one(args):
    img_path_str, root_str, model, processor = args
    img_path = Path(img_path_str)
    root     = Path(root_str)
    img      = Image.open(img_path).convert("RGB")
    vec      = encode_image(img, model, processor)
    rel      = img_path.relative_to(root).as_posix()
    sub      = img_path.parent.relative_to(root).as_posix()
    sub      = "" if sub == "." else sub
    return rel, img_path.name, sub, vec.tolist()


def bulk_index_folder(uploads_dir: str, db: Database,
                      model, processor, force: bool = False) -> int:
    root = Path(uploads_dir)
    if not root.exists():
        logger.error("❌ UPLOADS_DIR not found: %s", uploads_dir)
        return 0

    # Only skip docs that have 'embedding' field (CLIP indexed)
    # Skip docs that only have 'hash' field (old pHash — needs re-indexing)
    already_indexed = set()
    if not force:
        for doc in db.image_embeddings.find({"embedding": {"$exists": True}}, {"relative_path": 1}):
            already_indexed.add(doc["relative_path"])

    pending = []
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXT:
            continue
        rel = img_path.relative_to(root).as_posix()
        if rel not in already_indexed:
            pending.append(str(img_path))

    total = len(pending)
    logger.info("📂 Root          : %s", root)
    logger.info("📦 CLIP-indexed  : %d", len(already_indexed))
    logger.info("🔄 To process    : %d images | workers=%d", total, WORKERS)

    if total == 0:
        logger.info("✅ All images already indexed.")
        return 0

    count  = 0
    errors = 0
    args   = [(p, str(root), model, processor) for p in pending]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_process_one, a): a for a in args}
        for future in as_completed(futures):
            try:
                rel, filename, sub, emb = future.result()
                index_one(db, emb, rel, filename, sub)
                count += 1
                if count % 100 == 0:
                    logger.info("  ✅ %d / %d indexed...", count, total)
            except Exception as e:
                errors += 1
                logger.warning("  ⚠️  %s", e)

    logger.info("✅ Done — indexed: %d | errors: %d", count, errors)
    return count
