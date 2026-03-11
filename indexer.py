# crmApp/image-service/indexer.py

import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from bson import ObjectId
from PIL import Image
from pymongo.database import Database

logger    = logging.getLogger(__name__)
IMAGE_EXT = {".jpg", ".jpeg"}


def index_one(
    db:            Database,
    hash_str:      str,
    relative_path: str,
    filename:      str,
    subfolder:     str,
) -> ObjectId:
    """
    Upsert one image hash into MongoDB.
    Keyed on relative_path — safe to call multiple times.
    """
    result = db.image_embeddings.update_one(
        {"relative_path": relative_path},
        {"$set": {
            "hash":          hash_str,
            "relative_path": relative_path,
            "filename":      filename,
            "subfolder":     subfolder,
            "indexed_at":    datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    if result.upserted_id:
        return result.upserted_id
    return db.image_embeddings.find_one(
        {"relative_path": relative_path}, {"_id": 1}
    )["_id"]


def bulk_index_folder(uploads_dir: str, db: Database, force: bool = False) -> int:
    """
    Recursively walk uploads_dir.
    Hash every .jpg/.jpeg not already indexed.
    Returns count of newly indexed images.
    """
    root = Path(uploads_dir)
    if not root.exists():
        logger.error("❌ UPLOADS_DIR not found: %s", uploads_dir)
        return 0

    # Load already-indexed relative paths to skip them
    already_indexed = set()
    if not force:
        for doc in db.image_embeddings.find({}, {"relative_path": 1}):
            already_indexed.add(doc["relative_path"])

    logger.info("📂 Scanning: %s", uploads_dir)
    logger.info("📦 Already indexed: %d", len(already_indexed))

    count  = 0
    errors = 0

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXT:
            continue

        # relative path e.g. "rings/img001.jpg"
        rel = img_path.relative_to(root).as_posix()

        if not force and rel in already_indexed:
            continue

        try:
            raw     = img_path.read_bytes()
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            h       = str(imagehash.phash(pil_img, hash_size=8))

            sub = img_path.parent.relative_to(root).as_posix()
            sub = "" if sub == "." else sub

            index_one(db, h, rel, img_path.name, sub)
            count += 1

            if count % 500 == 0:
                logger.info("  ✅ Indexed %d images so far...", count)

        except Exception as e:
            errors += 1
            logger.warning("  ⚠️  Skip %s — %s", img_path.name, e)

    logger.info("=" * 40)
    logger.info("  Bulk index complete!")
    logger.info("  Newly indexed : %d", count)
    logger.info("  Errors        : %d", errors)
    logger.info("=" * 40)
    return count
