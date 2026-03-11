# crmApp/image-service/indexer.py

import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from bson import ObjectId
from PIL import Image
from pymongo.database import Database
from rembg import remove

logger    = logging.getLogger(__name__)
IMAGE_EXT = {".jpg", ".jpeg"}


def index_one(
    db:            Database,
    hash_str:      str,
    relative_path: str,
    filename:      str,
    subfolder:     str,
) -> ObjectId:
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


def remove_bg_and_hash(img_bytes: bytes, session) -> str:
    """Remove background then compute pHash on white canvas."""
    # Remove background
    result = remove(img_bytes, session=session)
    nobg   = Image.open(io.BytesIO(result)).convert("RGBA")

    # Paste on white canvas
    canvas = Image.new("RGB", nobg.size, (255, 255, 255))
    canvas.paste(nobg, mask=nobg.split()[3])

    return str(imagehash.phash(canvas, hash_size=8))


def bulk_index_folder(
    uploads_dir: str,
    db:          Database,
    session,                # rembg session
    force:       bool = False,
) -> int:
    root = Path(uploads_dir)
    if not root.exists():
        logger.error("❌ UPLOADS_DIR not found: %s", uploads_dir)
        return 0

    # Load already-indexed paths
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

        rel = img_path.relative_to(root).as_posix()
        if not force and rel in already_indexed:
            continue

        try:
            raw = img_path.read_bytes()
            h   = remove_bg_and_hash(raw, session)
            sub = img_path.parent.relative_to(root).as_posix()
            sub = "" if sub == "." else sub

            index_one(db, h, rel, img_path.name, sub)
            count += 1

            if count % 100 == 0:
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
