"""
crmApp/image-service/main.py

Image similarity search using perceptual hashing (pHash).
- No ML model, no GPU needed
- Works with local MongoDB
- Handles 40K+ images efficiently
- Images served from https://syncdrive.jools.in/images/

Run:
    uvicorn main:app --host 127.0.0.1 --port 8001 --reload
"""

from dotenv import load_dotenv
load_dotenv()

import io
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import imagehash
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from db import get_db
from indexer import bulk_index_folder, index_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config from .env ──────────────────────────────────────────────────────────
UPLOADS_DIR      = os.getenv("UPLOADS_DIR",      "")
PUBLIC_BASE      = os.getenv("PUBLIC_BASE",      "http://localhost:5200/uploads")
MAX_DISTANCE     = int(os.getenv("MAX_DISTANCE",     "10"))
AUTO_INDEX_HOURS = float(os.getenv("AUTO_INDEX_HOURS", "6"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("  CRM Image Search Service")
    logger.info("=" * 55)
    logger.info(f"  UPLOADS_DIR  : {UPLOADS_DIR}")
    logger.info(f"  PUBLIC_BASE  : {PUBLIC_BASE}")
    logger.info(f"  MAX_DISTANCE : {MAX_DISTANCE}")

    p = Path(UPLOADS_DIR)
    if not p.exists():
        logger.error(f"  ❌ UPLOADS_DIR not found: {UPLOADS_DIR}")
        logger.error("     Update UPLOADS_DIR in image-service/.env")
    else:
        count = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg"})
        logger.info(f"  🖼️  Images on disk : {count}")

    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    logger.info(f"  📦 Indexed in DB  : {indexed}")

    if indexed == 0:
        logger.warning("  ⚠️  No images indexed yet!")
        logger.warning("     Run: curl.exe -X POST http://127.0.0.1:8001/index/all")

    logger.info("  ✅ Service ready on http://127.0.0.1:8001")
    logger.info("=" * 55)
    yield


app = FastAPI(title="CRM Image Search", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:5200", "https://syncdrive.jools.in"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core helpers ──────────────────────────────────────────────────────────────

def compute_hash(img: Image.Image) -> str:
    """Perceptual hash — 64 hex chars representing image visuals."""
    return str(imagehash.phash(img.convert("RGB"), hash_size=8))


def hamming_distance(h1: str, h2: str) -> int:
    """Count differing bits. 0 = identical, 64 = completely different."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def make_url(relative_path: str) -> str:
    """Build full public URL from relative path."""
    clean = relative_path.replace("\\", "/").lstrip("/")
    return f"{PUBLIC_BASE}/{clean}"


def read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    p       = Path(UPLOADS_DIR)
    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    total   = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg"}) if p.exists() else 0
    return {
        "status":        "ok",
        "uploads_dir":   UPLOADS_DIR,
        "folder_exists": p.exists(),
        "public_base":   PUBLIC_BASE,
        "total_files":   total,
        "total_indexed": indexed,
        "ready":         indexed > 0,
    }


@app.post("/search")
async def search(
    file:         UploadFile = File(...),
    top_k:        int = 12,
    max_distance: int = None,
):
    """
    Upload a reference image.
    Returns top_k most visually similar images from the indexed collection.
    Each result has an imageUrl pointing to syncdrive.jools.in.
    """
    threshold = max_distance if max_distance is not None else MAX_DISTANCE

    # 1. Hash the uploaded image
    raw        = await file.read()
    query_hash = compute_hash(read_image(raw))
    logger.info(f"🔍 Search | hash={query_hash} | threshold={threshold} | top_k={top_k}")

    db      = get_db()
    indexed = db.image_embeddings.count_documents({})

    if indexed == 0:
        logger.warning("⚠️  Collection is empty — run POST /index/all first")
        return {
            "count":   0,
            "results": [],
            "warning": "No images indexed yet. Run POST /index/all first.",
        }

    # 2. Compare against every stored hash
    # Fast for 40K+ — each doc is tiny (one short string + one path)
    results = []
    for doc in db.image_embeddings.find(
        {},
        {"_id": 1, "hash": 1, "relative_path": 1, "filename": 1, "subfolder": 1}
    ):
        h = doc.get("hash", "")
        if not h:
            continue
        dist = hamming_distance(query_hash, h)
        if dist <= threshold:
            results.append({
                "_id":           str(doc["_id"]),
                "filename":      doc.get("filename", ""),
                "subfolder":     doc.get("subfolder", ""),
                "relative_path": doc.get("relative_path", ""),
                "distance":      dist,
                "score":         round(1 - (dist / 64), 3),
                "imageUrl":      make_url(doc.get("relative_path", "")),
            })

    # 3. Sort by similarity, return top_k
    results.sort(key=lambda x: x["distance"])
    results = results[:top_k]

    logger.info(f"✅ Found {len(results)} matches (scanned {indexed} indexed images)")
    return {"count": len(results), "results": results}


@app.post("/index/all")
async def index_all(force: bool = False):
    """
    Walk UPLOADS_DIR recursively.
    Hash every .jpg/.jpeg and store in MongoDB.
    Safe to re-run — skips already indexed images unless force=true.
    40K images ≈ 40 minutes on any PC.
    """
    logger.info(f"📂 Starting bulk index | force={force} | dir={UPLOADS_DIR}")
    db    = get_db()
    count = bulk_index_folder(UPLOADS_DIR, db, force=force)
    return {"success": True, "newly_indexed": count}


@app.post("/index/single")
async def index_single(
    file:          UploadFile = File(...),
    relative_path: str = "",
    subfolder:     str = "",
):
    """
    Index one newly uploaded image immediately.
    Call this from Express after saving a new image to disk.
    """
    raw  = await file.read()
    h    = compute_hash(read_image(raw))
    db   = get_db()
    rel  = relative_path or f"{subfolder}/{file.filename}".lstrip("/")
    sub  = subfolder or (rel.split("/")[0] if "/" in rel else "")
    _id  = index_one(db, h, rel, file.filename, sub)
    return {
        "success":    True,
        "indexed_id": str(_id),
        "public_url": make_url(rel),
    }


@app.get("/stats")
def stats():
    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    p       = Path(UPLOADS_DIR)
    total   = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg"}) if p.exists() else 0
    return {
        "uploads_dir":   UPLOADS_DIR,
        "folder_exists": p.exists(),
        "public_base":   PUBLIC_BASE,
        "total_files":   total,
        "total_indexed": indexed,
        "pending":       max(total - indexed, 0),
    }
