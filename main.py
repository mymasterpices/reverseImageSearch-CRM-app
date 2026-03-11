"""
crmApp/image-service/main.py

Image similarity search using rembg (background removal) + pHash.

HOW IT WORKS:
  1. Index: open image → remove background → pHash the product only → store in MongoDB
  2. Search: remove background from query image → pHash → compare against stored hashes

WHY THIS WORKS:
  - pHash compares only the actual product shape/structure
  - Background is stripped before hashing so it has zero effect
  - Fast search (milliseconds) even for 40K+ images
  - No GPU needed

Run:
    uvicorn main:app --host 127.0.0.1 --port 8001 --reload
"""

import io
import os
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import imagehash
from PIL import Image
from rembg import remove, new_session
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from db import get_db
from indexer import bulk_index_folder, index_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOADS_DIR      = os.getenv("UPLOADS_DIR",      "")
PUBLIC_BASE      = os.getenv("PUBLIC_BASE",      "https://syncdrive.jools.in/images")
MAX_DISTANCE     = int(os.getenv("MAX_DISTANCE",     "10"))
AUTO_INDEX_HOURS = float(os.getenv("AUTO_INDEX_HOURS", "6"))

# rembg session — loaded once at startup
rembg_session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rembg_session

    logger.info("=" * 55)
    logger.info("  CRM Image Search — rembg + pHash")
    logger.info("=" * 55)
    logger.info(f"  UPLOADS_DIR      : {UPLOADS_DIR}")
    logger.info(f"  PUBLIC_BASE      : {PUBLIC_BASE}")
    logger.info(f"  MAX_DISTANCE     : {MAX_DISTANCE}")
    logger.info(f"  AUTO_INDEX_HOURS : {AUTO_INDEX_HOURS}h")

    # Load rembg model (downloads ~170MB on first run, then cached)
    logger.info("  Loading rembg background removal model...")
    rembg_session = new_session("u2net")
    logger.info("  ✅ rembg ready")

    p = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    if not p or not p.exists():
        logger.error(f"  ❌ UPLOADS_DIR not found: {UPLOADS_DIR}")
    else:
        count = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg"})
        logger.info(f"  🖼️  Images on disk : {count}")

    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    logger.info(f"  📦 Indexed in DB  : {indexed}")

    if indexed == 0:
        logger.warning("  ⚠️  No images indexed yet!")
        logger.warning("     Run: curl.exe -X POST http://127.0.0.1:8001/index/all")

    # Start auto-index background task
    task = asyncio.create_task(auto_index_loop())
    logger.info(f"  ✅ Auto-indexer scheduled every {AUTO_INDEX_HOURS}h")
    logger.info("  ✅ Service ready on http://127.0.0.1:8001")
    logger.info("=" * 55)

    yield
    task.cancel()


async def auto_index_loop():
    while True:
        await asyncio.sleep(AUTO_INDEX_HOURS * 3600)
        try:
            logger.info("🔄 Auto-index starting...")
            db    = get_db()
            count = bulk_index_folder(UPLOADS_DIR, db, rembg_session, force=False)
            logger.info(f"🔄 Auto-index done — {count} new images indexed")
        except Exception as e:
            logger.error(f"🔄 Auto-index error: {e}")


app = FastAPI(title="CRM Image Search — rembg+pHash", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:5200", "https://syncdrive.jools.in", "https://crm.jools.in"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core helpers ──────────────────────────────────────────────────────────────

def remove_bg(pil_img: Image.Image) -> Image.Image:
    """Remove background using rembg. Returns RGBA image."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    result = remove(buf.getvalue(), session=rembg_session)
    return Image.open(io.BytesIO(result)).convert("RGBA")


def compute_hash(pil_img: Image.Image) -> str:
    """
    Remove background then compute pHash.
    Paste on white canvas so background pixels don't affect hash.
    """
    nobg   = remove_bg(pil_img)
    canvas = Image.new("RGB", nobg.size, (255, 255, 255))
    canvas.paste(nobg, mask=nobg.split()[3])   # use alpha as mask
    return str(imagehash.phash(canvas, hash_size=8))


def hamming_distance(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def make_url(relative_path: str) -> str:
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
    p       = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    total   = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg"}) if p and p.exists() else 0
    return {
        "status":           "ok",
        "uploads_dir":      UPLOADS_DIR,
        "folder_exists":    p.exists() if p else False,
        "public_base":      PUBLIC_BASE,
        "total_files":      total,
        "total_indexed":    indexed,
        "rembg_ready":      rembg_session is not None,
        "auto_index_every": f"{AUTO_INDEX_HOURS}h",
        "ready":            indexed > 0,
    }


@app.post("/search")
async def search(
    file:         UploadFile = File(...),
    top_k:        int = 12,
    max_distance: int = None,
):
    threshold  = max_distance if max_distance is not None else MAX_DISTANCE
    raw        = await file.read()

    logger.info(f"🔍 Search | file={file.filename} | threshold={threshold} | top_k={top_k}")

    # Remove background from query image, then hash
    query_hash = compute_hash(read_image(raw))
    logger.info(f"   query hash: {query_hash}")

    db      = get_db()
    indexed = db.image_embeddings.count_documents({})

    if indexed == 0:
        return {
            "count":   0,
            "results": [],
            "warning": "No images indexed yet. Run POST /index/all first.",
        }

    results = []
    for doc in db.image_embeddings.find(
        {}, {"_id": 1, "hash": 1, "relative_path": 1, "filename": 1, "subfolder": 1}
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

    results.sort(key=lambda x: x["distance"])
    results = results[:top_k]

    logger.info(f"✅ Found {len(results)} matches (scanned {indexed} indexed images)")
    return {"count": len(results), "results": results}


@app.post("/index/all")
async def index_all(force: bool = False):
    """
    Walk UPLOADS_DIR, remove background from each image, hash, store in MongoDB.
    Use force=true to re-index everything from scratch.
    40K images ≈ 11 hours on CPU (runs in background).
    """
    logger.info(f"📂 Manual index triggered | force={force}")
    db    = get_db()
    count = bulk_index_folder(UPLOADS_DIR, db, rembg_session, force=force)
    return {"success": True, "newly_indexed": count}


@app.post("/index/single")
async def index_single(
    file:          UploadFile = File(...),
    relative_path: str = "",
    subfolder:     str = "",
):
    raw  = await file.read()
    h    = compute_hash(read_image(raw))
    db   = get_db()
    rel  = relative_path or f"{subfolder}/{file.filename}".lstrip("/")
    sub  = subfolder or (rel.split("/")[0] if "/" in rel else "")
    _id  = index_one(db, h, rel, file.filename, sub)
    return {"success": True, "indexed_id": str(_id), "public_url": make_url(rel)}


@app.get("/stats")
def stats():
    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    p       = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    total   = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg"}) if p and p.exists() else 0
    return {
        "uploads_dir":      UPLOADS_DIR,
        "folder_exists":    p.exists() if p else False,
        "public_base":      PUBLIC_BASE,
        "total_files":      total,
        "total_indexed":    indexed,
        "pending":          max(total - indexed, 0),
        "auto_index_every": f"{AUTO_INDEX_HOURS}h",
    }
