"""
crmApp/image-service/main.py
CLIP-based image similarity search — localhost version
"""

import io
import os
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from db import get_db
from indexer import bulk_index_folder, index_one, encode_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOADS_DIR      = os.getenv("UPLOADS_DIR", "")
PUBLIC_BASE      = os.getenv("PUBLIC_BASE", "http://localhost:5200/uploads")
MIN_SCORE        = float(os.getenv("MIN_SCORE", "0.60"))
AUTO_INDEX_HOURS = float(os.getenv("AUTO_INDEX_HOURS", "6"))

clip_model     = None
clip_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_model, clip_processor

    logger.info("=" * 55)
    logger.info("  CRM Image Search — CLIP (localhost)")
    logger.info("=" * 55)
    logger.info(f"  UPLOADS_DIR  : {UPLOADS_DIR}")
    logger.info(f"  PUBLIC_BASE  : {PUBLIC_BASE}")
    logger.info(f"  MIN_SCORE    : {MIN_SCORE}")

    logger.info("  Loading CLIP model...")
    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        logger.info("  ✅ CLIP ready")
    except Exception as e:
        logger.error(f"  ❌ CLIP load failed: {e}")
        raise

    p = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    if not p or not p.exists():
        logger.error(f"  ❌ UPLOADS_DIR not found: '{UPLOADS_DIR}'")
    else:
        count = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        logger.info(f"  🖼️  Images on disk: {count}")

    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    logger.info(f"  📦 Indexed in DB : {indexed}")

    if indexed == 0:
        logger.warning("  ⚠️  Run: curl.exe -X POST http://127.0.0.1:8002/index/all")

    asyncio.create_task(auto_index_loop())
    logger.info("  ✅ Service ready → http://127.0.0.1:8002")
    logger.info("=" * 55)
    yield


async def auto_index_loop():
    while True:
        await asyncio.sleep(AUTO_INDEX_HOURS * 3600)
        try:
            db = get_db()
            bulk_index_folder(UPLOADS_DIR, db, clip_model, clip_processor, force=False)
        except Exception as e:
            logger.error(f"Auto-index error: {e}")


app = FastAPI(title="CRM Image Search", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def make_url(relative_path: str) -> str:
    clean = relative_path.replace("\\", "/").lstrip("/")
    return f"{PUBLIC_BASE}/{clean}"


def read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")


@app.get("/health")
def health():
    p       = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    total   = 0
    if p and p.exists():
        total = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return {
        "status":        "ok",
        "uploads_dir":   UPLOADS_DIR,
        "folder_exists": p.exists() if p else False,
        "total_files":   total,
        "total_indexed": indexed,
        "clip_ready":    clip_model is not None,
        "ready":         indexed > 0,
    }


@app.post("/search")
async def search(
    file:      UploadFile = File(...),
    top_k:     int   = 12,
    min_score: float = None,
):
    threshold = min_score if min_score is not None else MIN_SCORE
    raw       = await file.read()
    img       = read_image(raw)

    logger.info(f"🔍 {file.filename} | threshold={threshold} | top_k={top_k}")

    query_vec = encode_image(img, clip_model, clip_processor)

    db      = get_db()
    indexed = db.image_embeddings.count_documents({})

    if indexed == 0:
        return {"success": False, "count": 0, "results": [],
                "message": "No images indexed yet. Run POST /index/all first."}

    results = []
    for doc in db.image_embeddings.find(
        {}, {"_id": 1, "embedding": 1, "relative_path": 1, "filename": 1, "subfolder": 1}
    ):
        emb = doc.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_vec, np.array(emb, dtype=np.float32))
        if score >= threshold:
            results.append({
                "_id":           str(doc["_id"]),
                "filename":      doc.get("filename", ""),
                "subfolder":     doc.get("subfolder", ""),
                "relative_path": doc.get("relative_path", ""),
                "score":         round(score, 3),
                "imageUrl":      make_url(doc.get("relative_path", "")),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    logger.info(f"✅ Found {len(results)} matches (scanned {indexed})")
    return {"success": True, "count": len(results), "results": results}


@app.post("/index/all")
async def index_all(force: bool = False):
    logger.info(f"📂 Index triggered | force={force}")
    db    = get_db()
    count = bulk_index_folder(UPLOADS_DIR, db, clip_model, clip_processor, force=force)
    return {"success": True, "newly_indexed": count}


@app.post("/index/single")
async def index_single(
    file:          UploadFile = File(...),
    relative_path: str = "",
    subfolder:     str = "",
):
    raw = await file.read()
    img = read_image(raw)
    emb = encode_image(img, clip_model, clip_processor)
    db  = get_db()
    rel = relative_path or f"{subfolder}/{file.filename}".lstrip("/")
    sub = subfolder or (rel.split("/")[0] if "/" in rel else "")
    _id = index_one(db, emb.tolist(), rel, file.filename, sub)
    return {"success": True, "indexed_id": str(_id), "public_url": make_url(rel)}


@app.get("/stats")
def stats():
    db      = get_db()
    indexed = db.image_embeddings.count_documents({})
    p       = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    total   = 0
    if p and p.exists():
        total = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return {
        "uploads_dir":   UPLOADS_DIR,
        "folder_exists": p.exists() if p else False,
        "total_files":   total,
        "total_indexed": indexed,
        "pending":       max(total - indexed, 0),
    }
