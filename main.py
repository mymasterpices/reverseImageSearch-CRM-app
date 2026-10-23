"""
crmApp/image-service/main.py
Image similarity + text-to-image search (SigLIP 2 by default) — localhost version
"""
import asyncio
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from db import get_db
from embedder import Embedder
from indexer import IMAGE_EXT, bulk_index_folder, index_one
from store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

UPLOADS_DIR = os.getenv("UPLOADS_DIR", "")
PUBLIC_BASE = os.getenv("PUBLIC_BASE", "http://localhost:5200/uploads")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.50"))
AUTO_INDEX_HOURS = float(os.getenv("AUTO_INDEX_HOURS", "6"))

embedder: Embedder = None
store: VectorStore = None
index_lock = threading.Lock()   # one bulk index job at a time


def count_files() -> int:
    p = Path(UPLOADS_DIR) if UPLOADS_DIR else None
    if not p or not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)


def run_bulk(force=False) -> int:
    if not index_lock.acquire(blocking=False):
        raise HTTPException(409, "Indexing already running")
    try:
        n = bulk_index_folder(UPLOADS_DIR, get_db(), embedder, force=force)
        store.load(get_db(), embedder.name)
        return n
    finally:
        index_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store
    logger.info("Loading embedding model...")
    embedder = await asyncio.to_thread(Embedder)
    logger.info("Model=%s device=%s dim=%d text_search=%s",
                embedder.name, embedder.device, embedder.dim, embedder.supports_text)
    store = VectorStore(embedder.dim)
    await asyncio.to_thread(store.load, get_db(), embedder.name)
    logger.info("Loaded %d vectors | %d files on disk", len(store), count_files())
    if len(store) == 0:
        logger.warning("Nothing indexed for this model. Run: curl -X POST http://127.0.0.1:8002/index/all")
    task = asyncio.create_task(auto_index_loop())
    yield
    task.cancel()


async def auto_index_loop():
    while True:
        await asyncio.sleep(AUTO_INDEX_HOURS * 3600)
        try:
            await asyncio.to_thread(run_bulk, False)
        except Exception as e:
            logger.error("Auto-index error: %s", e)


app = FastAPI(title="CRM Image Search", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def make_url(rel: str) -> str:
    return f"{PUBLIC_BASE}/{rel.replace(chr(92), '/').lstrip('/')}"


def read_image(raw: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")


def format_results(hits):
    return [{"_id": m["_id"], "filename": m["filename"], "subfolder": m["subfolder"],
             "relative_path": m["relative_path"], "score": round(s, 3),
             "imageUrl": make_url(m["relative_path"])} for m, s in hits]


# Plain `def` endpoints run in FastAPI's threadpool, so model inference never blocks the event loop.
@app.get("/health")
def health():
    return {"status": "ok", "model": embedder.name, "device": embedder.device,
            "text_search": embedder.supports_text, "total_files": count_files(),
            "total_indexed": len(store), "ready": len(store) > 0}


@app.post("/search")
def search(file: UploadFile = File(...), top_k: int = 12, min_score: float = None):
    threshold = MIN_SCORE if min_score is None else min_score
    img = read_image(file.file.read())
    if len(store) == 0:
        return {"success": False, "count": 0, "results": [],
                "message": "No images indexed yet. Run POST /index/all first."}
    q = embedder.encode_images([img])[0]
    results = format_results(store.search(q, top_k, threshold))
    logger.info("search %s | threshold=%.2f | %d matches / %d", file.filename, threshold, len(results), len(store))
    return {"success": True, "count": len(results), "results": results}


@app.post("/search/text")
def search_text(query: str, top_k: int = 12, min_score: float = 0.05):
    """Search by description, e.g. 'red cotton kurta with floral print'.
    Text->image scores are naturally lower than image->image, hence the low default."""
    if not embedder.supports_text:
        raise HTTPException(400, f"{embedder.name} has no text encoder. Use a SigLIP/CLIP model.")
    q = embedder.encode_text(query)
    results = format_results(store.search(q, top_k, min_score))
    return {"success": True, "count": len(results), "results": results}


@app.post("/index/all")
def index_all(force: bool = False):
    return {"success": True, "newly_indexed": run_bulk(force)}


@app.post("/index/single")
def index_single(file: UploadFile = File(...), relative_path: str = "", subfolder: str = ""):
    img = read_image(file.file.read())
    vec = embedder.encode_images([img])[0]
    rel = relative_path or f"{subfolder}/{file.filename}".lstrip("/")
    sub = subfolder or (rel.split("/")[0] if "/" in rel else "")
    _id = index_one(get_db(), vec, rel, file.filename, sub, embedder.name)
    store.upsert({"_id": str(_id), "relative_path": rel, "filename": file.filename, "subfolder": sub}, vec)
    return {"success": True, "indexed_id": str(_id), "public_url": make_url(rel)}


@app.get("/stats")
def stats():
    total = count_files()
    return {"model": embedder.name, "uploads_dir": UPLOADS_DIR, "total_files": total,
            "total_indexed": len(store), "pending": max(total - len(store), 0)}
