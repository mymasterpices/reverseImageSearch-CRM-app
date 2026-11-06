# CRM Image Search Service

The `image-service/` directory contains a Python FastAPI service for visual product search and text-to-image search. It creates normalized image embeddings with a Hugging Face model, stores those embeddings in MongoDB, and keeps the active model's vectors in memory for similarity searches.

The service is used alongside the Express API and Angular frontend in the repository. It does not replace either application.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Install and run](#install-and-run)
- [HTTP API](#http-api)
- [Indexing workflow](#indexing-workflow)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Integration and deployment notes](#integration-and-deployment-notes)
- [Security and privacy](#security-and-privacy)

## Features

- Search the indexed image collection using an uploaded image.
- Search using a text description when the selected model has a text encoder.
- Batch-index images under the configured upload directory.
- Index or update one uploaded image.
- Load persisted embeddings from MongoDB at service startup.
- Periodically scan for unindexed images.
- Report service readiness and index counts.

Supported indexed image extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Requirements

- Python version compatible with the installed PyTorch and Transformers packages.
- MongoDB reachable by the service.
- Access to the configured image directory.
- Internet access on first model load unless the Hugging Face model is already cached.
- Optional CUDA or Apple MPS hardware for accelerated inference; CPU is supported.

The dependency list is in [`requirements.txt`](./requirements.txt). PyTorch and model weights can be large downloads. On a GPU-less host, install the appropriate CPU-only PyTorch build before installing or upgrading the remaining requirements, following the official PyTorch installation instructions for that host.

## Configuration

The service loads environment variables from `.env` using `python-dotenv`. The `.env` file is local configuration and is excluded from Git. Do not copy secret values into this README or commit them.

| Variable           | Purpose                                                            | Default                           |
| ------------------ | ------------------------------------------------------------------ | --------------------------------- |
| `MONGO_URI`        | MongoDB connection string                                          | `mongodb://localhost:27017`       |
| `DB_NAME`          | MongoDB database containing `image_embeddings`                     | `test`                            |
| `UPLOADS_DIR`      | Root directory recursively scanned for catalogue images            | Empty (no files to scan)          |
| `PUBLIC_BASE`      | URL prefix prepended to result image paths                         | `http://localhost:5200/uploads`   |
| `EMBED_MODEL`      | Hugging Face model used to encode images and, when supported, text | `google/siglip2-base-patch16-224` |
| `MIN_SCORE`        | Default image-search cosine-similarity threshold                   | `0.50`                            |
| `AUTO_INDEX_HOURS` | Interval between non-forced background scans                       | `6`                               |

Example local configuration with placeholder values:

```dotenv
MONGO_URI=mongodb://localhost:27017
DB_NAME=crmdb
UPLOADS_DIR=../backend/uploads
PUBLIC_BASE=http://localhost:5200/uploads
EMBED_MODEL=google/siglip2-base-patch16-224
MIN_SCORE=0.50
AUTO_INDEX_HOURS=6
```

Paths are resolved relative to the process working directory, not automatically relative to the `.env` file. Start Uvicorn from the `image-service/` directory or configure an absolute `UPLOADS_DIR`.

The MongoDB collection used by the index is `image_embeddings` in `DB_NAME`. Embeddings are specific to the configured model; use the same `EMBED_MODEL` when indexing and searching.

### Available embedding models

The model wrapper currently documents these model identifiers:

- `google/siglip2-base-patch16-224` — default; supports image and text queries.
- `google/siglip2-so400m-patch14-384` — larger model; generally needs more memory and benefits from a GPU.
- `facebook/dinov2-base` — image similarity only; text queries are unavailable.
- `openai/clip-vit-base-patch32` — legacy CLIP model.

Changing the model changes the embedding space. The service loads only vectors stored for its active model; re-index the images after switching models to populate that model's index.

## Install and run

Run commands from this `image-service/` directory.

### Create a virtual environment

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
py -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the development service

```bash
uvicorn main:app --host 127.0.0.1 --port 8002 --reload
```

The first startup loads the embedding model, determines its vector dimension, and loads matching vectors from MongoDB. For remote use, a process manager, or service-to-service deployment, configure the bind address and network access deliberately rather than exposing the development server by default.

### Verify

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/stats
```

Interactive API documentation is available at `/docs` while the service is running.

## HTTP API

Unless noted, request bodies and responses are JSON. Image upload endpoints use `multipart/form-data`.

### `GET /health`

Returns the current model, device, text-search support, number of discovered files and indexed vectors, and whether the in-memory index contains at least one vector.

`ready` is false when there are no vectors loaded. The endpoint does not itself run an index job.

### `POST /search`

Searches by an uploaded image.

| Parameter   | Location             | Purpose                                       |
| ----------- | -------------------- | --------------------------------------------- |
| `file`      | Multipart form field | Required query image                          |
| `top_k`     | Query parameter      | Maximum number of results (default `12`)      |
| `min_score` | Query parameter      | Similarity threshold; defaults to `MIN_SCORE` |

Returns `success`, `count`, and a `results` array. Each result includes the stored image ID, filename, subfolder, relative path, similarity score, and public image URL. If the vector index is empty, the service returns an unsuccessful response with instructions to index images.

### `POST /search/text`

Searches images using a text description.

| Parameter   | Location        | Purpose                                  |
| ----------- | --------------- | ---------------------------------------- |
| `query`     | Query parameter | Description to encode                    |
| `top_k`     | Query parameter | Maximum number of results (default `12`) |
| `min_score` | Query parameter | Similarity threshold (default `0.05`)    |

Text scores are model-dependent and are not directly comparable to image-query scores. A model without a text encoder returns HTTP 400.

### `POST /index/all`

Recursively scans `UPLOADS_DIR` and indexes images not yet stored for the active embedding model.

| Parameter | Location        | Purpose                                       |
| --------- | --------------- | --------------------------------------------- |
| `force`   | Query parameter | Set `true` to reprocess already indexed paths |

Only one bulk-index operation can run at a time; a conflicting request returns HTTP 409. The response includes the number of newly indexed images. Missing or empty upload paths are logged and result in no new images indexed.

### `POST /index/single`

Encodes an uploaded image, upserts its vector in MongoDB, and updates the in-memory index.

| Parameter       | Location             | Purpose                                 |
| --------------- | -------------------- | --------------------------------------- |
| `file`          | Multipart form field | Required image                          |
| `relative_path` | Form field           | Optional path relative to `UPLOADS_DIR` |
| `subfolder`     | Form field           | Optional subfolder metadata             |

Returns the MongoDB document ID and the constructed public URL.

### `GET /stats`

Returns model name, configured upload directory, discovered image count, indexed-vector count, and a non-negative pending count.

## Indexing workflow

1. Configure `MONGO_URI`, `DB_NAME`, and `UPLOADS_DIR`.
2. Ensure the image files are readable and MongoDB is available.
3. Start the service and wait for the model and stored vectors to load.
4. Run an initial index:

   ```bash
   curl -X POST http://127.0.0.1:8002/index/all
   ```

5. Check progress and readiness using `/stats` and `/health`.
6. Additions under the configured directory are picked up by the automatic scan, which runs every `AUTO_INDEX_HOURS`; an explicit `POST /index/all` can also be used.

The bulk indexer uses multiple workers to decode files and processes model input in batches. Unreadable images and failed batches are logged and skipped; inspect the service logs when the indexed count is lower than expected.

## Architecture

### Request lifecycle

`main.py` defines the FastAPI application and endpoints. During application startup it:

1. Loads the configured embedding model on CUDA, Apple MPS, or CPU.
2. Determines the embedding dimension.
3. Loads stored vectors for the selected model into a NumPy-backed in-memory vector store.
4. Starts a background loop that periodically scans the upload directory.

Search requests encode the query, perform a normalized vector dot product against the in-memory matrix, apply a score threshold, and return the highest-scoring matches. MongoDB remains the persistent source of truth for embeddings and image metadata.

### Source modules

- `embedder.py` handles model selection, device selection, image/text encoding, and vector normalization.
- `indexer.py` scans and decodes files, batches embedding inference, and upserts records in MongoDB.
- `store.py` holds the active model's vectors and metadata in memory and performs similarity ranking.
- `db.py` lazily creates the MongoDB client and selects the configured database.
- `main.py` configures FastAPI, startup/shutdown lifecycle, periodic indexing, URL formatting, image decoding, and HTTP endpoints.

## Project structure

```text
image-service/
├── main.py                 # FastAPI app, lifecycle, search and indexing endpoints
├── embedder.py             # Hugging Face image/text embeddings and device choice
├── indexer.py              # Filesystem scan, image loading, batch embedding, Mongo upserts
├── store.py                # In-memory vector matrix and similarity search
├── db.py                   # MongoDB client and database selection
├── requirements.txt        # Python dependencies
├── README.md               # This guide
├── .gitignore              # Excludes local environment, virtualenv, and Python caches
├── .env                    # Local configuration; do not commit or document values
├── venv/                   # Local virtual environment; generated, not committed
└── __pycache__/            # Generated Python bytecode cache
```

The tree intentionally describes generated and local-only directories without listing their contents. Model caches may be stored outside this project by the model library.

## Integration and deployment notes

- The Express API's image-search routes forward requests to this service. Configure the backend's `PYTHON_SERVICE_URL` to the reachable service address; its backend default is `http://127.0.0.1:8002`.
- `PUBLIC_BASE` must resolve to a URL that the browser can access for result thumbnails. It should correspond to the static route used to serve the same files in the surrounding application.
- The Angular frontend typically calls the authenticated Express API rather than this Python service directly.
- The service's Mongo database name and upload root must match the intended image-search data and inventory files.
- The in-memory vector store is populated on startup. A full restart is needed to load external database changes that were not applied through this process's indexing endpoints.
- The model and vectors consume memory. Select the model and deployment hardware based on the image count and memory budget.

## Security and privacy

- The service currently enables permissive CORS and does not define authentication or authorization for its endpoints in `main.py`. Restrict network exposure, origins, and access through deployment networking or an authenticated gateway before making it reachable outside a trusted local environment.
- Indexing endpoints can traverse the configured upload tree and write embeddings. Do not expose them to untrusted clients.
- Do not commit `.env`, MongoDB credentials, private keys, service-account files, customer data, or production image data.
- Treat `/health`, `/stats`, and result paths as operational information. Expose only what the deployment requires.
- Use TLS and appropriate access controls when the service communicates across machines or networks.

## Testing

No Python test suite or test command is currently defined in this project. A practical local smoke check is to start the service, request `/health`, index a small known image set, and make a search request using `/search`. Avoid running forced indexing against production data as a test.
