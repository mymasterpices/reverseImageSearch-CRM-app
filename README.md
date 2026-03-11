# crmApp/image-service — Setup for localhost

## Prerequisites
- Python 3.10 or 3.11  (check: python3 --version)
- Your MongoDB running locally

---

## 1. Create a virtual environment

```bash
cd crmApp/image-service

python3 -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First install downloads the CLIP model weights (~350 MB). One time only.

---

## 3. Configure .env

Edit `crmApp/image-service/.env`:

```
MONGO_URI=mongodb://localhost:27017
DB_NAME=crmdb                          # must match your backend DB name
UPLOADS_DIR=../backend/uploads         # path to your images folder
PUBLIC_BASE=http://localhost:5000/images  # Express port + static path
```

---

## 4. Start the service

```bash
# Make sure venv is active, then:
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

You should see:
```
INFO  Loading CLIP model…
INFO  CLIP ready ✓
INFO  Uvicorn running on http://127.0.0.1:8001
```

---

## 5. Verify it's running

```bash
curl http://127.0.0.1:8001/health
```

Expected:
```json
{ "status": "ok", "uploads_dir": "../backend/uploads", "public_base": "http://localhost:5000/images" }
```

---

## 6. Index your existing images (run once)

```bash
curl -X POST http://127.0.0.1:8001/index/all
```

Check progress:
```bash
curl http://127.0.0.1:8001/stats
# { "total_files": 120, "total_indexed": 80, "pending": 40 }
```

---

## Folder structure after setup

```
crmApp/
├── backend/               ← your existing Express app
│   ├── uploads/           ← images read from here by Python
│   └── routes/
│       └── imageSearch.routes.js   ← ADD THIS FILE
├── frontend/              ← your existing Angular app
│   └── src/app/features/
│       └── image-search/  ← ADD THIS FOLDER
└── image-service/         ← NEW (this folder)
    ├── main.py
    ├── db.py
    ├── indexer.py
    ├── requirements.txt
    ├── .env
    └── venv/              ← created by you, not committed to git
```

---

## Running everything together (localhost)

Open 3 terminals:

```bash
# Terminal 1 — Python service
cd crmApp/image-service && source venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8001 --reload

# Terminal 2 — Express backend
cd crmApp/backend && npm run dev

# Terminal 3 — Angular frontend
cd crmApp/frontend && ng serve
```

Then open: http://localhost:4200/image-search
