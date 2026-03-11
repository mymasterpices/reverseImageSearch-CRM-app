# crmApp/image-service/db.py

import os
from pymongo import MongoClient

_client = None
_db     = None

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME",   "test")   # your db name


def get_client():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()[DB_NAME]
    return _db
