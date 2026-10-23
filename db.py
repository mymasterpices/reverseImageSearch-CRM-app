# crmApp/image-service/db.py
import os
from pymongo import MongoClient
from pymongo.database import Database

_client: MongoClient = None


def get_db() -> Database:
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    return _client[os.getenv("DB_NAME", "test")]
