import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI) if MONGODB_URI else None

db = client["Atlas_Hackathon"] if client else None

itineraries_collection = db["itineraries"] if db else None
user_collection = db["users"] if db else None


def has_mongo() -> bool:
    return client is not None and itineraries_collection is not None