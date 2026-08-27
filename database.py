import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in .env")

client = MongoClient(MONGODB_URI)

db = client["Atlas_Hackathon"]

itineraries_collection = db["itineraries"]
user_collection = db["users"]