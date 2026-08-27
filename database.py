import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI="mongodb+srv://rcjc729_db_user:Cgm9dmX78NCAs6EA@cluster0.8eeicbs.mongodb.net/?appName=Cluster0"

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in .env")

client = MongoClient(MONGODB_URI)

db = client["Atlas_Hackathon"]

itineraries_collection = db["itineraries"]
user_collection = db["users"]