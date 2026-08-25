import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("STAYAPI_API_KEY")
if not API_KEY:
    raise ValueError("Missing STAYAPI_API_KEY. Please set it in your .env file.")

headers = {
    "x-api-key": API_KEY,
    "Accept": "application/json"
}

# Example dest_ids:
# Kota Kinabalu = "-2404090", Kuala Lumpur = "-2690040", Bali = "-2694760"
params = {
    "dest_id": "-2404090",
    "checkin": "2026-10-22",
    "checkout": "2026-10-25",
    "adults": 2,
    "rooms": 1,
    "currency": "USD"
}

url = "https://api.stayapi.com/v1/booking/search"

response = requests.get(url, headers=headers, params=params)

print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print("--- RAW RESPONSE PREVIEW ---")
    print(json.dumps(data, indent=2)[:1500])  # View first 1500 characters
else:
    print(f"Error {response.status_code}: {response.text}")