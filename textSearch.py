import os
import requests
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY. Please set it in your .env file.")

def get_places(query: str, max_results: int = 5) -> list[dict]:
    url = "https://places.googleapis.com/v1/places:searchText"

    # Only requested fields
    field_mask = [
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.regularOpeningHours.weekdayDescriptions",
        "places.priceLevel",
        "places.primaryType",
        #"places.websiteUri"   # Added the place's URI for user to redirect to the place's website
    ]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(field_mask),
    }

    payload = {"textQuery": query, "pageSize": max_results}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    places = response.json().get("places", [])

    # Clean data structure for LLM consumption
    results = []
    for place in places:
        results.append(
            {
                "displayName": place.get("displayName", {}).get("text"),
                "formattedAddress": place.get("formattedAddress"),
                "location": place.get("location"),  # {'latitude': ..., 'longitude': ...}
                "rating": place.get("rating"),
                "weekdayDescriptions": place.get("regularOpeningHours", {}).get(
                    "weekdayDescriptions", []
                ),
                "priceLevel": place.get("priceLevel", "PRICE_LEVEL_UNSPECIFIED"),
                "primaryType": place.get("primaryType"),
                #"websiteUri": place.get("websiteUri"),
            }
        )

    return results




# Example test
if __name__ == "__main__":
    data = get_places("Asia Pacific University", max_results=2)
    import json

    print(json.dumps(data, indent=2))