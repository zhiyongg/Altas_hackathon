import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

BASE_URL = "https://places.googleapis.com/v1/places:searchNearby"

# Fields to request (Pro + Enterprise SKU fields)
FIELD_MASK = [
    "places.displayName",       # Pro
    "places.location",
    "places.primaryType",
    "places.types",
    "places.formattedAddress",
    "places.rating",            # Enterprise
    "places.regularOpeningHours.weekdayDescriptions",
    "places.priceLevel",
]


def search_nearby(
    center_lat: float,
    center_lng: float,
    radius: float = 500.0,
    included_types: list[str] | None = None,
    excluded_types: list[str] | None = None,
    max_results: int = 10,
    language_code: str = "en",
) -> dict:
    """
    Search for nearby places using the Google Places API (New) Nearby Search endpoint.

    Returns the raw JSON response as a dict.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(FIELD_MASK),
    }

    payload = {
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": center_lat,
                    "longitude": center_lng,
                },
                "radius": radius,
            }
        },
        "languageCode": language_code,
    }

    if included_types:
        payload["includedTypes"] = included_types

    if excluded_types:
        payload["excludedTypes"] = excluded_types

    response = requests.post(BASE_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_restaurants_near_bukit_jalil():
    """Test 1: Search for restaurants near Bukit Jalil, Kuala Lumpur."""
    print("\n>>> TEST 1: Mcd near APU (radius=1000m, max=5)")
    result = search_nearby(
        center_lat=3.0550753,
        center_lng=101.7005763,
        radius=1000.0,
        included_types=["restaurant"],
        max_results=5,
    )
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY is not set. Please add it to your .env file.")
        exit(1)

    print("=" * 60)
    print("  Google Places Nearby Search (New) API - Test Suite")
    print("=" * 60)

    # Run all tests
    test_restaurants_near_bukit_jalil()
    # test_cafes_near_klcc()
    # test_all_types_near_pavilion()
    # test_hotels_excluding_motels()
    # test_no_results_area()

    print("\n\nAll tests completed.")
