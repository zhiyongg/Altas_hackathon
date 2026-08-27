import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("TRIPADVISOR_API_KEY")

# FIX #1: the search endpoint lives under /api/catalog/locations/search,
# not /api/locations/search. The old URL was silently returning a
# valid-shaped 200 response with zero matches instead of a 404.
URL = "https://terra.tripadvisor.com/api/catalog/locations/search"

PARAMS = {
    "query": "Tokyo Tower",
    "country_code": "JP",
    "category": "ATTRACTION",  # allowed values: RESTAURANT, ATTRACTION, HOTEL
    "locale": "en-US",
    "page": 1,
    "size": 3,
}

HEADERS = {
    "accept": "application/json",
    "X-API-Key": API_KEY,
}


# ============================================================
# ONE API CALL ONLY
# ============================================================

response = requests.get(
    URL,
    headers=HEADERS,
    params=PARAMS,
    timeout=30
)

print("=" * 80)
print("HTTP STATUS")
print("=" * 80)
print(response.status_code)


# ============================================================
# ERROR HANDLING
# ============================================================

if response.status_code != 200:

    print("\nAPI ERROR:")
    print(response.text)

    if response.status_code == 400:
        print("\n400 = Bad Request")
    elif response.status_code == 401:
        print("\n401 = Unauthorized")
    elif response.status_code == 403:
        print("\n403 = Your API key may not have access to this endpoint")
    elif response.status_code == 404:
        print("\n404 = Location/resource not found")
    elif response.status_code == 429:
        print("\n429 = Rate limit exceeded")
    elif response.status_code == 500:
        print("\n500 = Tripadvisor server error")

    raise SystemExit


# ============================================================
# PARSE RESPONSE
# ============================================================

data = response.json()

print("\n")
print("=" * 80)
print("RAW API RESPONSE")
print("=" * 80)
print(json.dumps(data, indent=2, ensure_ascii=False))


# ============================================================
# CHECK FOR A RESULTS ARRAY (FIX #2: distinguish "missing field"
# from "field present but empty" so the message is accurate)
# ============================================================

if "data" not in data:
    print("\nUnexpected response shape: no 'data' field found.")
    print("Top-level response keys:")
    print(list(data.keys()))
    raise SystemExit

results = data["data"]
pagination = data.get("pagination", {})

if not results:
    print("\nQuery ran successfully but matched 0 locations.")
    print(f"Pagination info: {pagination}")
    print(
        "\nTry loosening the filters — e.g. drop 'geo_name'/'country_code', "
        "or double-check the 'category' value is one of RESTAURANT/ATTRACTION/HOTEL."
    )
    raise SystemExit


# ============================================================
# PRINT EACH MATCHING LOCATION
# (Note: this is the CATALOG search endpoint, which returns only a
# lightweight projection per location — id, names, addresses,
# coordinates, categories, and match info. Rich fields like ratings,
# opening hours, awards, etc. are NOT part of this response; see the
# optional Location Details call below.)
# ============================================================

print("\n")
print("=" * 80)
print("MATCHING LOCATIONS (catalog projection)")
print("=" * 80)

for index, location in enumerate(results, start=1):

    print(f"\n{'-' * 80}")
    print(f"RESULT #{index}")
    print(f"{'-' * 80}")

    print("Tripadvisor ID:", location.get("id"))

    print("\nName:")
    for name in location.get("names", []):
        print(f"  [{name.get('language')}] {name.get('value')}")

    print("\nAddress:")
    for address in location.get("addresses", []):
        print(f"  {address.get('formatted')}")

    print("\nCoordinates:")
    coordinates = location.get("coordinates", {})
    print(f"  Latitude:  {coordinates.get('latitude')}")
    print(f"  Longitude: {coordinates.get('longitude')}")

    print("\nCategories:")
    for category in location.get("categories", []):
        print(f"  {category.get('display_name')}")
        print(f"  Hierarchy: {category.get('hierarchy')}")
        print(f"  Top level: {category.get('top_level_category')}")

    print("\nFull raw entry:")
    print(json.dumps(location, indent=2, ensure_ascii=False))


# ============================================================
# OPTIONAL: FETCH FULL DETAILS FOR THE TOP MATCH
# This is where all the rich fields (descriptions, opening hours,
# traveler ratings, rankings, awards, photos, etc.) actually live.
# ============================================================

top_match_id = results[0].get("id")

if top_match_id:

    print("\n")
    print("=" * 80)
    print(f"FETCHING FULL DETAILS FOR LOCATION ID {top_match_id}")
    print("=" * 80)

    details_url = f"https://terra.tripadvisor.com/api/locations/{top_match_id}"

    details_response = requests.get(
        details_url,
        headers=HEADERS,
        params={"locale": ["en-US"]},
        timeout=30,
    )

    print("HTTP STATUS:", details_response.status_code)

    if details_response.status_code == 200:
        details = details_response.json()
        print(json.dumps(details, indent=2, ensure_ascii=False))
    else:
        print("Could not fetch full details:")
        print(details_response.text)