"""Parsing helpers that turn Google Places API (New) responses into
frontend-ready ActivityOption objects for the "Add Activity" flow. Mirrors
flights.py's role for Atlas and hotels.py's role for StayAPI: tools.py owns
the raw HTTP call in its agent-facing (@tool, pre-formatted-string) form
(text_search); this module owns a dedicated raw call plus the typed/numeric
extraction used by the REST endpoint (api.py's /activity/search), so the
frontend gets real lat/lng and numeric ratings instead of parsing display
strings back out — same reasoning flights.py documents for why /flight/change
doesn't go through the @tool wrapper.
"""
import math
from typing import Optional

import requests

from schemas import ActivityOption
from tools import GOOGLE_MAPS_API_KEY

PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

_FIELD_MASK = [
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.primaryType",
    "places.types",
    "places.photos",
]

# Maps Google's primaryType/types onto the frontend's coarse category
# buckets (Dining, Cafe, Culture, Nature, Shopping). Anything unmatched
# falls back to "Activity" — the frontend's general bucket.
_TYPE_TO_CATEGORY = {
    "restaurant": "Dining", "meal_takeaway": "Dining", "meal_delivery": "Dining",
    "bar": "Dining", "food": "Dining",
    "cafe": "Cafe", "coffee_shop": "Cafe", "bakery": "Cafe",
    "museum": "Culture", "art_gallery": "Culture", "tourist_attraction": "Culture",
    "place_of_worship": "Culture", "historical_landmark": "Culture",
    "park": "Nature", "national_park": "Nature", "botanical_garden": "Nature",
    "zoo": "Nature", "beach": "Nature", "hiking_area": "Nature",
    "shopping_mall": "Shopping", "clothing_store": "Shopping", "store": "Shopping",
    "market": "Shopping",
}

_CATEGORY_ICON = {
    "Dining": "🍜", "Cafe": "☕", "Culture": "🏛️", "Nature": "🌳", "Shopping": "🛍️",
}

_PRICE_LEVEL_LABEL = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _category_for(place: dict) -> str:
    primary = place.get("primaryType")
    if primary and primary in _TYPE_TO_CATEGORY:
        return _TYPE_TO_CATEGORY[primary]
    for t in place.get("types", []) or []:
        if t in _TYPE_TO_CATEGORY:
            return _TYPE_TO_CATEGORY[t]
    return "Activity"


def _distance_label(
    origin_lat: Optional[float], origin_lng: Optional[float],
    place_lat: Optional[float], place_lng: Optional[float],
) -> str:
    if origin_lat is None or origin_lng is None or place_lat is None or place_lng is None:
        return "—"
    km = _haversine_km(origin_lat, origin_lng, place_lat, place_lng)
    if km < 1:
        return f"{round(km * 1000)}m"
    return f"{km:.1f}km"


def _photo_url(place: dict, max_width: int = 640) -> str:
    photos = place.get("photos") or []
    if not photos:
        return ""
    photo_name = photos[0].get("name")
    if not photo_name:
        return ""
    # Places API (New) photo media endpoint — the returned URL is handed
    # straight to an <img src> on the frontend, so the key has to travel
    # in the query string rather than being used server-side only.
    return (
        f"https://places.googleapis.com/v1/{photo_name}/media"
        f"?maxWidthPx={max_width}&key={GOOGLE_MAPS_API_KEY}"
    )


def _to_activity_option(
    place: dict,
    origin_lat: Optional[float],
    origin_lng: Optional[float],
) -> Optional[ActivityOption]:
    place_id = place.get("id")
    if not place_id:
        return None

    name = (place.get("displayName") or {}).get("text") or "Untitled place"
    loc = place.get("location") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    category = _category_for(place)

    rating_raw = place.get("rating")
    try:
        rating = round(float(rating_raw), 1) if rating_raw is not None else 0.0
    except (ValueError, TypeError):
        rating = 0.0

    return ActivityOption(
        id=str(place_id),
        title=name,
        category=category,
        categoryIcon=_CATEGORY_ICON.get(category, "📍"),
        rating=rating,
        reviewsCount=place.get("userRatingCount"),
        distance=_distance_label(origin_lat, origin_lng, lat, lng),
        priceLabel=_PRICE_LEVEL_LABEL.get(place.get("priceLevel"), ""),
        image=_photo_url(place),
        # Google's Places API has no notion of "sponsored" — every result
        # here is organic search ranking, so this always comes back False
        # (the frontend's Featured section simply won't render, which is
        # correct: there's nothing genuinely sponsored to show there).
        isSponsored=False,
        description=place.get("formattedAddress") or "",
        latitude=lat,
        longitude=lng,
        address=place.get("formattedAddress"),
    )


def search_activities_raw(
    query: str,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius_m: float = 1500.0,
) -> dict:
    """Call Google Places Text Search (New) and return the raw JSON response
    (or {"error": ...} on failure).

    Text Search — rather than Nearby Search — is used as the primary path
    here because AddActivityModal's search box and category pills both
    produce free text ("museums", "ramen shops", "shopping"); Nearby
    Search's includedTypes param only accepts Google's fixed type enum, not
    arbitrary text, so it can't express what the UI actually sends.
    """
    if not GOOGLE_MAPS_API_KEY:
        return {"error": "Missing GOOGLE_MAPS_API_KEY."}

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(_FIELD_MASK),
    }
    payload: dict = {"textQuery": query, "pageSize": 20}
    if lat is not None and lng is not None:
        payload["locationBias"] = {
            "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": radius_m}
        }

    try:
        response = requests.post(PLACES_SEARCH_TEXT_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_activity_options(
    query: str,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
) -> list[ActivityOption]:
    """REST-facing entry point — search_activities_raw() + typed extraction,
    mirroring hotels.py's get_hotel_ui_cards() and flights.py's
    get_flight_options()."""
    raw = search_activities_raw(query, lat, lng)
    if "error" in raw:
        return []
    places = raw.get("places") or []
    options = [_to_activity_option(p, lat, lng) for p in places]
    return [o for o in options if o is not None]