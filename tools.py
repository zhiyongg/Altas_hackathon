import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Optional
from langchain_core.tools import tool
from schemas import HotelSearchInput

load_dotenv()

# ==========================================
# API Setup & Credentials
# ==========================================
ATLAS_BASE_URL = os.getenv("ATLAS_SANDBOX_BASE_URL", "https://sandbox.atriptech.com").rstrip("/")
ATLAS_CLIENT_ID = os.getenv("ATLAS_CLIENT_ID")
ATLAS_CLIENT_SECRET = os.getenv("ATLAS_CLIENT_SECRET")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

STAYAPI_BASE_URL = "https://api.stayapi.com"
STAYAPI_KEY = os.getenv("STAYAPI_KEY")

# ==========================================
# Helper Functions (Business Logic)
# ==========================================
# NOTE: flight response parsing lives in flights.py (extract_flight_options).
# tools.py used to carry a second, near-identical copy of it
# (parse_datetime + extract_flight_ui_cards) whose only difference was that it
# pre-formatted prices as display strings like "$311.00" — which the agent then
# had to parse back into FlightPrice numbers, and often guessed wrong. The
# @tool below now reuses the typed parser and emits real numbers.

# ==========================================
# Flight Search (raw HTTP call, reusable outside the @tool wrapper)
# ==========================================
def search_flights_raw(
    origin: str,
    destination: str,
    from_date: str,
    return_date: Optional[str],
    adults: int,
    children: int,
    infants: int,
) -> dict:
    """Call the Atlas Flight API and return the raw JSON response (or
    {"error": ...} on failure). Extracted out of search_flights_atlas so
    REST endpoints (api.py's /flight/change) can call it directly without
    going through the LangChain @tool wrapper or its pre-formatted-string
    output (flights.py needs numeric fields, not "$311.00" strings)."""
    if not ATLAS_CLIENT_ID or not ATLAS_CLIENT_SECRET:
        return {"error": "Missing Atlas credentials."}

    url = f"{ATLAS_BASE_URL}/search.do"
    payload = {
        "tripType": "2" if return_date else "1",  # 1 = Oneway, 2 = Return
        "adultNum": adults,
        "childNum": children,
        "infantNum": infants,
        "fromCity": origin.upper(),
        "toCity": destination.upper(),
        "fromDate": from_date.replace("-", ""),
        "retDate": return_date.replace("-", "") if return_date else "",
        "includeMultipleFareFamily": False,
        "currency": "USD",
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-atlas-client-id": ATLAS_CLIENT_ID,
        "x-atlas-client-secret": ATLAS_CLIENT_SECRET,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res = response.json()

        if res.get('status') == "Success" or res.get('routings'):
            return res
        return {"error": "No flights found or error in request", "details": res}
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# Agent Tools
# ==========================================
@tool
def search_flights_atlas(origin: str, destination: str, fromDate: str, returnDate: str, adults: int, children: int, infants: int) -> str:
    """
    Search for flights using the Atlas Flight API.
    Returns a JSON list of flight options with NUMERIC prices and ISO dates.
    Args:
        origin: Origin airport code (e.g., KUL)
        destination: Destination airport code (e.g., BKI)
        fromDate: Flight date (YYYY-MM-DD)
        returnDate: Return date (YYYY-MM-DD), or an empty string for one-way
        adults: Number of adult passengers
        children: Number of child passengers
        infants: Number of infant passengers

    """
    # Imported here, not at module scope: flights.py imports search_flights_raw
    # from this module, so a top-level import would be circular.
    from flights import extract_flight_options

    res = search_flights_raw(origin, destination, fromDate, returnDate, adults, children, infants)
    if "error" in res:
        return json.dumps(res)
    return json.dumps([f.model_dump() for f in extract_flight_options(res)])

@tool
def nearby_search(location: str, keyword: str) -> str:
    """
    Search for nearby places using Google Places Nearby Search API.
    Args:
        location: The center coordinates as a comma-separated string (e.g., "3.055,101.700")
        keyword: The type of place (e.g., cafe, restaurant)
    """
    if not GOOGLE_MAPS_API_KEY:
        return json.dumps({"error": "Missing GOOGLE_MAPS_API_KEY."})
        
    try:
        lat, lng = map(float, location.replace(" ", "").split(","))
    except (ValueError, AttributeError):
        return json.dumps({"error": "Failed to search nearby. Ensure location is 'lat,lng'."})

    url = "https://places.googleapis.com/v1/places:searchNearby"
    field_mask = [
        "places.displayName",
        "places.location",
        "places.primaryType",
        "places.types",
        "places.formattedAddress",
        "places.rating",
        "places.regularOpeningHours.weekdayDescriptions",
        "places.priceLevel",
        "places.websiteUri",
    ]
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(field_mask),
    }

    payload = {
        "maxResultCount": 10,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 500.0,
            }
        },
        "languageCode": "en",
    }

    if keyword:
        payload["includedTypes"] = [keyword]

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        return json.dumps(res.get('places', []))
    except Exception as e:
        return json.dumps({"error": f"API Error: {str(e)}"})

@tool
def text_search(query: str) -> str:
    """
    Search for places using Google Places Text Search API.
    Args:
        query: The search query (e.g., "best coastal cafes in Kota Kinabalu")
    """
    if not GOOGLE_MAPS_API_KEY:
        return json.dumps({"error": "Missing GOOGLE_MAPS_API_KEY."})
        
    url = "https://places.googleapis.com/v1/places:searchText"
    field_mask = [
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.regularOpeningHours.weekdayDescriptions",
        "places.priceLevel",
        "places.primaryType",
    ]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(field_mask),
    }

    payload = {"textQuery": query, "pageSize": 5}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        places = response.json().get("places", [])
        
        # Clean data structure
        results = []
        for place in places:
            results.append({
                "displayName": place.get("displayName", {}).get("text"),
                "formattedAddress": place.get("formattedAddress"),
                "location": place.get("location"),
                "rating": place.get("rating"),
                "weekdayDescriptions": place.get("regularOpeningHours", {}).get("weekdayDescriptions", []),
                "priceLevel": place.get("priceLevel", "PRICE_LEVEL_UNSPECIFIED"),
                "primaryType": place.get("primaryType"),
            })
                
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"API Error: {str(e)}"})

@tool
def plan_itinerary(
    destination: str, 
    start_date: str, 
    end_date: str, 
    arrival_datetime: str, 
    departure_datetime: str, 
    hotel_name: str, 
    hotel_lat: float, 
    hotel_lng: float, 
    preferences: list[str],
    airport_name: str = "",
    airport_lat: float = 0.0,
    airport_lng: float = 0.0,
    travel_style: str = "moderate",
    transport_mode: str = "TRANSIT",
    group_size: int = 2,
    budget: str = "medium",
    check_in_time: str = "15:00",
    check_out_time: str = "11:00",
    custom_vibe: str = ""
) -> str:
    """
    Plan the daily itinerary of attractions and meals.
    
    Call this AFTER finding flights and a hotel to combine them into an itinerary.
    Args:
        destination: The destination city
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        arrival_datetime: YYYY-MM-DDTHH:MM:SS
        departure_datetime: YYYY-MM-DDTHH:MM:SS
        hotel_name: The name of the chosen hotel
        hotel_lat: Latitude of the hotel
        hotel_lng: Longitude of the hotel
        preferences: List of strings for travel preferences (e.g. ['culture', 'food', 'scenery'])
        airport_name: Name of the arrival/departure airport
        airport_lat: Latitude of the airport
        airport_lng: Longitude of the airport
        travel_style: Travel pacing style ('relaxed', 'moderate', or 'packed')
        transport_mode: Primary mode of transportation ('TRANSIT', 'DRIVE', 'WALK' or 'BICYCLE')
        group_size: Number of travelers
        budget: Overall trip budget level ('low', 'medium', or 'high')
        check_in_time: Hotel check-in time (HH:MM, 24-hour)
        check_out_time: Hotel check-out time (HH:MM, 24-hour)
        custom_vibe: Specific semantic vibe (e.g. 'Cyberpunk', 'Romantic')
    """
    from itineraryPlanner import build_itinerary, TripConfig
    
    cfg = TripConfig(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        arrival_datetime=arrival_datetime,
        departure_datetime=departure_datetime,
        hotel={"name": hotel_name, "latitude": hotel_lat, "longitude": hotel_lng},
        airport={"name": airport_name, "latitude": airport_lat, "longitude": airport_lng} if airport_lat and airport_lng else {},
        selected_preferences=preferences,
        travel_style=travel_style,
        transport_mode=transport_mode,
        group_size=group_size,
        budget=budget,
        check_in_time=check_in_time,
        check_out_time=check_out_time,
        custom_vibe=custom_vibe
    )
    
    result = build_itinerary(cfg)
    return json.dumps(result)

# ==========================================
# Date Clamping (StayAPI requires check_in >= today)
# ==========================================
def _clamp_stay_dates(checkin: str, checkout: str) -> tuple[str, str]:
    """Ensure checkin >= today and checkout > checkin.

    StayAPI returns HTTP 400 INVALID_DATES when check_in is before today.
    This can happen when a user opens "Change Accommodation" on a trip
    whose original dates are now in the past. We clamp checkin to today
    and preserve the original trip length (or at least 1 night).
    """
    today = datetime.now().date()
    try:
        ci = datetime.strptime(checkin, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        ci = today
    try:
        co = datetime.strptime(checkout, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        co = ci + timedelta(days=1)

    # Preserve original night count when shifting forward
    original_nights = (co - ci).days

    if ci < today:
        ci = today
        co = ci + timedelta(days=max(original_nights, 1))

    if co <= ci:
        co = ci + timedelta(days=1)

    return ci.strftime("%Y-%m-%d"), co.strftime("%Y-%m-%d")


# ==========================================
# StayAPI Helper
# ==========================================
def _stayapi_request(path: str, params: Optional[dict] = None) -> dict:
    """Generic GET request to StayAPI. Raises on failure."""
    if not STAYAPI_KEY:
        raise RuntimeError("STAYAPI_KEY environment variable is not set.")
    url = f"{STAYAPI_BASE_URL}{path}"
    headers = {"x-api-key": STAYAPI_KEY, "Accept": "application/json"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"StayAPI error {resp.status_code} on GET {path}: {resp.text}")
    body = resp.json()
    if body.get("success") is False:
        raise RuntimeError(f"StayAPI returned success=false on GET {path}: {body}")
    return body


def search_hotels_raw(params: HotelSearchInput) -> dict:
    """Call StayAPI's hotel search endpoint and return the raw JSON response
    (or {"error": ...} on failure). Feeds extract_hotel_ui_cards()/get_hotel_ui_cards().

    NOTE: query param names below follow the Booking.com-style StayAPI
    convention used elsewhere in this codebase (lookup_destination,
    docstrings referencing /v1/booking/search). Verify against StayAPI's
    actual docs/Postman collection before relying on this in production —
    param names for a given provider can vary.
    """
    # Clamp dates so StayAPI never receives a past check_in
    clamped_checkin, clamped_checkout = _clamp_stay_dates(params.checkin, params.checkout)
    query = {
        "dest_id": params.dest_id,
        "dest_type": params.dest_type,
        "check_in": clamped_checkin,
        "check_out": clamped_checkout,
        "adults_number": params.adults,
        "room_number": params.rooms,
        "children_number": params.children,
        "units": "metric",
        "order_by": "popularity",
        "filter_by_currency": params.currency,
        "locale": "en-us",
        "page_number": params.offset // params.rows_per_page if params.rows_per_page else 0,
    }
    if params.children and params.children_ages:
        query["children_ages"] = ",".join(str(a) for a in params.children_ages)

    return _stayapi_request("/v1/booking/search", params=query)


def get_hotel_prices_raw(hotel_id: str, checkin: str, checkout: str, adults: int, rooms: int) -> dict:
    """Call StayAPI's per-hotel room-prices endpoint and return the raw JSON
    response (or {"error": ...} on failure). Feeds _parse_room_options().

    NOTE: same caveat as search_hotels_raw — verify param names/path against
    StayAPI's actual docs.
    """
    # Clamp dates so StayAPI never receives a past check_in
    clamped_checkin, clamped_checkout = _clamp_stay_dates(checkin, checkout)
    query = {
        "hotel_id": hotel_id,
        "check_in": clamped_checkin,
        "check_out": clamped_checkout,
        "adults_number": adults,
        "room_number": rooms,
        "units": "metric",
        "locale": "en-us",
    }
    return _stayapi_request("/v1/booking/hotel/prices", params=query)

# ==========================================
# Hotel / Stay Tools
# ==========================================
@tool
def lookup_destination(query: str) -> str:
    """
    Resolve a free-text city or region name (e.g. 'Paris', 'Bali', 'Kota Kinabalu')
    to a dest_id and dest_type that can be used to search hotels.
    Args:
        query: City or region name to look up.
    """
    try:
        raw = _stayapi_request("/v1/booking/destinations/lookup", params={"query": query, "language": "en-us"})
        return json.dumps(raw, indent=2, default=str)[:8000]
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def search_hotels(
    dest_id: str,
    checkin: str,
    checkout: str,
    dest_type: str = "CITY",
    adults: int = 2,
    rooms: int = 1,
    children: int = 0,
    currency: str = "USD",
) -> str:
    """
    Search Booking.com hotels in a destination for the given dates.
    Returns a list of hotels with hotel_id, name, rating, price, address, and cancellation flags.
    Use lookup_destination first to obtain dest_id and dest_type.
    Args:
        dest_id: Destination ID obtained from lookup_destination (numeric, e.g. '-372490').
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.
        dest_type: Destination type (e.g. CITY, DISTRICT) from lookup_destination.
        adults: Number of adult guests.
        rooms: Number of rooms.
        children: Number of child guests.
        currency: Currency code for prices (e.g. USD).
    """
    # Flat arguments on purpose: this used to take a single `params:
    # HotelSearchInput`, which makes LangChain advertise a nested
    # {"params": {...}} tool schema while the docstring promised flat fields.
    # The model followed the docstring and every call failed validation.
    params = HotelSearchInput(
        dest_id=dest_id,
        dest_type=dest_type,
        checkin=checkin,
        checkout=checkout,
        adults=adults,
        rooms=rooms,
        children=children,
        currency=currency,
    )
    raw = search_hotels_raw(params)
    return json.dumps(raw, indent=2, default=str)

@tool
def meta_search(hotel_name: str, location: str) -> str:
    """
    Find booking links and IDs for a hotel across every OTA StayAPI covers
    (Booking.com, Expedia, TripAdvisor, Agoda, etc.), matched by name and location.
    Args:
        hotel_name: The name of the hotel.
        location: The city or region where the hotel is located.
    """
    try:
        raw = _stayapi_request("/v1/meta/search", params={"hotel_name": hotel_name, "location": location})
        return json.dumps(raw, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_hotel_prices(hotel_id: str, checkin: str, checkout: str, adults: int, rooms: int) -> str:
    """
    Check live nightly rates per room type for a specific hotel across given dates.
    Use this after search_hotels or meta_search has narrowed down to a specific property.
    Args:
        hotel_id: The Booking.com hotel ID.
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.
        adults: Number of adult guests.
        rooms: Number of rooms.
    """
    raw = get_hotel_prices_raw(hotel_id, checkin, checkout, adults, rooms)
    return json.dumps(raw, indent=2, default=str)

if __name__ == "__main__":
    output = search_hotels_raw(HotelSearchInput(
        dest_id="246227", dest_type="CITY",
        checkin="2026-09-20", checkout="2026-09-22",
        adults=2, rooms=1, children=0, children_ages=[],
        currency="USD", offset=0, rows_per_page=10,
    ))
    print(output)
