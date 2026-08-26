import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
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
def parse_datetime(dt_str: str) -> tuple[str, str]:
    if not dt_str or len(dt_str) < 12:
        return "--:--", "Unknown Date"
    dt = datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
    date_formatted = f"{dt.day} {dt.strftime('%b, %A')}"
    return dt.strftime("%H:%M"), date_formatted

def extract_flight_ui_cards(res_data: dict) -> list[dict]:
    routings = res_data.get("routings", [])
    cards = []

    for item in routings:
        segments = item.get("fromSegments") or item.get("segments", [])
        if not segments:
            continue

        first_seg = segments[0]
        last_seg = segments[-1]

        dep_raw = first_seg.get("depTime") or first_seg.get("departureTime", "")
        arr_raw = last_seg.get("arrTime") or last_seg.get("arrivalTime", "")
        dep_time, dep_date = parse_datetime(dep_raw)
        arr_time, arr_date = parse_datetime(arr_raw)

        dep_airport = first_seg.get("depAirport") or first_seg.get("departureAirport", "")
        arr_airport = last_seg.get("arrAirport") or last_seg.get("arrivalAirport", "")
        route_str = f"{dep_airport} - {arr_airport}"

        carrier_val = first_seg.get("carrier")
        if isinstance(carrier_val, dict):
            carrier_name = carrier_val.get("name") or carrier_val.get("code", "Airline")
        elif isinstance(carrier_val, str) and carrier_val.strip():
            carrier_name = carrier_val
        else:
            carrier_name = (
                first_seg.get("carrierName")
                or first_seg.get("marketingAirline")
                or first_seg.get("operatingAirline")
                or "Airline"
            )

        flight_number = first_seg.get("flightNumber", "")

        price_info = item.get("adultPrice") or item.get("price") or item.get("totalPrice") or 0
        currency = item.get("currency") or "USD"
        formatted_price = f"${float(price_info):.2f}" if currency == "USD" else f"{currency} {price_info}"

        num_stops = len(segments) - 1
        layover_text = "Direct"
        if num_stops > 0:
            transfer_airport = first_seg.get("arrAirport") or first_seg.get("arrivalAirport", "")
            layover_text = f"{num_stops} stop in {transfer_airport}"

        card = {
            "routingIdentifier": item.get("routingIdentifier") or item.get("id"),
            "airline": f"{carrier_name} {flight_number}".strip(),
            "route": route_str,
            "departure": {
                "time": dep_time,
                "date": dep_date,
                "airport": first_seg.get("depAirportName") or dep_airport,
            },
            "arrival": {
                "time": arr_time,
                "date": arr_date,
                "airport": last_seg.get("arrAirportName") or arr_airport,
            },
            "layover": layover_text,
            "price": formatted_price,
            "seats_available": f"{item.get('seats') or first_seg.get('seatCount') or 9} Seats Available",
            "refundable": "Refundable" if item.get("refundable") else "Non-refundable",
        }
        cards.append(card)
    return cards

# ==========================================
# Agent Tools
# ==========================================
@tool
def search_flights_atlas(origin: str, destination: str, fromDate: str, returnDate: str, adults: int, children: int, infants: int) -> str:
    """
    Search for flights using the Atlas Flight API.
    Args:
        origin: Origin airport code (e.g., KUL)
        destination: Destination airport code (e.g., BKI)
        date: Flight date (YYYY-MM-DD)
        adults: Number of adult passengers
        children: Number of child passengers
        infants: Number of infant passengers 
    """
    if not ATLAS_CLIENT_ID or not ATLAS_CLIENT_SECRET:
        return json.dumps({"error": "Missing Atlas credentials."})
        
    url = f"{ATLAS_BASE_URL}/search.do"
    clean_fromDate = fromDate.replace("-", "")
    clean_returnDate = returnDate.replace("-", "")
    payload = {
        "tripType": "1", # 1 = Oneway, 2 = Return
        "adultNum": adults,
        "childNum": children,
        "infantNum": infants,
        "fromCity": origin.upper(),
        "toCity": destination.upper(),
        "fromDate": clean_fromDate,
        "retDate": clean_returnDate,
        "includeMultipleFareFamily": False,
        "currency": "USD"
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
            flight_cards = extract_flight_ui_cards(res)
            return json.dumps(flight_cards)
        else:
            return json.dumps({"error": "No flights found or error in request", "details": res})
    except Exception as e:
        return json.dumps({"error": str(e)})

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
    except Exception as e:
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

# ==========================================
# StayAPI Helper
# ==========================================
def _stayapi_request(path: str, params: dict | None = None) -> dict:
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
def search_hotels(params: HotelSearchInput) -> str:
    """
    Search Booking.com hotels in a destination for the given dates.
    Returns a list of hotels with hotel_id, name, rating, price, address, and cancellation flags.
    Use lookup_destination first to obtain dest_id and dest_type.
    Args:
        dest_id: Destination ID obtained from lookup_destination.
        dest_type: Destination type (e.g. CITY, DISTRICT) from lookup_destination.
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.
        adults: Number of adult guests.
        rooms: Number of rooms.
        children: Number of child guests.
    """
    params = {
        "dest_id": params.dest_id,
        "dest_type": params.dest_type,
        "checkin": params.checkin,
        "checkout": params.checkout,
        "adults": params.adults,
        "rooms": params.rooms,
        "children": params.children,
        "children_ages": params.children_ages,
        "rows_per_page": 25,
        "offset": 0,
        "currency": "USD",
    }
    try:
        raw = _stayapi_request("/v1/booking/search", params=params)
        return json.dumps(raw, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

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
def get_hotel_details(hotel_id: str) -> str:
    """
    Fetch rich Booking.com hotel details: address, coordinates, amenities,
    star rating, and review score.
    Args:
        hotel_id: The Booking.com hotel ID.
    """
    try:
        raw = _stayapi_request("/v2/booking/hotel/details", params={"hotel_id": hotel_id})
        return json.dumps(raw, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def get_hotel_prices(hotel_id: str, checkin: str, checkout: str, adults: int = 2, rooms: int = 1) -> str:
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
    params = {
        "hotel_id": hotel_id,
        "check_in": checkin,
        "check_out": checkout,
        "adults": adults,
        "rooms": rooms,
    }
    try:
        raw = _stayapi_request("/v1/booking/hotel/prices", params=params)
        return json.dumps(raw, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
