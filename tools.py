import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ==========================================
# API Setup & Credentials
# ==========================================
ATLAS_BASE_URL = os.getenv("ATLAS_SANDBOX_BASE_URL", "https://sandbox.atriptech.com").rstrip("/")
ATLAS_CLIENT_ID = os.getenv("ATLAS_CLIENT_ID")
ATLAS_CLIENT_SECRET = os.getenv("ATLAS_CLIENT_SECRET")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

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
    check_in_time: str = "14:00",
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
        transport_mode: Primary mode of transportation ('TRANSIT', 'DRIVE', or 'WALK')
        group_size: Number of travelers
        budget: Overall trip budget level ('low', 'medium', or 'high')
        check_in_time: Hotel check-in time (HH:MM)
        check_out_time: Hotel check-out time (HH:MM)
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
