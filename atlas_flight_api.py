import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv(
    "ATLAS_SANDBOX_BASE_URL", "https://sandbox.atriptech.com"
).rstrip("/")
CLIENT_ID = os.getenv("ATLAS_CLIENT_ID")
CLIENT_SECRET = os.getenv("ATLAS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        f"Missing credentials! CLIENT_ID: {CLIENT_ID}, CLIENT_SECRET: {CLIENT_SECRET}\n"
        "Ensure your .env file is in the project root where you execute the script."
    )
    
HEADERS = {
    "Content-Type": "application/json",
    "Accept":"application/json",
    "Accept-Encoding": "gzip",
    "x-atlas-client-id": CLIENT_ID,
    "x-atlas-client-secret": CLIENT_SECRET,
}


def search_atlas_flights(
    origin: str,
    destination: str,
    departure_date: str,  # Accepts "YYYY-MM-DD" or "YYYYMMDD"
    adults: int = 1,
    trip_type: str = "1",  # "1" = One-Way, "2" = Round-Trip
    airlines: list = None,
) -> dict:
    url = f"{BASE_URL}/search.do"

    # Automatically sanitize date to YYYYMMDD
    clean_date = departure_date.replace("-", "")

    payload = {
        "tripType": str(trip_type),
        "adultNum": adults,
        "childNum": 0,
        "infantNum": 0,
        "fromCity": origin.upper(),
        "fromAirport": "",
        "toCity": destination.upper(),
        "toAirport": "",
        "fromDate": clean_date,
        "retDate": "",
        "airlines": airlines or ["OD"],  # Test carrier in sandbox
        "includeMultipleFareFamily": False,
        "currency": "USD",
        "requestSource": None,
    }

    response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def parse_datetime(dt_str: str) -> tuple[str, str]:
    """Converts 'YYYYMMDDHHMM' (e.g.

    '202610221200') to ('12:00', '22 Oct, Thursday')
    """
    if not dt_str or len(dt_str) < 12:
        return "--:--", "Unknown Date"
    dt = datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
    # dt.day formats the day without leading zeros cross-platform
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

        # 1. Times & Dates
        dep_raw = first_seg.get("depTime") or first_seg.get("departureTime", "")
        arr_raw = last_seg.get("arrTime") or last_seg.get("arrivalTime", "")
        dep_time, dep_date = parse_datetime(dep_raw)
        arr_time, arr_date = parse_datetime(arr_raw)

        # 2. Airports & Route
        dep_airport = first_seg.get("depAirport") or first_seg.get("departureAirport", "")
        arr_airport = last_seg.get("arrAirport") or last_seg.get("arrivalAirport", "")
        route_str = f"{dep_airport} - {arr_airport}"

        # 3. Safe Airline Carrier Extraction
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

        # 4. Total Price
        price_info = (
            item.get("adultPrice")
            or item.get("price")
            or item.get("totalPrice")
            or 0
        )
        currency = item.get("currency") or "USD"
        formatted_price = f"${float(price_info):.2f}" if currency == "USD" else f"{currency} {price_info}"

        # 5. Layover / Stop Details
        num_stops = len(segments) - 1
        layover_text = "Direct"
        if num_stops > 0:
            transfer_airport = first_seg.get("arrAirport") or first_seg.get("arrivalAirport", "")
            layover_text = f"{num_stops} stop in {transfer_airport}"

        # 6. Card Record
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

if __name__ == "__main__":
    res = search_atlas_flights("KUL", "BKI", "2026-10-22")
    print(f"Status: {res.get('status')} | Message: {res.get('msg')}")
    print(f"Found {len(res.get('routings', []))} flights.")

    flight_cards = extract_flight_ui_cards(res)
    print(f"Extracted {len(flight_cards)} flight cards for UI:")
    print(json.dumps(flight_cards, indent=2))