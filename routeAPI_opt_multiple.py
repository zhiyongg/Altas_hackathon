import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

BASE_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

FIELD_MASK = [
    "routes.duration",
    "routes.distanceMeters",
    "routes.polyline.encodedPolyline",
    "routes.description",
    "routes.legs",
    "routes.optimizedIntermediateWaypointIndex",
    "routes.localizedValues",
]


def optimize_route(
    origin: dict,
    destination: dict,
    intermediates: list[dict],
    travel_mode: str = "DRIVE",
    optimize_order: bool = True,
    routing_preference: str = "TRAFFIC_AWARE",  # or 'TRAFFIC_UNAWARE'
) -> dict:
    """
    Compute an optimized multi-stop route using the Google Routes API.

    Args:
        origin: Start point, e.g. {"lat": 3.0534, "lng": 101.6708}
        destination: End point, e.g. {"lat": 3.1491, "lng": 101.7136}
        intermediates: List of stops in between, e.g.
            [{"lat": 3.1579, "lng": 101.7116}, {"lat": 3.1344, "lng": 101.6863}]
        travel_mode: 'DRIVE', 'WALK', 'BICYCLE', 'TRANSIT', 'TWO_WHEELER'
        optimize_order: If True, API reorders intermediates for shortest route.
        routing_preference: 'TRAFFIC_AWARE', 'TRAFFIC_UNAWARE', 'TRAFFIC_AWARE_OPTIMAL'

    Returns:
        Raw JSON response as a dict.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(FIELD_MASK),
    }

    payload = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}
            }
        },
        "intermediates": [
            {
                "location": {
                    "latLng": {"latitude": stop["lat"], "longitude": stop["lng"]}
                }
            }
            for stop in intermediates
        ],
        "travelMode": travel_mode,
        "routingPreference": routing_preference,
        "computeAlternativeRoutes": False,
        "optimizeWaypointOrder": optimize_order,
    }

    response = requests.post(BASE_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_optimize_5_stops_kl():
    """Test 2: 5 intermediate stops across KL — tests larger optimization."""
    print("\n>>> TEST 2: Optimize 5 intermediate stops across KL")

    origin = {"lat": 3.053422, "lng": 101.670855}       # Bukit Jalil
    destination = {"lat": 3.2037, "lng": 101.7086}       # Mont Kiara
    intermediates = [
        {"lat": 3.0833, "lng": 101.6833},                # Mid Valley
        {"lat": 3.1491, "lng": 101.7136},                # Pavilion KL
        {"lat": 3.1344, "lng": 101.6863},                # KL Sentral
        {"lat": 3.1579, "lng": 101.7116},                # KLCC
        {"lat": 3.1689, "lng": 101.6840},                # Hartamas
    ]

    result = optimize_route(origin, destination, intermediates)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY is not set. Please add it to your .env file.")
        exit(1)

    print("=" * 60)
    print("  Google Routes API - Multi-Stop Route Optimization Tests")
    print("=" * 60)
    
    """Test 2: 5 intermediate stops across KL — tests larger optimization."""
    print("\n>>> TEST 2: Optimize 5 intermediate stops across KL")

    origin = {"lat": 3.053422, "lng": 101.670855}       # Bukit Jalil
    destination = {"lat": 3.2037, "lng": 101.7086}       # Mont Kiara
    intermediates = [
        {"lat": 3.0833, "lng": 101.6833},                # Mid Valley
        {"lat": 3.1491, "lng": 101.7136},                # Pavilion KL
        {"lat": 3.1344, "lng": 101.6863},                # KL Sentral
        {"lat": 3.1579, "lng": 101.7116},                # KLCC
        {"lat": 3.1689, "lng": 101.6840},                # Hartamas
    ]

    result = optimize_route(origin, destination, intermediates)
    print(json.dumps(result, indent=2))
