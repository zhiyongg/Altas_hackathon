import os
import requests
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_MAPS_API_KEY:
    raise ValueError(
        "Missing GOOGLE_MAPS_API_KEY. Please set it in your .env file."
    )


def get_route_matrix(
    origins: list[dict],
    destinations: list[dict],
    travel_mode: str = "DRIVE",
    routing_preference: str = "TRAFFIC_AWARE",
) -> list[dict]:
    """
    Calculate travel distance and duration between multiple origins
    and destinations using Google Routes API Compute Route Matrix.

    Args:
        origins:
            List of locations, e.g.
            [
                {"lat": 3.1579, "lng": 101.7116},
                {"lat": 3.1491, "lng": 101.7136}
            ]

        destinations:
            List of locations, e.g.
            [
                {"lat": 3.1344, "lng": 101.6863},
                {"lat": 3.0833, "lng": 101.6833}
            ]

        travel_mode:
            "DRIVE", "WALK", "BICYCLE", or "TRANSIT"

        routing_preference:
            "TRAFFIC_AWARE" or "TRAFFIC_UNAWARE"

    Returns:
        List of route matrix elements containing:
        - originIndex
        - destinationIndex
        - duration
        - distanceMeters
        - condition
        - status
    """

    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"

    # Only request the fields needed for itinerary optimization
    field_mask = [
        "originIndex",
        "destinationIndex",
        "duration",
        "distanceMeters",
        "condition",
        "status",
    ]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(field_mask),
    }

    payload = {
        "origins": [
            {
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": origin["lat"],
                            "longitude": origin["lng"],
                        }
                    }
                }
            }
            for origin in origins
        ],
        "destinations": [
            {
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": destination["lat"],
                            "longitude": destination["lng"],
                        }
                    }
                }
            }
            for destination in destinations
        ],
        "travelMode": travel_mode,
        "routingPreference": routing_preference,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        raise Exception(
            f"API Error {response.status_code}: {response.text}"
        )

    # Compute Route Matrix returns a JSON array
    # containing one result for each origin × destination pair.
    results = response.json()

    # Clean data structure for itinerary optimizer
    route_matrix = []

    for route in results:
        route_matrix.append(
            {
                "originIndex": route.get("originIndex"),
                "destinationIndex": route.get("destinationIndex"),
                "duration": route.get("duration"),
                "distanceMeters": route.get("distanceMeters"),
                "condition": route.get("condition"),
                "status": route.get("status"),
            }
        )

    return route_matrix


# ---------------------------------------------------------------------------
# Example test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    origins = [
        {"lat": 3.053422, "lng": 101.670855},  # Bukit Jalil
        {"lat": 3.0833, "lng": 101.6833},      # Mid Valley
        {"lat": 3.1491, "lng": 101.7136},      # Pavilion KL
    ]

    destinations = [
        {"lat": 3.1344, "lng": 101.6863},      # KL Sentral
        {"lat": 3.1579, "lng": 101.7116},      # KLCC
        {"lat": 3.1689, "lng": 101.6840},      # Hartamas
    ]

    data = get_route_matrix(
        origins=origins,
        destinations=destinations,
        travel_mode="DRIVE",
        routing_preference="TRAFFIC_AWARE",
    )

    import json

    print(json.dumps(data, indent=2))

