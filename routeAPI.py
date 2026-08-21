import os
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


def compute_route(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    travel_mode: str = "DRIVE",  # Options: 'DRIVE', 'WALK', 'BICYCLE', 'TRANSIT', 'TWO_WHEELER'
) -> dict:
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    # Define the fields you want to get back to save costs/bandwidth
    field_mask = [
        "routes.duration",
        "routes.distanceMeters",
        "routes.polyline.encodedPolyline",
        "routes.description",
    ]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(field_mask),
    }

    payload = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin_lat, "longitude": origin_lng}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": dest_lat, "longitude": dest_lng}
            }
        },
        "travelMode": travel_mode,
        "routingPreference": "TRAFFIC_AWARE",  # or 'TRAFFIC_UNAWARE'
        "computeAlternativeRoutes": False,
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    data = response.json()
    routes = data.get("routes", [])

    if not routes:
        return {"error": "No route found"}

    best_route = routes[0]

    # Convert duration (e.g. "820s") to integer minutes
    raw_duration = best_route.get("duration", "0s").replace("s", "")
    duration_mins = round(int(raw_duration) / 60)

    # Convert distance in meters to kilometers
    distance_km = round(best_route.get("distanceMeters", 0) / 1000, 2)

    return {
        "duration_minutes": duration_mins,
        "distance_km": distance_km,
        "polyline": best_route.get("polyline", {}).get("encodedPolyline"),
    }


# Example: Bukit Jalil to Pavilion KL
if __name__ == "__main__":
    route_info = compute_route(
        origin_lat=3.053422,
        origin_lng=101.670855,
        dest_lat=3.1491,
        dest_lng=101.7136,
        travel_mode="DRIVE",
    )
    print(route_info)