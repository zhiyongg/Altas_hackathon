from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from datetime import datetime, timedelta, timezone
import json
import math
import time
from agent import run_itinerary_agent
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# itineraryPlanner raises ValueError at import when API keys are missing from .env;
# the API server must still start, degrading /api/recalculate-route to haversine
# estimates only.
try:
    from itineraryPlanner import GOOGLE_MAPS_API_KEY, _safe_post, haversine_km, parse_dur
except Exception as e:
    GOOGLE_MAPS_API_KEY = None
    _safe_post = None
    haversine_km = None
    parse_dur = None
    logging.getLogger(__name__).warning(
        "itineraryPlanner helpers unavailable (%s); "
        "/api/recalculate-route will return haversine estimates only.", e
    )

app = FastAPI(title="Itinerary Planner API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # This perfectly mimics Flask's request logging!
    logger.info(f"{request.client.host} - \"{request.method} {request.url.path}\" {response.status_code} - {process_time:.2f}ms")
    return response

# Allow CORS for the Vite frontend (usually runs on port 3000 or 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TripRequest(BaseModel):
    user_request: str
    custom_messages: str = ""

class Waypoint(BaseModel):
    id: str
    lat: float
    lng: float

class RouteRecalcRequest(BaseModel):
    waypoints: list[Waypoint]
    mode: str = "TRANSIT"

class RouteLeg(BaseModel):
    fromId: str
    toId: str
    durationMinutes: float
    distanceMeters: float
    mode: Literal["walk", "subway", "bus", "taxi"]
    estimated: bool

class RouteRecalcResponse(BaseModel):
    legs: list[RouteLeg]

# ── Route recalculation helpers ──────────────────────────────────────────────
WALK_THRESHOLD_M     = 1200   # straight-line distance under which we route on foot
WALK_SPEED_M_PER_MIN = 80     # fallback walking speed
TRANSIT_SPEED_KMH    = 30     # fallback transit speed
TRANSIT_OVERHEAD_MIN = 10     # fallback wait/transfer overhead

_BUS_VEHICLES = {"BUS", "INTERCITY_BUS", "TROLLEYBUS"}

def _haversine_km_local(a: dict, b: dict) -> float:
    """Local great-circle distance so the fallback path never depends on itineraryPlanner."""
    lat1, lon1 = math.radians(a["latitude"]), math.radians(a["longitude"])
    lat2, lon2 = math.radians(b["latitude"]), math.radians(b["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    c = 2 * math.asin(math.sqrt(
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2))
    return 6371 * c

def _straight_meters(a: Waypoint, b: Waypoint) -> float:
    dist_fn = haversine_km if haversine_km is not None else _haversine_km_local
    return dist_fn({"latitude": a.lat, "longitude": a.lng},
                   {"latitude": b.lat, "longitude": b.lng}) * 1000

def _estimated_leg(a: Waypoint, b: Waypoint) -> RouteLeg:
    """Haversine fallback when the Routes API is unavailable for a pair."""
    dist = _straight_meters(a, b)
    if dist <= WALK_THRESHOLD_M:
        minutes = dist / WALK_SPEED_M_PER_MIN
        mode = "walk"
    else:
        minutes = (dist / 1000) / TRANSIT_SPEED_KMH * 60 + TRANSIT_OVERHEAD_MIN
        mode = "subway"
    return RouteLeg(fromId=a.id, toId=b.id,
                    durationMinutes=round(minutes, 1),
                    distanceMeters=round(dist),
                    mode=mode, estimated=True)

def _routed_leg(a: Waypoint, b: Waypoint, travel_mode: str) -> RouteLeg | None:
    """Single-pair Routes API call (reuses the planner's key + safe POST).
    Returns None on any failure so the caller can fall back to an estimate."""
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,"
                            "routes.legs.steps.travelMode,"
                            "routes.legs.steps.transitDetails.transitLine.vehicle.type",
    }
    payload = {
        "origin":      {"location": {"latLng": {"latitude": a.lat, "longitude": a.lng}}},
        "destination": {"location": {"latLng": {"latitude": b.lat, "longitude": b.lng}}},
        "travelMode": travel_mode,
    }
    if travel_mode == "TRANSIT":
        # Same departure-time convention as compute_route_matrix in itineraryPlanner
        payload["departureTime"] = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT09:00:00Z")

    resp = _safe_post(url, headers=headers, json_payload=payload)
    if not resp or resp.status_code != 200:
        logger.warning("computeRoutes failed (%s): %s",
                       resp.status_code if resp else "no response",
                       resp.text[:200] if resp else "")
        return None

    routes = resp.json().get("routes") or []
    if not routes:
        return None
    route = routes[0]
    duration_s = parse_dur(route.get("duration"))
    distance_m = route.get("distanceMeters", 0)
    if duration_s <= 0:
        return None

    # Map the API result onto the contract's mode enum
    if travel_mode == "WALK":
        mode = "walk"
    elif travel_mode == "DRIVE":
        mode = "taxi"
    else:
        vehicles = {
            step.get("transitDetails", {}).get("transitLine", {}).get("vehicle", {}).get("type")
            for leg in route.get("legs", []) for step in leg.get("steps", [])
            if step.get("transitDetails")
        }
        vehicles.discard(None)
        if not vehicles:
            mode = "walk"  # transit routing degenerated to a pure walking route
        elif vehicles <= _BUS_VEHICLES:
            mode = "bus"
        else:
            mode = "subway"

    return RouteLeg(fromId=a.id, toId=b.id,
                    durationMinutes=round(duration_s / 60, 1),
                    distanceMeters=distance_m,
                    mode=mode, estimated=False)

def build_route_legs(req: RouteRecalcRequest) -> list[RouteLeg]:
    """N-1 adjacent-pair legs; never raises for routing failures."""
    # Without the planner's key/helpers every leg is a haversine estimate
    helpers_available = all(x is not None for x in (GOOGLE_MAPS_API_KEY, _safe_post, parse_dur))
    legs: list[RouteLeg] = []
    for a, b in zip(req.waypoints, req.waypoints[1:]):
        travel_mode = "WALK" if _straight_meters(a, b) <= WALK_THRESHOLD_M \
                      else (req.mode or "TRANSIT").upper()
        leg = None
        if helpers_available:
            try:
                leg = _routed_leg(a, b, travel_mode)
            except Exception as e:
                logger.warning("Route leg %s->%s failed: %s", a.id, b.id, e)
        legs.append(leg or _estimated_leg(a, b))
    return legs

@app.post("/api/recalculate-route", response_model=RouteRecalcResponse)
def recalculate_route(req: RouteRecalcRequest):
    # Fewer than 2 waypoints is a no-op: respond 200 with an empty legs array
    if len(req.waypoints) < 2:
        return RouteRecalcResponse(legs=[])
    return RouteRecalcResponse(legs=build_route_legs(req))

@app.post("/api/generate")
async def generate_trip(req: TripRequest):
    logger.info(f"Received request: {req.user_request}")
    try:
        result_json = run_itinerary_agent(req.user_request, req.custom_messages)
        
        # Safely parse the returned JSON string back into a Python dict so FastAPI can serialize it properly
        try:
            return json.loads(result_json)
        except json.JSONDecodeError:
            logger.error(f"Agent did not return valid JSON: {result_json}")
            raise HTTPException(status_code=500, detail="Agent did not return valid JSON")
            
    except Exception as e:
        logger.error(f"Error generating trip: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
