#api.py

from pathlib import Path
import json
import logging
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Chat and Itinerary imports
from agent import run_itinerary_agent
from chat_agent import ChatEngine
from itinerary_repo import get_latest_itinerary, save_itinerary
from itineraryPlanner import TripConfig, build_itinerary

# Route helpers (imported with guard so server starts even without env keys)
try:
    from itineraryPlanner import (
        GOOGLE_MAPS_API_KEY as _GOOGLE_MAPS_API_KEY,
        _safe_post as _route_safe_post,
        parse_dur as _route_parse_dur,
    )
except Exception:
    _GOOGLE_MAPS_API_KEY = None
    _route_safe_post = None
    _route_parse_dur = None

import math as _math

def _local_haversine_km(a: dict, b: dict) -> float:
    """Haversine distance in km — local backup if itineraryPlanner import fails."""
    lat1, lon1 = _math.radians(a["latitude"]), _math.radians(a["longitude"])
    lat2, lon2 = _math.radians(b["latitude"]), _math.radians(b["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    c = 2 * _math.asin(_math.sqrt(
        _math.sin(dlat / 2) ** 2 +
        _math.cos(lat1) * _math.cos(lat2) * _math.sin(dlon / 2) ** 2))
    return 6371 * c

try:
    from itineraryPlanner import haversine_km as _haversine_km
except Exception:
    _haversine_km = _local_haversine_km

# Hotel, Tools, and Payment imports
from hotels import search_mock_hotels, get_hotel_ui_cards, _parse_room_options
from flights import get_flight_options
from activities import get_activity_options
from payment import create_trip_checkout_sessions, get_checkout_session_status
from tools import get_hotel_prices_raw
from schemas import (
    HotelSearchInput, HotelChangeRequest, CreateCheckoutSessionsRequest,
    FlightSearchRequest, FlightChangeApplyRequest, FlightOption,
    ActivitySearchRequest,
)
from sponsored_hotel import SPONSORED_MOCK_HOTELS, NOT_SPONSORED_MOCK_HOTELS, MOCK_ROOM_PRICES_RAW

# ==========================================
# Application Setup & Middleware
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "itinerary_output.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Itinerary Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - started) * 1000
    client = request.client.host if request.client else "unknown"
    logger.info(
        '%s - "%s %s" %s - %.2fms',
        client,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response

# ==========================================
# Pydantic Models: Route Recalculation
# ==========================================
class RouteWaypoint(BaseModel):
    id: str
    lat: float
    lng: float

class RecalculateRouteRequest(BaseModel):
    waypoints: list[RouteWaypoint]
    mode: str = "TRANSIT"

class RouteLeg(BaseModel):
    fromId: str
    toId: str
    durationMinutes: float
    distanceMeters: float
    mode: str          # "walk" | "subway" | "bus" | "taxi"
    estimated: bool

class RecalculateRouteResponse(BaseModel):
    legs: list[RouteLeg]

# ==========================================
# Pydantic Models
# ==========================================
# Trip & Chat Models
class TripRequest(BaseModel):
    user_request: str
    custom_messages: str = ""
    trip_config: Optional[dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str = "testing"
    trip_config: Optional[dict[str, Any]] = None
    output_path: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    itinerary: dict[str, Any] = Field(default_factory=dict)

# Hotel Models
class HotelPriceLookupRequest(BaseModel):
    checkin: str
    checkout: str
    adults: int
    rooms: int

# Payment Models
# NOTE: the member shape lives in schemas.TripMemberInput (used by
# CreateCheckoutSessionsRequest). There is deliberately no second copy here —
# the local duplicate that used to sit at this spot was never referenced, so
# editing it silently had no effect on the payment endpoint.

# ==========================================
# Chat Engine & State Helpers
# ==========================================
chat_engines: dict[str, ChatEngine] = {}

def _read_json(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No saved itinerary found. Generate one first.") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Saved itinerary_output.json is invalid JSON.") from exc

def _save_json(data: dict[str, Any], path: Path = OUTPUT_PATH) -> str:
    """Persist itinerary data to MongoDB when configured; otherwise save to a local JSON file."""
    if os.getenv("MONGODB_URI"):
        inserted_id = save_itinerary(data)
        logger.info("Saved itinerary to MongoDB with id=%s", inserted_id)
        return inserted_id

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return str(path)

def _planner_output(full_output: dict[str, Any]) -> dict[str, Any]:
    """Extract the shape expected by chat_agent from agent.py's full response."""
    planner_output = full_output.get("daily_itinerary")
    if planner_output is None and "days" in full_output:
        planner_output = full_output
    if not isinstance(planner_output, dict) or "days" not in planner_output:
        raise HTTPException(
            status_code=500,
            detail="Generated output does not contain daily_itinerary.days.",
        )
    return planner_output


def _normalize_itinerary_payload(full_output: dict[str, Any], updated_display: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return the exact same frontend contract for both initial and chat-updated itineraries.

    The internal chat layer stores a reduced planner-style display payload, but the
    frontend expects the larger ItineraryPlan schema from the original generation.
    This function preserves the original meta fields and swaps only the
    daily_itinerary.days content with the latest edited version.
    """
    base = dict(full_output or {})
    current_display = updated_display if isinstance(updated_display, dict) else _planner_output(base)

    normalized = {
        "trip_overview": base.get("trip_overview") or {
            "title": "",
            "origin_city": "",
            "destination_city": current_display.get("destination") or base.get("destination_city") or "",
            "start_date": current_display.get("start") or base.get("start_date") or "",
            "end_date": current_display.get("end") or base.get("end_date") or "",
            "total_days": max(len(current_display.get("days") or []), 0),
            "summary": "",
            "total_estimated_budget": {"amount": 0.0, "currency": "USD"},
        },
        "flights": base.get("flights") or [],
        "hotels": base.get("hotels") or [],
        "daily_itinerary": {
            "destination": current_display.get("destination") or base.get("destination_city") or "",
            "start": current_display.get("start") or base.get("start_date") or "",
            "end": current_display.get("end") or base.get("end_date") or "",
            "days": current_display.get("days") or [],
            "error": None,
        },
        "cost_breakdown": base.get("cost_breakdown") or {
            "flights": 0.0,
            "hotels": 0.0,
            "activities": 0.0,
            "food_dining": 0.0,
            "transportation": 0.0,
            "currency": "USD",
        },
        "travel_tips": base.get("travel_tips") or [],
    }

    # Preserve the original field names when they exist and keep the latest edited days.
    if "daily_itinerary" in base and isinstance(base["daily_itinerary"], dict):
        normalized["daily_itinerary"].update({
            key: value for key, value in base["daily_itinerary"].items() if key not in {"destination", "start", "end", "days"}
        })
    return normalized


def _build_trip_config(
    full_output: dict[str, Any], overrides: Optional[dict[str, Any]] = None
) -> TripConfig:
    """Build the chat planner config, using request values when available."""
    overrides = dict(overrides or {})
    overrides.pop("session_id", None)
    planner_output = _planner_output(full_output)
    overview = full_output.get("trip_overview") or {}
    days = planner_output.get("days") or []
    hotels = full_output.get("hotels") or []
    first_hotel = hotels[0] if hotels and isinstance(hotels[0], dict) else {}

    # 1. Scrape the fallback coordinates from the schedule
    fallback_lat = 0.0
    fallback_lng = 0.0
    if days and days[0].get("schedule"):
        for stop in days[0]["schedule"]:
            if stop.get("kind") == "hotel" and stop.get("location"):
                fallback_lat = stop["location"].get("latitude") or 0.0
                fallback_lng = stop["location"].get("longitude") or 0.0
                break

    # 2. Safely build the hotel config using the 'or' operator and the fallbacks
    generated_hotel = {
        "name": first_hotel.get("name") or "Trip hotel",
        "latitude": first_hotel.get("latitude") or fallback_lat,
        "longitude": first_hotel.get("longitude") or fallback_lng,
        "address": first_hotel.get("address") or "",
    }

    start_date = overrides.pop("start_date", None) or overview.get("start_date") or planner_output.get("start")
    end_date = overrides.pop("end_date", None) or overview.get("end_date") or planner_output.get("end")
    destination = (
        overrides.pop("destination", None)
        or overview.get("destination_city")
        or planner_output.get("destination")
    )
    if not destination or not start_date or not end_date:
        raise HTTPException(
            status_code=400,
            detail="trip_config must include destination, start_date, and end_date when they are missing from the generated output.",
        )

    defaults = {
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "arrival_datetime": f"{start_date}T08:00:00",
        "departure_datetime": f"{end_date}T22:00:00",
        "hotel": generated_hotel,
        "airport": {},
        "travel_style": "packed",
        "transport_mode": "TRANSIT",
        "group_size": 2,
        "budget": "medium",
        "check_in_time": "15:00",
        "check_out_time": "11:00",
        "selected_preferences": [],
        "preferences": {},
        "custom_vibe": "",
    }
    defaults.update(overrides)

    if not days:
        raise HTTPException(status_code=400, detail="The generated itinerary contains no days.")
    return TripConfig(**defaults)

def _get_chat_engine(
    session_id: str,
    full_output: dict[str, Any],
    trip_config: Optional[dict[str, Any]],
    reset: bool = False,
) -> ChatEngine:
    engine = chat_engines.get(session_id)
    if engine and engine.state is not None and not reset:
        return engine

    engine = ChatEngine(session_id=session_id, store_root=str(BASE_DIR / "sessions"))
    cfg = _build_trip_config(full_output, trip_config)
    if reset:
        engine.state = None
    engine.start_session_from_output(
        _planner_output(full_output),
        cfg=cfg,
    )
    chat_engines[session_id] = engine
    return engine

# ==========================================
# Endpoints: Trip Generation & Chat
# ==========================================
@app.post("/api/generate")
async def generate_trip(req: TripRequest):
    logger.info("Received generation request: %s", req.user_request)
    try:
        result_json = run_itinerary_agent(req.user_request, req.custom_messages)
        result = json.loads(result_json)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Agent returned a non-object JSON value.")

        if req.trip_config is not None:
            session_id = str(req.trip_config.get("session_id", "testing"))
            chat_engines.pop(session_id, None)
            engine = _get_chat_engine(session_id, result, req.trip_config, reset=True)

            # The hotel search provider (see the raw `data.hotels[]` shape)
            # never returns a check-in/check-out policy field — so cfg's
            # value (mocked to 15:00/11:00 when nothing overrides it) is the
            # ONLY real source of truth for it. Surface it here so the
            # frontend reads the exact value the planner scheduled around,
            # instead of maintaining its own separate guess.
            cfg = engine.state.cfg
            for hotel in result.get("hotels", []) or []:
                hotel.setdefault("stay_schedule", {})
                hotel["stay_schedule"].setdefault("check_in_time", cfg.check_in_time)
                hotel["stay_schedule"].setdefault("check_out_time", cfg.check_out_time)

        _save_json(result)
        return result
    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        logger.exception("Agent did not return valid JSON")
        raise HTTPException(status_code=500, detail="Agent did not return valid JSON.") from exc
    except Exception as exc:
        logger.exception("Error generating trip")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    try:
        path = Path(req.output_path) if req.output_path else OUTPUT_PATH
        full_output = _read_json(path)
        engine = _get_chat_engine(req.session_id, full_output, req.trip_config)
        response = engine.process_message(req.message)
        normalized_itinerary = _normalize_itinerary_payload(full_output, engine.state.display_itinerary)
        return ChatResponse(
            response=response,
            session_id=req.session_id,
            itinerary=normalized_itinerary,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error processing chat message")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/api/saved-itinerary")
async def saved_itinerary():
    if os.getenv("MONGODB_URI"):
        itinerary = get_latest_itinerary()
        if itinerary is None:
            raise HTTPException(status_code=404, detail="No saved itinerary found in MongoDB.")
        return itinerary
    return _read_json()

# ==========================================
# Endpoints: Hotel Modifications & Pricing
# ==========================================
def _backfill_card_price(card: dict, checkin: str, checkout: str, adults: int, rooms: int) -> bool:
    """Fetch per-room pricing for one hotel card and mutate it in place
    (selected_room/available_rooms). Returns True if pricing was found.

    Defined above both callers on purpose: /hotel/change used to inline a
    byte-for-byte copy of this body in its backfill loop, so any fix to one
    (e.g. the sort key, or which room is treated as selected) silently missed
    the other.

    Checks MOCK_ROOM_PRICES_RAW first — mock hotel_ids (sponsored-*, regular-*)
    don't exist in real StayAPI and would just error there — then falls back
    to get_hotel_prices_raw unchanged, so real hotel_ids keep working once
    StayAPI access is restored. Either way the raw dict is run through the
    same _parse_room_options() parser, untouched.
    """
    raw_prices = MOCK_ROOM_PRICES_RAW.get(card["hotel_id"])
    if raw_prices is None:
        raw_prices = get_hotel_prices_raw(card["hotel_id"], checkin, checkout, adults, rooms)
        if "error" in raw_prices:
            return False
    parsed_rooms = _parse_room_options(raw_prices, checkin, checkout)
    if not parsed_rooms:
        return False
    sorted_rooms = sorted(parsed_rooms, key=lambda r: r.total_price)
    card["selected_room"] = sorted_rooms[0].model_dump()
    card["available_rooms"] = [r.model_dump() for r in sorted_rooms[1:]]
    return True

@app.post("/hotel/change")
async def change_hotel(req: HotelChangeRequest):
    logger.info(f"Received hotel change request: dest_id={req.dest_id} {req.checkin}->{req.checkout}")
    try:
        # Fallback for old/hallucinated trips that have a Google Place ID instead of a Booking.com ID
        actual_dest_id = req.dest_id
        if actual_dest_id.startswith("ChIJ"):
            logger.warning(f"Intercepted legacy Google Place ID {actual_dest_id}. Falling back to Kota Kinabalu dest_id -2404760.")
            actual_dest_id = "-2404760"
            
        search_params = HotelSearchInput(
            dest_id=actual_dest_id,
            dest_type=req.dest_type,
            checkin=req.checkin,
            checkout=req.checkout,
            adults=req.adults,
            rooms=req.rooms,
            children=req.children,
            children_ages=req.children_ages,
        )

        # StayAPI is currently blocked, so /hotel/change is fully mocked:
        # search_mock_hotels over SPONSORED_MOCK_HOTELS + NOT_SPONSORED_MOCK_HOTELS
        # replaces the old get_hotel_ui_cards(search_params) call to the real
        # search endpoint. Parsing is unchanged (search_mock_hotels already
        # reads each item's price/coords/etc. the same way); the only fix
        # needed there was per-item is_sponsored instead of a hardcoded True,
        # so the combined list still splits correctly into featured/regular
        # on the frontend.
        all_mock_hotels = SPONSORED_MOCK_HOTELS + NOT_SPONSORED_MOCK_HOTELS
        hotels = search_mock_hotels(search_params, all_mock_hotels)
        cards = [h.model_dump() for h in hotels]

        # Eagerly backfill available_rooms on every card (unlike the old real-
        # StayAPI backfill loop, this doesn't need price_lookup_limit — it's a
        # MOCK_ROOM_PRICES_RAW dict lookup, not a network call). Without this,
        # each card's selected_room is already priced from search_mock_hotels
        # but available_rooms stays empty, so the frontend's "hasPrice" check
        # short-circuits before it ever calls fetchPriceFor — the ONLY thing
        # that currently populates available_rooms — and the "View N room
        # options" dropdown never has more than one room to show.
        for card in cards:
            _backfill_card_price(card, req.checkin, req.checkout, req.adults, req.rooms)

        return {"hotels": cards}

    except Exception as e:
        logger.error(f"Error searching hotels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hotel/{hotel_id}/prices")
async def get_hotel_price_for_room(hotel_id: str, req: HotelPriceLookupRequest):
    """On-demand pricing for a single hotel — used by the frontend when a
    card comes back from /hotel/change without a price (outside the eager
    backfill's price_lookup_limit)."""
    card = {"hotel_id": hotel_id}
    try:
        found = _backfill_card_price(card, req.checkin, req.checkout, req.adults, req.rooms)
    except Exception as e:
        logger.error(f"Error fetching price for hotel {hotel_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not found:
        raise HTTPException(status_code=404, detail="No room pricing available for this hotel and date range")

    return {"selected_room": card["selected_room"], "available_rooms": card["available_rooms"]}

# ==========================================
# Endpoints: Flight Search & Change
# ==========================================
@app.post("/flight/change")
async def change_flight(req: FlightSearchRequest):
    """Search flights via Atlas — no per-item backfill loop and no
    sponsored/featured list, unlike /hotel/change: Atlas returns full
    per-routing pricing in a single call."""
    logger.info(f"Received flight search: {req.origin}->{req.destination} {req.depart_date}")
    try:
        options = get_flight_options(
            origin=req.origin, destination=req.destination,
            depart_date=req.depart_date, return_date=req.return_date,
            adults=req.adults, children=req.children, infants=req.infants,
        )
        return {"flights": [f.model_dump() for f in options]}
    except Exception as e:
        logger.error(f"Error searching flights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flight/apply")
async def apply_flight_change(req: FlightChangeApplyRequest):
    """Apply a newly selected flight: shift arrival/departure times and
    re-run itinerary planning so the day's schedule recalculates around the
    new flight — unlike /hotel/change, which swaps the hotel card in place
    without regenerating the schedule."""
    path = Path(req.output_path) if req.output_path else OUTPUT_PATH
    full_output = _read_json(path)

    overrides = dict(req.trip_config or {})

    def _leg_dt(seg) -> str:
        return f"{seg.date}T{seg.time}:00"

    # A FlightOption is ONE direction, so only ONE end of the trip window moves.
    # This used to set arrival_datetime from flight.arrival AND
    # departure_datetime from flight.departure of the same option — but on an
    # outbound option `departure` is the take-off from the ORIGIN, which is
    # earlier than the arrival. classify_days then derived
    # end_time = departure_dt - 3h and collapsed the trip to a negative window.
    if req.flight.direction == "return":
        overrides["departure_datetime"] = _leg_dt(req.flight.departure)
    else:
        overrides["arrival_datetime"] = _leg_dt(req.flight.arrival)

    cfg = _build_trip_config(full_output, overrides)
    result = build_itinerary(cfg)

    if result.get("error") and not result.get("days"):
        raise HTTPException(status_code=502, detail=f"Replanning failed: {result['error']}")

    full_output["daily_itinerary"] = result

    # Replace only the leg that changed. Assigning a single-element list here
    # used to delete the other direction from the saved trip.
    legs = list(full_output.get("flights") or [])
    slot = 1 if req.flight.direction == "return" else 0
    while len(legs) <= slot:
        legs.append(None)
    legs[slot] = req.flight.model_dump()
    full_output["flights"] = [leg for leg in legs if leg is not None]

    _save_json(full_output, path)

    # Re-seed the chat session so subsequent chat edits build on the new schedule
    chat_engines.pop(req.session_id, None)
    _get_chat_engine(req.session_id, full_output, req.trip_config, reset=True)

    return _normalize_itinerary_payload(full_output)

# ==========================================
# Endpoints: Activity Search
# ==========================================
@app.post("/activity/search")
async def search_activities(req: ActivitySearchRequest):
    """Search Google Places for activities — powers the "Add Activity"
    modal. Reuses the same Places Text Search API tools.py's text_search
    @tool calls, via a dedicated raw+typed extraction pair (activities.py)
    following the same pattern as hotels.py/flights.py, rather than the
    agent tool's pre-formatted-string output."""
    logger.info(f"Received activity search: query={req.query!r} near=({req.latitude},{req.longitude})")
    try:
        options = get_activity_options(req.query, req.latitude, req.longitude)
        return {"activities": [a.model_dump() for a in options]}
    except Exception as e:
        logger.error(f"Error searching activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Endpoints: Payments
# ==========================================
@app.post("/payment/create-checkout-sessions")
async def create_checkout_sessions(req: CreateCheckoutSessionsRequest):
    logger.info(
        f"Creating checkout session(s) for trip={req.trip_id} "
        f"split={req.split} total={req.total_cost} members={len(req.members)}"
    )
    try:
        members = [m.model_dump() for m in req.members]
        sessions = create_trip_checkout_sessions(
            trip_id=req.trip_id,
            trip_destination=req.destination,
            total_cost=req.total_cost,
            members=members,
            split=req.split,
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            currency=req.currency,
        )
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error creating checkout sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/payment/session-status/{session_id}")
async def checkout_session_status(session_id: str):
    try:
        return get_checkout_session_status(session_id)
    except Exception as e:
        logger.error(f"Error retrieving checkout session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ItineraryItemInput(BaseModel):
    id: str
    time: str
    kind: str  # 'activity' | 'hotel' | 'meal' | 'flight'
    name: str
    location: Optional[dict[str, Any]] = None
    notes: Optional[str] = None
    # Every planner-emitted schedule entry carries a duration, and the chat
    # layer's re-scheduling reads it. Manually added entries used to omit it
    # entirely, so a later chat edit treated them as zero-length stops.
    duration_min: int = 60

class UpdateItineraryItemRequest(BaseModel):
    session_id: str
    day_number: int  # 1-indexed
    item: ItineraryItemInput
    delete: bool = False

@app.post("/api/itinerary/item", response_model=ChatResponse)
async def update_itinerary_item(req: UpdateItineraryItemRequest):
    engine = chat_engines.get(req.session_id)
    if engine is None or engine.state is None:
        raise HTTPException(
            status_code=404,
            detail="No active session for this trip — generate or chat with the itinerary before editing items directly.",
        )

    days = engine.state.display_itinerary.get("days", [])
    idx = req.day_number - 1
    if idx < 0 or idx >= len(days):
        raise HTTPException(status_code=400, detail="day_number out of range")

    schedule = days[idx].setdefault("schedule", [])

    if req.delete:
        schedule[:] = [s for s in schedule if s.get("_client_id") != req.item.id]
    else:
        entry = {
            "_client_id": req.item.id,
            "time": req.item.time,
            "kind": req.item.kind,
            "name": req.item.name,
            "location": req.item.location,
            "notes": req.item.notes,
            "duration_min": req.item.duration_min,
        }
        existing = next((i for i, s in enumerate(schedule) if s.get("_client_id") == req.item.id), None)
        if existing is not None:
            schedule[existing] = entry
        else:
            schedule.append(entry)
        schedule.sort(key=lambda s: s.get("time", ""))

    full_output = _read_json()
    normalized = _normalize_itinerary_payload(full_output, engine.state.display_itinerary)

    # Persist. Without these two lines the mutation lived only in the in-memory
    # engine: a page reload re-read the unchanged itinerary_output.json, and an
    # "undo" in chat restored a snapshot that never contained the manual edit.
    engine.store.save_snapshot(engine.state)
    _save_json(normalized)

    return ChatResponse(response="Updated itinerary item.", session_id=req.session_id, itinerary=normalized)

# ==========================================
# Endpoint: Route Recalculation
# ==========================================
_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

_BUS_TYPES = {"BUS", "INTERCITY_BUS", "TROLLEYBUS"}

def _map_transit_mode(step_type: str) -> str:
    """Map a Google Routes transit step type to the frontend mode string."""
    upper = (step_type or "").upper()
    if upper in _BUS_TYPES:
        return "bus"
    return "subway"

def _google_route_leg(
    origin: dict, destination: dict, travel_mode: str
) -> dict | None:
    """Call Google Routes API for a single origin→destination pair.
    Returns parsed leg dict on success, None on any failure."""
    if not _GOOGLE_MAPS_API_KEY or _route_safe_post is None:
        return None
    payload = {
        "origin": {"location": {"latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}}},
        "travelMode": travel_mode,
    }
    if travel_mode == "TRANSIT":
        payload["transitPreferences"] = {"routingPreference": "LESS_WALKING"}
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.legs.navigationInstruction,routes.legs.steps.navigationInstruction,routes.legs.steps.travelMode,routes.legs.steps.transitDetails,routes.legs.localizedValues",
    }
    resp = _route_safe_post(_ROUTES_URL, headers=headers, json_payload=payload)
    if resp is None or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    routes = data.get("routes")
    if not routes:
        return None
    legs = routes[0].get("legs")
    if not legs:
        return None
    return legs[0]

def _fallback_leg(wp_a: RouteWaypoint, wp_b: RouteWaypoint) -> RouteLeg:
    """Haversine-based fallback when the Routes API is unavailable or fails."""
    dist_km = _haversine_km(
        {"latitude": wp_a.lat, "longitude": wp_a.lng},
        {"latitude": wp_b.lat, "longitude": wp_b.lng},
    )
    dist_m = dist_km * 1000
    if dist_m <= 1200:
        dur_min = dist_m / 80.0  # 80 m/min walking
        mode = "walk"
    else:
        dur_min = (dist_km / 30.0) * 60 + 10  # 30 km/h + 10 min overhead
        mode = "subway"
    return RouteLeg(
        fromId=wp_a.id, toId=wp_b.id,
        durationMinutes=round(dur_min, 1),
        distanceMeters=round(dist_m, 0),
        mode=mode, estimated=True,
    )

@app.post("/api/recalculate-route", response_model=RecalculateRouteResponse)
async def recalculate_route(req: RecalculateRouteRequest):
    waypoints = req.waypoints
    if len(waypoints) < 2:
        return RecalculateRouteResponse(legs=[])

    travel_mode = (req.mode or "TRANSIT").upper()
    legs: list[RouteLeg] = []

    for i in range(len(waypoints) - 1):
        wp_a, wp_b = waypoints[i], waypoints[i + 1]
        dist_km = _haversine_km(
            {"latitude": wp_a.lat, "longitude": wp_a.lng},
            {"latitude": wp_b.lat, "longitude": wp_b.lng},
        )
        dist_m = dist_km * 1000

        # Short distance → always walk
        if dist_m <= 1200:
            leg_data = _google_route_leg(
                {"lat": wp_a.lat, "lng": wp_a.lng},
                {"lat": wp_b.lat, "lng": wp_b.lng},
                "WALK",
            )
            if leg_data:
                try:
                    dur_str = leg_data.get("localizedValues", {}).get("duration", {}).get("text", "")
                    dur_min = _route_parse_dur(dur_str) if _route_parse_dur else 0
                    if dur_min <= 0:
                        dur_min = dist_m / 80.0
                    dist_val = leg_data.get("localizedValues", {}).get("distance", {}).get("text", "")
                    # Parse distance text like "1.2 km" or "800 m"
                    import re as _re
                    dist_match = _re.search(r"([\d.]+)\s*(km|m)", str(dist_val))
                    if dist_match:
                        real_dist = float(dist_match.group(1))
                        if dist_match.group(2) == "km":
                            real_dist *= 1000
                    else:
                        real_dist = dist_m
                    legs.append(RouteLeg(
                        fromId=wp_a.id, toId=wp_b.id,
                        durationMinutes=round(dur_min, 1),
                        distanceMeters=round(real_dist, 0),
                        mode="walk", estimated=False,
                    ))
                    continue
                except Exception:
                    pass
            # Fallback for walk
            dur_min = dist_m / 80.0
            legs.append(RouteLeg(
                fromId=wp_a.id, toId=wp_b.id,
                durationMinutes=round(dur_min, 1),
                distanceMeters=round(dist_m, 0),
                mode="walk", estimated=True,
            ))
            continue

        # Longer distance → use requested travel mode
        leg_data = _google_route_leg(
            {"lat": wp_a.lat, "lng": wp_a.lng},
            {"lat": wp_b.lat, "lng": wp_b.lng},
            travel_mode,
        )
        if leg_data:
            try:
                dur_str = leg_data.get("localizedValues", {}).get("duration", {}).get("text", "")
                dur_min = _route_parse_dur(dur_str) if _route_parse_dur else 0
                if dur_min <= 0:
                    dur_min = (dist_km / 30.0) * 60 + 10

                dist_val = leg_data.get("localizedValues", {}).get("distance", {}).get("text", "")
                import re as _re
                dist_match = _re.search(r"([\d.]+)\s*(km|m)", str(dist_val))
                if dist_match:
                    real_dist = float(dist_match.group(1))
                    if dist_match.group(2) == "km":
                        real_dist *= 1000
                else:
                    real_dist = dist_m

                # Determine mode from steps
                steps = leg_data.get("steps", [])
                mode = "subway"  # default for transit
                if travel_mode == "DRIVE":
                    mode = "taxi"
                else:
                    for step in steps:
                        step_mode = (step.get("travelMode") or "").upper()
                        transit = step.get("transitDetails", {})
                        vehicle = (transit.get("transitLine", {}).get("vehicle", {}).get("type") or "").upper() if transit else ""
                        if vehicle in _BUS_TYPES or step_mode == "BUS":
                            mode = "bus"
                            break
                        elif step_mode == "WALK":
                            continue  # skip walk steps for mode detection
                        elif step_mode in ("SUBWAY", "RAIL", "TRAM", "FERRY"):
                            mode = "subway"
                            break

                legs.append(RouteLeg(
                    fromId=wp_a.id, toId=wp_b.id,
                    durationMinutes=round(dur_min, 1),
                    distanceMeters=round(real_dist, 0),
                    mode=mode, estimated=False,
                ))
                continue
            except Exception:
                pass

        # Fallback for transit
        legs.append(_fallback_leg(wp_a, wp_b))

    return RecalculateRouteResponse(legs=legs)

# ==========================================
# Run Server
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)