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
from itineraryPlanner import TripConfig

# Hotel, Tools, and Payment imports
from hotels import search_mock_hotels, get_hotel_ui_cards, _parse_room_options
from payment import create_trip_checkout_sessions, get_checkout_session_status
from tools import get_hotel_prices_raw
from schemas import HotelSearchInput, HotelChangeRequest, CreateCheckoutSessionsRequest
from sponsored_hotel import SPONSORED_MOCK_HOTELS

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
class TripMemberInput(BaseModel):
    id: str
    name: str
    isCurrentUser: bool = False

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

        _save_json(result)
        # Start a fresh chat session from exactly the output just generated when
        # the caller supplies config details for the chat layer.
        if req.trip_config is not None:
            session_id = str(req.trip_config.get("session_id", "testing"))
            chat_engines.pop(session_id, None)
            _get_chat_engine(session_id, result, req.trip_config, reset=True)
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

        sponsored = search_mock_hotels(search_params, SPONSORED_MOCK_HOTELS)
        searched_cards = get_hotel_ui_cards(search_params)

        for card in searched_cards[: req.price_lookup_limit]:
            raw_prices = get_hotel_prices_raw(
                card["hotel_id"], req.checkin, req.checkout, req.adults, req.rooms
            )
            if "error" in raw_prices:
                continue
            rooms = _parse_room_options(raw_prices, req.checkin, req.checkout)
            if not rooms:
                continue
            sorted_rooms = sorted(rooms, key=lambda r: r.total_price)
            card["selected_room"] = sorted_rooms[0].model_dump()
            card["available_rooms"] = [r.model_dump() for r in sorted_rooms[1:]]

        return {"hotels": [h.model_dump() for h in sponsored] + searched_cards}

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

def _backfill_card_price(card: dict, checkin: str, checkout: str, adults: int, rooms: int) -> bool:
    """Fetch real per-room pricing for one hotel card and mutate it in place
    (selected_room/available_rooms). Returns True if pricing was found."""
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

# ==========================================
# Run Server
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)