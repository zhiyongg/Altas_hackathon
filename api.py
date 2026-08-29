from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
from agent import run_itinerary_agent
import logging
from hotels import search_mock_hotels, get_hotel_ui_cards, _parse_room_options
from payment import create_trip_checkout_sessions, get_checkout_session_status


from tools import get_hotel_prices_raw

from schemas import HotelSearchInput, HotelChangeRequest, CreateCheckoutSessionsRequest
from sponsored_hotel import SPONSORED_MOCK_HOTELS

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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


@app.post("/hotel/change")
async def change_hotel(req: HotelChangeRequest):
    logger.info(f"Received hotel change request: dest_id={req.dest_id} {req.checkin}->{req.checkout}")
    try:
        search_params = HotelSearchInput(
            dest_id=req.dest_id,
            dest_type=req.dest_type,
            checkin=req.checkin,
            checkout=req.checkout,
            adults=req.adults,
            rooms=req.rooms,
            children=req.children,
            children_ages=req.children_ages,
        )

        # Sponsored/featured cards from local mock data (see SPONSORED_MOCK_HOTELS above).
        sponsored = search_mock_hotels(search_params, SPONSORED_MOCK_HOTELS)

        # Real hotels from StayAPI, shaped into Hotel-schema dicts (not raw JSON, not Hotel objects).
        searched_cards = get_hotel_ui_cards(search_params)

        # Backfill real room pricing for a capped number of results so we don't
        # fire off one StayAPI call per hotel on every request.
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

        # Single flat list — each hotel already carries is_sponsored, so the
        # frontend splits "Featured" vs "All Options" the same way it did
        # against the old local stayOptionsList mock.
        return {"hotels": [h.model_dump() for h in sponsored] + searched_cards}

    except Exception as e:
        logger.error(f"Error searching hotels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class HotelPriceLookupRequest(BaseModel):
    checkin: str
    checkout: str
    adults: int
    rooms: int


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
    (selected_room/available_rooms). Returns True if pricing was found.
    Shared by /hotel/change's eager batch backfill and the on-demand
    single-hotel endpoint below, so both use identical parsing/sorting."""
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


class TripMemberInput(BaseModel):
    id: str
    name: str
    isCurrentUser: bool = False


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)