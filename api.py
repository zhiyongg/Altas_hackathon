from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
from agent import run_itinerary_agent
import logging
from hotels import search_mock_hotels, get_hotel_ui_cards, _parse_room_options
from tools import get_hotel_prices_raw

from schemas import HotelSearchInput

# TODO: replace with real sponsored/featured hotel data (or a partner feed).
# search_mock_hotels() needs each dict to carry hotel_id, hotel_name, address,
# city, coordinates, and a dest_id matching the StayAPI dest_id being searched.
SPONSORED_MOCK_HOTELS: list[dict] = []

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


class HotelChangeRequest(BaseModel):
    dest_id: str
    dest_type: str = "CITY"
    checkin: str
    checkout: str
    adults: int = 2
    rooms: int = 1
    children: int = 0
    children_ages: list[int] | None = None
    # How many of the top (cheapest-first, after sort) searched hotels to
    # backfill with real per-room pricing via get_hotel_prices. Each one is
    # an extra StayAPI call, so keep this small.
    price_lookup_limit: int = 5


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)