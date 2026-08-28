import json

from schemas import Hotel, StaySchedule, RoomOption, HotelSearchInput
from tools import search_hotels_raw

def _parse_review_and_image(item: dict) -> tuple[float | None, int | None, str | None]:
    """Shared helper: pull guest rating, review count, and a photo URL out of a
    StayAPI/Booking.com-shaped hotel dict, trying the field names StayAPI is
    known to use plus a few common aliases."""
    rating_raw = item.get("review_score") or item.get("rating") or item.get("reviewScore")
    try:
        rating = round(float(rating_raw), 1) if rating_raw is not None else None
    except (ValueError, TypeError):
        rating = None

    count_raw = item.get("review_nr") or item.get("review_count") or item.get("reviewsCount")
    try:
        review_count = int(count_raw) if count_raw is not None else None
    except (ValueError, TypeError):
        review_count = None

    image_url = (
        item.get("image_url")
        or item.get("main_photo_url")
        or item.get("max_photo_url")
        or item.get("image")
        or item.get("photo_url")
    )

    return rating, review_count, image_url


def search_mock_hotels(
    input: HotelSearchInput,
    mock_hotels: list[dict],
    dest_coords: tuple[float, float] | None = None,
    radius_km: float = 20.0,
) -> list[Hotel]:
    """Search through predefined hotel data and return filtered Hotel objects.

    Works like the real search flow (search_hotels + _parse_hotels_from_search)
    but operates on locally provided mock data instead of hitting StayAPI.

    Each mock hotel dict should carry:
    - ``hotel_id``, ``hotel_name``, ``address``, ``city``, ``coordinates``
      (same field names as the StayAPI /v1/booking/search response)
    - ``dest_id`` — the Booking.com destination this hotel belongs to,
      used to filter against ``input.dest_id``

    Filtering logic:
    1. **dest_id** — only hotels whose ``dest_id`` matches ``input.dest_id``
       are included.
    2. **Proximity radius** — when *dest_coords* ``(lat, lon)`` is provided,
       hotels with coordinates are further filtered to those within
       *radius_km* km of the destination centre (Haversine). Hotels without
       coordinates are kept (can't prove they're outside).
    3. **Pagination** — ``input.offset`` and ``input.rows_per_page`` are
       honoured just like the real API.

    Args:
        input: HotelSearchInput with dest_id, dates, adults, etc.
        mock_hotels: List of hotel dicts in StayAPI search response format.
        dest_coords: Optional ``(lat, lon)`` of the destination centre for
            proximity filtering (similar to Google Nearby Search radius).
        radius_km: Search radius in km (default 20).

    Returns:
        List of Hotel schema objects with stay_schedule populated.
        selected_room is a placeholder — call get_hotel_prices + backfill_hotel_rooms
        to fill in room-level pricing.
    """
    from datetime import datetime as _dt

    checkin = input.checkin
    checkout = input.checkout
    try:
        total_nights = (
            _dt.strptime(checkout, "%Y-%m-%d") - _dt.strptime(checkin, "%Y-%m-%d")
        ).days
    except (ValueError, TypeError):
        total_nights = 1
    if total_nights < 1:
        total_nights = 1

    # --- 1. Filter by dest_id ---
    # Each mock hotel dict declares which destination it belongs to.
    dest_id = str(input.dest_id)
    filtered = [
        h for h in mock_hotels
        if isinstance(h, dict)
        and str(h.get("dest_id", "")) == dest_id
    ]

    # --- 2. Optional proximity filter (radius from destination centre) ---
    # Mimics Google Nearby Search: keep hotels within radius_km of dest_coords.
    if dest_coords is not None:
        dest_lat, dest_lon = dest_coords
        within_radius: list[dict] = []
        for h in filtered:
            coords = h.get("coordinates") or h.get("location") or h.get("geo") or {}
            lat = coords.get("latitude") or coords.get("lat")
            lon = coords.get("longitude") or coords.get("lon") or coords.get("lng")
            if lat is not None and lon is not None:
                try:
                    dist = _haversine_km(dest_lat, dest_lon, float(lat), float(lon))
                    if dist <= radius_km:
                        within_radius.append(h)
                except (ValueError, TypeError):
                    within_radius.append(h)
            else:
                # No coordinates — keep it (can't prove it's outside)
                within_radius.append(h)
        filtered = within_radius

    # --- 3. Pagination (same semantics as the real API) ---
    offset = input.offset
    limit = input.rows_per_page
    filtered = filtered[offset : offset + limit]

    # --- 4. Build Hotel objects (same shape as _parse_hotels_from_search) ---
    hotels: list[Hotel] = []
    for item in filtered:
        if not isinstance(item, dict):
            continue

        hotel_id = str(item.get("hotel_id") or item.get("id") or "")
        if not hotel_id:
            continue

        coords = (
            item.get("coordinates")
            or item.get("location")
            or item.get("geo")
            or {}
        )

        sr = item.get("star_rating") or item.get("stars") or item.get("class")
        try:
            star_rating = int(sr) if sr is not None else None
        except (ValueError, TypeError):
            star_rating = None

        rating, review_count, image_url = _parse_review_and_image(item)

        hotels.append(Hotel(
            hotel_id=hotel_id,
            name=str(item.get("hotel_name") or item.get("name") or ""),
            address=item.get("address") or item.get("hotel_address"),
            city=item.get("city") or item.get("city_name"),
            latitude=coords.get("latitude") or coords.get("lat") or item.get("latitude"),
            longitude=coords.get("longitude") or coords.get("lon") or coords.get("lng") or item.get("longitude"),
            star_rating=star_rating,
            rating=rating,
            review_count=review_count,
            image_url=image_url,
            is_sponsored=True,  # everything in search_mock_hotels comes from the sponsored/featured list
            dest_id=str(input.dest_id),
            dest_type=input.dest_type,
            stay_schedule=StaySchedule(
                check_in_date=checkin,
                check_out_date=checkout,
                total_nights=total_nights,
            ),
            selected_room=RoomOption(
                room_name="",
                max_occupancy=0,
                price_per_night=0.0,
                total_price=0.0,
            ),
        ))

    return hotels


def backfill_hotel_rooms(
    hotels: list[Hotel],
    room_data: dict[str, list[RoomOption]],
) -> list[Hotel]:
    """Backfill room-level pricing into Hotel objects after get_hotel_prices.

    After search_hotels, every Hotel has a placeholder selected_room
    (room_name="", zeros for prices). This function enriches those hotels
    with real RoomOption data obtained from get_hotel_prices.

    Usage example::

        # 1. Search hotels (placeholder rooms)
        hotels = _parse_hotels_from_search(raw, checkin, checkout)

        # 2. Fetch room prices for each hotel you care about
        room_data = {}
        for hotel in hotels:
            raw_prices = _request("/v1/booking/hotel/prices", ...)
            room_data[hotel.hotel_id] = _parse_room_options(raw_prices, checkin, checkout)

        # 3. Backfill — sets selected_room to cheapest room, rest → available_rooms
        hotels = backfill_hotel_rooms(hotels, room_data)

    Args:
        hotels: List of Hotel objects from _parse_hotels_from_search or
                search_mock_hotels (with placeholder selected_room).
        room_data: Mapping of hotel_id → list[RoomOption] from _parse_room_options.

    Returns:
        Updated list of Hotel objects with selected_room set to the cheapest
        room and all remaining rooms in available_rooms. Hotels without a
        matching entry in room_data are returned unchanged.
    """
    updated: list[Hotel] = []
    for hotel in hotels:
        rooms = room_data.get(hotel.hotel_id)
        if not rooms:
            updated.append(hotel)
            continue

        # Cheapest room becomes selected_room; the rest go to available_rooms
        sorted_rooms = sorted(rooms, key=lambda r: r.total_price)
        selected = sorted_rooms[0]
        available = sorted_rooms[1:]

        # BUG FIX: this used to rebuild `Hotel(...)` from scratch and only
        # copied 7 of the ~13 fields, silently dropping rating, review_count,
        # image_url, is_sponsored (resets to False -> breaks Featured/All
        # Options split), dest_id and dest_type (breaks "Change
        # Accommodation" re-search). model_copy(update=...) keeps every
        # other field as-is and only touches the two we're actually changing.
        updated.append(hotel.model_copy(update={
            "selected_room": selected,
            "available_rooms": available,
        }))

    return updated

def _parse_room_options(raw: dict, checkin: str, checkout: str) -> list[RoomOption]:
    """Parse room data from StayAPI prices response into RoomOption objects.
    Calculates total_price from nightly rate x number of nights."""
    from datetime import datetime as _dt
    try:
        nights = (_dt.strptime(checkout, "%Y-%m-%d") - _dt.strptime(checkin, "%Y-%m-%d")).days
    except (ValueError, TypeError):
        nights = 1
    if nights < 1:
        nights = 1

    # Find room blocks — could be nested under "data", "rooms", "room_blocks", etc.
    blocks = raw
    if isinstance(raw.get("data"), (dict, list)):
        blocks = raw["data"]
    if isinstance(blocks, dict):
        for key in ("rooms", "room_blocks", "blocks", "roomTypes", "room_types"):
            if key in blocks:
                blocks = blocks[key]
                break
    if isinstance(blocks, dict):
        blocks = list(blocks.values()) if all(isinstance(v, dict) for v in blocks.values()) else []
    if not isinstance(blocks, list):
        return []

    rooms: list[RoomOption] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue

        room_name = (block.get("room_name") or block.get("room_type")
                     or block.get("name") or block.get("roomName")
                     or block.get("block_name") or "Room")

        # --- price ---
        price_raw = (block.get("price") or block.get("price_per_night")
                     or block.get("nightly_price") or block.get("amount") or 0)
        if isinstance(price_raw, dict):
            p = price_raw.get("amount") or price_raw.get("value") or 0
            cur = price_raw.get("currency") or "USD"
        else:
            try:
                p = float(price_raw)
            except (ValueError, TypeError):
                p = 0.0
            cur = block.get("currency") or "USD"

        nightly = float(p)

        # total — prefer explicit total, else nightly * nights
        total_raw = block.get("total_price") or block.get("total")
        if total_raw is not None:
            if isinstance(total_raw, dict):
                total = float(total_raw.get("amount") or total_raw.get("value") or nightly * nights)
            else:
                try:
                    total = float(total_raw)
                except (ValueError, TypeError):
                    total = nightly * nights
        else:
            total = nightly * nights

        occupancy = (block.get("max_occupancy") or block.get("max_persons")
                     or block.get("max_guests") or block.get("occupancy")
                     or block.get("max_occupancy_persons") or 2)

        # --- breakfast ---
        breakfast_raw = (block.get("breakfast_included") or block.get("has_breakfast")
                         or block.get("breakfast") or block.get("meal_plan") or "")
        if isinstance(breakfast_raw, bool):
            breakfast = breakfast_raw
        elif isinstance(breakfast_raw, str):
            breakfast = "breakfast" in breakfast_raw.lower() and breakfast_raw.lower() not in ("no", "false", "0", "")
        else:
            breakfast = bool(breakfast_raw)

        # --- refundable ---
        refund_raw = (block.get("is_refundable") or block.get("refundable")
                      or block.get("free_cancellation") or False)
        if isinstance(refund_raw, str):
            refund = refund_raw.lower() in ("true", "yes", "1", "free")
        else:
            refund = bool(refund_raw)

        cancel = (block.get("cancellation_policy") or block.get("cancellation")
                  or block.get("cancel_policy"))
        if isinstance(cancel, dict):
            cancel = cancel.get("description") or cancel.get("label") or str(cancel)

        try:
            occupancy = int(occupancy)
        except (ValueError, TypeError):
            occupancy = 2

        rooms.append(RoomOption(
            room_name=str(room_name),
            max_occupancy=occupancy,
            price_per_night=round(nightly, 2),
            total_price=round(total, 2),
            currency=str(cur),
            breakfast_included=breakfast,
            is_refundable=refund,
            cancellation_policy=str(cancel) if cancel else None,
        ))

    return rooms

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres between two (lat, lon) points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def extract_hotel_ui_cards(
    res_data: dict, checkin: str, checkout: str, dest_id: str | None = None, dest_type: str | None = None
) -> list[Hotel]:
    """Parse hotel search results into Hotel objects with StaySchedule populated.

    Returns Pydantic Hotel objects, NOT dicts/cards — selected_room is a zeroed
    placeholder here. Call get_hotel_ui_cards() instead if you want plain
    dict cards ready to serialize to the frontend; use backfill_hotel_rooms()
    first if you also want real per-room pricing filled in from get_hotel_prices.
    dest_id/dest_type are the destination this search was run against - StayAPI's
    per-hotel search results don't carry it, so it's passed in from the caller
    and stamped onto every Hotel so the frontend can re-search this destination
    later (e.g. for Change Accommodation).
    """
    from datetime import datetime as _dt
    try:
        total_nights = (_dt.strptime(checkout, "%Y-%m-%d") - _dt.strptime(checkin, "%Y-%m-%d")).days
    except (ValueError, TypeError):
        total_nights = 1
    if total_nights < 1:
        total_nights = 1

    # BUG FIX: StayAPI actually returns {"data": {"hotels": [...]}} — the old
    # chain here (`res_data.get("data")`) grabbed the *dict* wrapper, not the
    # list inside it, so `isinstance(results_list, list)` failed and this
    # silently returned [] every time, even on a successful search with real
    # results (confirmed via raw StayAPI response logging). Drill into common
    # nested keys the same way _parse_room_options already does for prices.
    results_list = res_data.get("results") or res_data.get("data") or res_data.get("hotels") or []
    if isinstance(results_list, dict):
        for key in ("hotels", "results", "hotel_list", "properties", "items"):
            if key in results_list:
                results_list = results_list[key]
                break
    if not isinstance(results_list, list):
        return []

    hotels: list[Hotel] = []
    for item in results_list:
        if not isinstance(item, dict):
            continue

        hotel_id = str(item.get("hotel_id") or item.get("id") or "")
        if not hotel_id:
            continue

        coords = (item.get("coordinates") or item.get("location")
                  or item.get("geo") or {})

        # Try to parse star_rating as int
        sr = item.get("star_rating") or item.get("stars") or item.get("class")
        try:
            star_rating = int(sr) if sr is not None else None
        except (ValueError, TypeError):
            star_rating = None

        rating, review_count, image_url = _parse_review_and_image(item)

        hotels.append(Hotel(
            hotel_id=hotel_id,
            name=str(item.get("hotel_name") or item.get("name") or ""),
            address=item.get("address") or item.get("hotel_address"),
            city=item.get("city") or item.get("city_name"),
            # BUG FIX: this StayAPI response puts latitude/longitude at the
            # top level of each hotel dict, not nested under coordinates/
            # location/geo — fall back to the top-level keys so pins aren't
            # silently None.
            latitude=coords.get("latitude") or coords.get("lat") or item.get("latitude"),
            longitude=coords.get("longitude") or coords.get("lon") or coords.get("lng") or item.get("longitude"),
            star_rating=star_rating,
            rating=rating,
            review_count=review_count,
            image_url=image_url,
            is_sponsored=False,  # real StayAPI results, not the featured/mock list
            dest_id=dest_id,
            dest_type=dest_type,
            stay_schedule=StaySchedule(
                check_in_date=checkin,
                check_out_date=checkout,
                total_nights=total_nights,
            ),
            selected_room=RoomOption(
                room_name="",
                max_occupancy=0,
                price_per_night=0.0,
                total_price=0.0,
            ),
            available_rooms=[],
        ))

    return hotels


def get_hotel_ui_cards(params: HotelSearchInput) -> list[dict]:
    """Search StayAPI for real hotels and shape them into UI-ready card dicts.

    This is the function REST endpoints (e.g. api.py's /hotel/change) should
    call — it wraps search_hotels_raw() + extract_hotel_ui_cards() so callers
    get a plain list[dict] matching the Hotel schema, not raw StayAPI JSON
    and not a list of Pydantic Hotel objects.
    """
    raw = search_hotels_raw(params)
    if "error" in raw:
        return []
    hotels = extract_hotel_ui_cards(raw, params.checkin, params.checkout, str(params.dest_id), params.dest_type)
    return [h.model_dump() for h in hotels]