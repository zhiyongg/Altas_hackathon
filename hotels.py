import json

from schemas import Hotel, StaySchedule, RoomOption, HotelSearchInput
from tools import search_hotels_raw, get_hotel_prices_raw


def _parse_review_and_image(item: dict) -> tuple[float | None, int | None, str | None]:
    """Pull guest rating, review count, and a photo URL out of a
    StayAPI/Booking.com-shaped hotel dict. Real /v1/booking/search results
    nest rating as {"rating": {"score": 8.3, "review_count": 3634, ...}} —
    treating that dict as a bare number (float(rating_raw) on a dict) raises
    TypeError, which the old except clause silently swallowed, dropping
    every real rating. This checks for the dict shape first."""
    rating_field = item.get("rating")
    if isinstance(rating_field, dict):
        rating_raw = rating_field.get("score")
        count_raw = rating_field.get("review_count")
    else:
        rating_raw = rating_field or item.get("review_score") or item.get("reviewScore")
        count_raw = item.get("review_nr") or item.get("review_count") or item.get("reviewsCount")

    try:
        rating = round(float(rating_raw), 1) if rating_raw is not None else None
    except (ValueError, TypeError):
        rating = None

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


def _parse_lead_price(item: dict, nights: int = 1) -> tuple[float | None, float | None, str]:
    """Pull the lead-in price off a /v1/booking/search result item.

    Confirmed against a real StayAPI response: "price" — either a bare
    amount or {"amount": 873.69, "currency": "USD", ...} — is the TOTAL
    for the whole stay being searched (checkin -> checkout), not a nightly
    rate. Returns (price_per_night, total_price, currency) so callers store
    both correctly instead of treating an already-total figure as nightly
    and then re-multiplying it by `nights` on top (the old bug: a $291/night,
    3-night stay totalling $874 was displayed as "$874/night" and its
    "total" computed as $874 x 3 = $2,621).
    """
    price_raw = item.get("price")
    currency = "USD"
    if isinstance(price_raw, dict):
        amount = price_raw.get("amount") or price_raw.get("value")
        currency = price_raw.get("currency") or currency
    else:
        amount = price_raw
    try:
        total = float(amount) if amount is not None else None
    except (ValueError, TypeError):
        total = None

    if total is None:
        return None, None, currency

    nightly = total / nights if nights else total
    return nightly, total, currency


def _parse_coords(item: dict) -> tuple[float | None, float | None]:
    """Pull latitude/longitude off a hotel dict. Real StayAPI /v1/booking/search
    results put these as top-level fields, not nested — but a few other
    hotel-shaped sources (mock data, other endpoints) nest them under
    coordinates/location/geo, so both are checked."""
    if item.get("latitude") is not None or item.get("longitude") is not None:
        lat, lon = item.get("latitude"), item.get("longitude")
    else:
        nested = item.get("coordinates") or item.get("location") or item.get("geo") or {}
        lat = nested.get("latitude") or nested.get("lat")
        lon = nested.get("longitude") or nested.get("lon") or nested.get("lng")
    try:
        lat = float(lat) if lat is not None else None
    except (ValueError, TypeError):
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except (ValueError, TypeError):
        lon = None
    return lat, lon


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
            lat, lon = _parse_coords(h)
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

        lat, lon = _parse_coords(item)

        sr = item.get("star_rating") or item.get("stars") or item.get("class")
        try:
            star_rating = int(sr) if sr is not None else None
        except (ValueError, TypeError):
            star_rating = None

        rating, review_count, image_url = _parse_review_and_image(item)
        lead_nightly, lead_total, lead_currency = _parse_lead_price(item, total_nights)
        # Real search results carry a top-level room_name (e.g. "Superior
        # Double Room") even before per-room backfill runs — keep it instead
        # of "" so this card isn't excluded by getSelectableRooms on the
        # frontend (which filters out rooms with no room_name).
        lead_room_name = item.get("room_name") or ""

        hotels.append(Hotel(
            hotel_id=hotel_id,
            name=str(item.get("hotel_name") or item.get("name") or ""),
            address=item.get("address") or item.get("hotel_address"),
            city=item.get("city") or item.get("city_name"),
            latitude=lat,
            longitude=lon,
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
            # Uses the search endpoint's own lead-in price if StayAPI gave one;
            # otherwise stays a 0.0 placeholder until backfill_hotel_rooms runs.
            selected_room=RoomOption(
                room_name=lead_room_name,
                max_occupancy=0,
                price_per_night=round(lead_nightly, 2) if lead_nightly is not None else 0.0,
                total_price=round(lead_total, 2) if lead_total is not None else 0.0,
                currency=lead_currency,
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

        updated.append(Hotel(
            hotel_id=hotel.hotel_id,
            name=hotel.name,
            address=hotel.address,
            city=hotel.city,
            latitude=hotel.latitude,
            longitude=hotel.longitude,
            star_rating=hotel.star_rating,
            stay_schedule=hotel.stay_schedule,
            selected_room=selected,
            available_rooms=available,
        ))

    return updated

def _first_float(*values) -> float | None:
    """Return the first value that parses as a float, trying each in order.
    Handles StayAPI's currency-formatted strings ("$311", "1,234.50") as well
    as plain numbers. Returns None if nothing parses (caller decides the
    fallback - unlike a bare `or` chain, this doesn't treat 0.0 as missing)."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace(",", "").replace("$", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                continue
    return None


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
        # Only explicitly-named "per night" fields count as nightly. Bare
        # "price"/"price_value"/"amount" fields are conventionally the STAY
        # TOTAL on these block shapes (confirmed on the /v1/booking/search
        # item-level "price" field against a real response — see
        # _parse_lead_price), not a nightly rate, so they belong on the
        # total side. Treating them as nightly and then re-multiplying by
        # `nights` for total double-counts the stay length.
        nightly = _first_float(
            block.get("price_per_night_value"),
            block.get("price_per_night"),
            block.get("nightly_price"),
        )
        total = _first_float(
            block.get("total_price_value"),
            block.get("total_value"),
            block.get("total_price"),
            block.get("total"),
            block.get("price_value"),
            block.get("price"),
            block.get("amount"),
        )

        if nightly is None and total is not None:
            nightly = total / nights
        if nightly is None:
            nightly = 0.0
        if total is None:
            total = nightly * nights

        cur = block.get("currency") or "USD"

        occupancy = (block.get("max_occupancy") or block.get("max_persons")
                     or block.get("max_guests") or block.get("occupancy")
                     or block.get("max_occupancy_persons") or 2)

        # --- breakfast ---
        # breakfast_included/is_refundable are real booleans in StayAPI's
        # response, and False is a meaningful value here (not "missing") -
        # `or` chaining treats False as falsy and would incorrectly fall
        # through to the next field (e.g. a paid-breakfast meal_plan string
        # that happens to contain the word "breakfast"), so these must be
        # checked with "is not None" rather than truthiness.
        if block.get("breakfast_included") is not None:
            breakfast = bool(block.get("breakfast_included"))
        else:
            breakfast_raw = block.get("has_breakfast")
            if breakfast_raw is None:
                breakfast_raw = block.get("breakfast")
            if breakfast_raw is None:
                breakfast_raw = block.get("meal_plan") or ""
            if isinstance(breakfast_raw, bool):
                breakfast = breakfast_raw
            elif isinstance(breakfast_raw, str):
                breakfast = "breakfast" in breakfast_raw.lower() and breakfast_raw.lower() not in ("no", "false", "0", "")
            else:
                breakfast = bool(breakfast_raw)

        # --- refundable ---
        if block.get("is_refundable") is not None:
            refund = bool(block.get("is_refundable"))
        else:
            refund_raw = block.get("refundable")
            if refund_raw is None:
                refund_raw = block.get("free_cancellation")
            if refund_raw is None:
                refund_raw = False
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

    # Real StayAPI /v1/booking/search shape is {"data": {"hotels": [...]}} -
    # "data" itself is a dict, not the list, so it must be drilled into before
    # falling back to other shapes.
    data_field = res_data.get("data")
    if isinstance(data_field, dict):
        results_list = data_field.get("hotels") or []
    elif isinstance(data_field, list):
        results_list = data_field
    else:
        results_list = res_data.get("results") or res_data.get("hotels") or []
    if not isinstance(results_list, list):
        return []

    hotels: list[Hotel] = []
    for item in results_list:
        if not isinstance(item, dict):
            continue

        hotel_id = str(item.get("hotel_id") or item.get("id") or "")
        if not hotel_id:
            continue

        lat, lon = _parse_coords(item)

        # Try to parse star_rating as int
        sr = item.get("star_rating") or item.get("stars") or item.get("class")
        try:
            star_rating = int(sr) if sr is not None else None
        except (ValueError, TypeError):
            star_rating = None

        rating, review_count, image_url = _parse_review_and_image(item)
        lead_nightly, lead_total, lead_currency = _parse_lead_price(item, total_nights)
        lead_room_name = item.get("room_name") or ""

        hotels.append(Hotel(
            hotel_id=hotel_id,
            name=str(item.get("hotel_name") or item.get("name") or ""),
            address=item.get("address") or item.get("hotel_address"),
            city=item.get("city") or item.get("city_name"),
            latitude=lat,
            longitude=lon,
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
            # Uses the search endpoint's own lead-in price if StayAPI gave one
            # (often null there); get_hotel_ui_cards's caller (api.py) backfills
            # this with real per-room data from get_hotel_prices afterwards.
            selected_room=RoomOption(
                room_name=lead_room_name,
                max_occupancy=0,
                price_per_night=round(lead_nightly, 2) if lead_nightly is not None else 0.0,
                total_price=round(lead_total, 2) if lead_total is not None else 0.0,
                currency=lead_currency,
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