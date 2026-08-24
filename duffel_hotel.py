"""
Duffel has two ways to search:
  1. Location-based: lat/lng + radius_km. Returns all matching accommodation.
  2. Accommodation-ID based: a curated list of known accommodation_ids
     This mode also supports `fetch_rates=True` to get bookable rates back inline,
     skipping the separate `get_hotel_rates` call.

added with output parsing also
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from pydantic import BaseModel, Field, model_validator


def _load_dotenv_if_present() -> None:
    """
    Minimal .env loader — no python-dotenv dependency required. Reads a
    .env file next to this script (KEY=VALUE per line, # comments allowed)
    and sets any keys not already present in the real environment.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv_if_present()

DUFFEL_BASE_URL = "https://api.duffel.com"
DUFFEL_VERSION = "v2"

# Duffel Stays test hotels only appear in test mode when searching this
# exact coordinate.
TEST_HOTEL_LATITUDE = -24.38
TEST_HOTEL_LONGITUDE = -128.32


class DuffelAPIError(RuntimeError):
    """Raised on any non-2xx response from Duffel."""


def _headers() -> dict:
    api_key = os.getenv("DUFFEL_API_KEY")
    if not api_key:
        raise DuffelAPIError("DUFFEL_API_KEY environment variable is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Duffel-Version": DUFFEL_VERSION,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, json_body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    url = f"{DUFFEL_BASE_URL}{path}"
    resp = requests.request(method, url, headers=_headers(), json=json_body, params=params, timeout=30)
    if not resp.ok:
        raise DuffelAPIError(f"Duffel API error {resp.status_code} on {method} {path}: {resp.text}")
    return resp.json()


# --------------------------------------------------------------------------- #
# Output parsers — shrink Duffel's raw schema down to frontend-friendly dicts
# --------------------------------------------------------------------------- #

def _parse_address(location: Optional[dict]) -> Optional[dict]:
    if not location:
        return None
    addr = location.get("address") or {}
    coords = location.get("geographic_coordinates") or {}
    return {
        "line_one": addr.get("line_one"),
        "city": addr.get("city_name"),
        "region": addr.get("region"),
        "postal_code": addr.get("postal_code"),
        "country_code": addr.get("country_code"),
        "latitude": coords.get("latitude"),
        "longitude": coords.get("longitude"),
    }


def _parse_accommodation_summary(acc: dict) -> dict:
    """Light-weight accommodation info suitable for a search-results list."""
    return {
        "accommodation_id": acc.get("id"),
        "name": acc.get("name"),
        "rating": acc.get("rating"),
        "review_score": acc.get("review_score"),
        "review_count": acc.get("review_count"),
        "address": _parse_address(acc.get("location")),
        "photo_url": (acc.get("photos") or [{}])[0].get("url"),
        "chain": (acc.get("chain") or {}).get("name"),
    }


def _parse_accommodation_details(acc: dict) -> dict:
    """Full accommodation info for `get_hotel_details`."""
    summary = _parse_accommodation_summary(acc)
    summary.update({
        "description": acc.get("description"),
        "phone_number": acc.get("phone_number"),
        "email": acc.get("email"),
        "photos": [p.get("url") for p in acc.get("photos", [])],
        "amenities": [a.get("description") or a.get("type") for a in acc.get("amenities", [])],
        "check_in_information": acc.get("check_in_information"),
        "key_collection": (acc.get("key_collection") or {}).get("instructions"),
        "supported_loyalty_programme": acc.get("supported_loyalty_programme"),
    })
    return summary


def _parse_rate(rate: dict) -> dict:
    return {
        "rate_id": rate.get("id"),
        "name": rate.get("name"),
        "total_amount": rate.get("total_amount"),
        "total_currency": rate.get("total_currency"),
        "board_type": rate.get("board_type"),
        "payment_type": rate.get("payment_type"),
        "refundable": bool(rate.get("cancellation_timeline")),
        "cancellation_timeline": rate.get("cancellation_timeline"),
        "benefits": [b.get("title") for b in rate.get("benefits", [])],
        "conditions": rate.get("conditions"),
        "loyalty_programme_required": rate.get("loyalty_programme_required"),
        "supported_loyalty_programme": rate.get("supported_loyalty_programme"),
        "expires_at": rate.get("expires_at"),
    }


def _parse_room(room: dict) -> dict:
    return {
        "room_name": room.get("name"),
        "beds": room.get("beds"),
        "photos": [p.get("url") for p in room.get("photos", [])],
        "rates": [_parse_rate(r) for r in room.get("rates", [])],
    }


def _parse_search_result(result: dict) -> dict:
    acc = result.get("accommodation") or {}
    parsed = {
        "search_result_id": result.get("id"),
        "check_in_date": result.get("check_in_date"),
        "check_out_date": result.get("check_out_date"),
        "cheapest_total_amount": result.get("cheapest_rate_total_amount"),
        "cheapest_total_currency": result.get("cheapest_rate_currency"),
        "accommodation": _parse_accommodation_summary(acc),
    }
    # Present when fetch_rates=True (accommodation-ID search) so callers can
    # skip straight to create_hotel_quote without a separate rates call.
    if acc.get("rooms"):
        parsed["rooms"] = [_parse_room(r) for r in acc["rooms"]]
    return parsed


def _parse_quote(quote: dict) -> dict:
    return {
        "quote_id": quote.get("id"),
        "total_amount": quote.get("total_amount"),
        "total_currency": quote.get("total_currency"),
        "expires_at": quote.get("expires_at"),
        "supported_loyalty_programme": quote.get("supported_loyalty_programme"),
    }


def _parse_booking(booking: dict) -> dict:
    acc = booking.get("accommodation") or {}
    return {
        "booking_id": booking.get("id"),
        "status": booking.get("status"),
        "reference": booking.get("reference"),
        "check_in_date": booking.get("check_in_date"),
        "check_out_date": booking.get("check_out_date"),
        "email": booking.get("email"),
        "phone_number": booking.get("phone_number"),
        "guests": booking.get("guests"),
        "accommodation": _parse_accommodation_summary(acc) if acc else None,
        "key_collection": (acc.get("key_collection") or {}).get("instructions") if acc else None,
        "cancelled_at": booking.get("cancelled_at"),
        "confirmed_at": booking.get("confirmed_at"),
    }


# structs---------------------------------------------------------------------- #

class GeoLocation(BaseModel):
    latitude: float
    longitude: float
    radius_km: int = Field(10, ge=1, le=100, description="Search radius in km")


class AccommodationIdSearch(BaseModel):
    ids: list[str] = Field(..., min_length=1, description="Known accommodation_ids to search, from a prior location search")
    fetch_rates: bool = Field(False, description="If true, return bookable rates inline (skips get_hotel_rates)")


class GuestName(BaseModel):
    given_name: str
    family_name: str
    born_on: Optional[str] = Field(None, description="YYYY-MM-DD, optional")


def _build_guests(adult_count: int, child_count: int, child_ages: Optional[list[int]]) -> list[dict]:
    guests = [{"type": "adult"} for _ in range(adult_count)]
    for i in range(child_count):
        guest = {"type": "child"}
        if child_ages and i < len(child_ages):
            guest["age"] = child_ages[i]
        guests.append(guest)
    return guests


# 1. Search accommodation (location OR accommodation IDs)
class HotelSearchInput(BaseModel):
    check_in_date: str = Field(..., description="YYYY-MM-DD")
    check_out_date: str = Field(..., description="YYYY-MM-DD")
    adult_count: int = Field(1, ge=1)
    child_count: int = Field(0, ge=0)
    child_ages: Optional[list[int]] = Field(
        None, description="Age per child, same order/length as child_count"
    )
    rooms: int = Field(1, ge=1, le=10)
    location: Optional[GeoLocation] = Field(
        None, description="Search by coordinates + radius. Mutually exclusive with `accommodation`."
    )
    accommodation: Optional[AccommodationIdSearch] = Field(
        None, description="Search a curated list of accommodation_ids. Mutually exclusive with `location`."
    )

    @model_validator(mode="after")
    def _validate_search_mode(self) -> "HotelSearchInput":
        if bool(self.location) == bool(self.accommodation):
            raise ValueError(
                "Provide exactly one of `location` or `accommodation` — Duffel does not accept both."
            )
        if self.child_count and self.child_ages and len(self.child_ages) != self.child_count:
            raise ValueError("child_ages length must match child_count when provided.")
        return self


def search_hotels(params: HotelSearchInput) -> dict:
    """
    Each result has a `search_result_id`. If searched by location, or by
    accommodation ID without fetch_rates, call `get_hotel_rates` on it next.
    If searched by accommodation ID with fetch_rates=True, the result already
    includes bookable `rooms`/`rates` — skip straight to `create_hotel_quote`.

    Returns a parsed list of search results, after parsing raw Duffel payload.
    """
    body: dict = {
        "check_in_date": params.check_in_date,
        "check_out_date": params.check_out_date,
        "rooms": params.rooms,
        "guests": _build_guests(params.adult_count, params.child_count, params.child_ages),
    }
    if params.location:
        body["location"] = {
            "geographic_coordinates": {"latitude": params.location.latitude, "longitude": params.location.longitude},
            "radius": params.location.radius_km,
        }
    else:
        body["accommodation"] = {"ids": params.accommodation.ids, "fetch_rates": params.accommodation.fetch_rates}

    raw = _request("POST", "/stays/search", json_body={"data": body})
    results = raw.get("data", {}).get("results", [])
    return {"results": [_parse_search_result(r) for r in results]}


# --------------------------------------------------------------------------- #
# 2. Get property details
def get_hotel_details(accommodation_id: str) -> dict:
    raw = _request("GET", f"/stays/accommodation/{accommodation_id}")
    return _parse_accommodation_details(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 3. Get bookable rates for a search result
def get_hotel_rates(search_result_id: str) -> dict:
    """
    Get the current bookable room rates for a specific search result
    Not needed if the original search used accommodation IDs with
    fetch_rates=True

    Returns parsed rooms/rates (rate_id, room name, price, cancellation
    policy, board type), not the raw Duffel payload. Rates are time-limited
    """
    raw = _request("POST", f"/stays/search_results/{search_result_id}/actions/fetch_all_rates")
    rooms = raw.get("data", {}).get("accommodation", {}).get("rooms", [])
    return {"rooms": [_parse_room(r) for r in rooms]}


# --------------------------------------------------------------------------- #
# 4. Lock in a price via quote
def create_hotel_quote(rate_id: str) -> dict:
    """
    Lock in a rate's price for a short window before booking.
    """
    body = {"data": {"rate_id": rate_id}}
    raw = _request("POST", "/stays/quotes", json_body=body)
    return _parse_quote(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 5. Book
def book_hotel(
    quote_id: str,
    guests: list[GuestName],
    email: str,
    phone_number: str,
    confirmed_price: float,
    confirmed_currency: str,
    accommodation_special_requests: Optional[str] = None,
) -> dict:
    """
    Create a confirmed hotel booking from a locked-in quote.
    """
    quote = _request("GET", f"/stays/quotes/{quote_id}")
    quote_data = quote.get("data", {})
    live_price = float(quote_data.get("total_amount", 0))
    live_currency = quote_data.get("total_currency", "")

    if abs(live_price - confirmed_price) > 0.01 or live_currency != confirmed_currency:
        return {
            "code": "price_mismatch",
            "message": "Live quote differs from the price the user confirmed. Re-confirm before booking.",
            "live_price": live_price,
            "live_currency": live_currency,
            "confirmed_price": confirmed_price,
            "confirmed_currency": confirmed_currency,
        }

    body = {
        "data": {
            "quote_id": quote_id,
            "guests": [g.model_dump(exclude_none=True) for g in guests],
            "email": email,
            "phone_number": phone_number,
            **({"accommodation_special_requests": accommodation_special_requests} if accommodation_special_requests else {}),
        }
    }
    raw = _request("POST", "/stays/bookings", json_body=body)
    return _parse_booking(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 6. Retrieve booking status
def get_hotel_booking(booking_id: str) -> dict:
    raw = _request("GET", f"/stays/bookings/{booking_id}")
    return _parse_booking(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 7. Cancel a booking  (was missing — no way to undo a booking before)
def cancel_hotel_booking(booking_id: str) -> dict:
    raw = _request("POST", f"/stays/bookings/{booking_id}/actions/cancel")
    return _parse_booking(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 8. List bookings  (was missing — needed for "show my trips" type asks)
def list_hotel_bookings(limit: int = 50, after: Optional[str] = None) -> dict:
    params = {"limit": limit}
    if after:
        params["after"] = after
    raw = _request("GET", "/stays/bookings", params=params)
    bookings = raw.get("data", [])
    return {
        "bookings": [_parse_booking(b) for b in bookings],
        "next_cursor": raw.get("meta", {}).get("after"),
    }


# --------------------------------------------------------------------------- #
# Demo — runs the real sandbox flow and prints each step's output.
# Requires DUFFEL_API_KEY to be set to a duffel_test_... key.
#
#     python hotel_module.py            # search -> rates -> quote, no booking
#     python hotel_module.py --book     # also books, fetches, and cancels
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    from datetime import date, timedelta

    def _print_section(title: str) -> None:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    if not os.getenv("DUFFEL_API_KEY"):
        print("Set DUFFEL_API_KEY to a duffel_test_... sandbox key before running this.")
        sys.exit(1)

    do_book = "--book" in sys.argv
    check_in = (date.today() + timedelta(days=30)).isoformat()
    check_out = (date.today() + timedelta(days=32)).isoformat()

    _print_section(f"1. search_hotels — test hotels at ({TEST_HOTEL_LATITUDE}, {TEST_HOTEL_LONGITUDE})")
    search_result = search_hotels(
        HotelSearchInput(
            check_in_date=check_in,
            check_out_date=check_out,
            adult_count=1,
            location=GeoLocation(latitude=TEST_HOTEL_LATITUDE, longitude=TEST_HOTEL_LONGITUDE, radius_km=5),
        )
    )
    print(search_result)

    results = search_result["results"]
    if not results:
        print("No test hotels returned — double-check the key is a duffel_test_... sandbox key.")
        sys.exit(1)
    first = results[0]

    _print_section(f"2. get_hotel_rates — search_result_id={first['search_result_id']}")
    rates = get_hotel_rates(search_result_id=first["search_result_id"])
    print(rates)
    first_rate = rates["rooms"][0]["rates"][0]

    _print_section(f"3. create_hotel_quote — rate_id={first_rate['rate_id']}")
    quote = create_hotel_quote(rate_id=first_rate["rate_id"])
    print(quote)

    if not do_book:
        print("\nStopping before booking (pass --book to also test book_hotel + cancel_hotel_booking).")
        sys.exit(0)

    _print_section("4. book_hotel")
    booking = book_hotel(
        quote_id=quote["quote_id"],
        guests=[GuestName(given_name="Ada", family_name="Lovelace")],
        email="[email protected]",
        phone_number="+442080160509",
        confirmed_price=float(quote["total_amount"]),
        confirmed_currency=quote["total_currency"],
    )
    print(booking)
    if booking.get("code") == "price_mismatch":
        print("Quote expired between steps — re-run.")
        sys.exit(1)

    _print_section(f"5. get_hotel_booking — booking_id={booking['booking_id']}")
    print(get_hotel_booking(booking_id=booking["booking_id"]))

    _print_section("6. list_hotel_bookings")
    print(list_hotel_bookings(limit=10))

    _print_section(f"7. cancel_hotel_booking — booking_id={booking['booking_id']}")
    print(cancel_hotel_booking(booking_id=booking["booking_id"]))