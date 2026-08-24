"""
only is hotel data API, not a transactional booking API
Auth is a single header, no OAuth: `x-api-key: YOUR_KEY`.
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv  


load_dotenv()
STAYAPI_BASE_URL = "https://api.stayapi.com"


class StayAPIError(RuntimeError):
    """Raised on any non-2xx response from StayAPI."""


def _headers() -> dict:
    api_key = os.getenv("STAYAPI_KEY")
    if not api_key:
        raise StayAPIError("STAYAPI_KEY environment variable is not set.")
    return {
        "x-api-key": api_key,
        "Accept": "application/json",
    }


def _request(path: str, params: Optional[dict] = None) -> dict:
    url = f"{STAYAPI_BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if not resp.ok:
        raise StayAPIError(f"StayAPI error {resp.status_code} on GET {path}: {resp.text}")
    body = resp.json()
    if body.get("success") is False:
        raise StayAPIError(f"StayAPI returned success=false on GET {path}: {body}")
    return body


# --------------------------------------------------------------------------- #
# Output parsers — StayAPI's raw JSON is already fairly flat, but we still
# normalize field names/shape here so callers don't touch raw payloads and
# so a future schema tweak on StayAPI's side only needs a change in one place.
# --------------------------------------------------------------------------- #

def _parse_destination_suggestion(dest: dict) -> dict:
    return {
        "dest_id": dest.get("dest_id"),
        "dest_type": dest.get("dest_type"),
        "label": dest.get("label"),
    }


def _parse_search_hit(hit: dict) -> dict:
    return {
        "hotel_id": hit.get("hotel_id"),
        "hotel_name": hit.get("hotel_name"),
        "url": hit.get("url"),
        "image_url": hit.get("image_url"),
        "star_rating": hit.get("star_rating"),
        "review_score": hit.get("review_score"),
        "review_score_word": hit.get("review_score_word"),
        "review_count": hit.get("review_count"),
        "address": hit.get("address"),
        "distance_from_center_km": hit.get("distance_from_center"),
        "unit_configuration_label": hit.get("unit_configuration_label"),
        "min_total_price": hit.get("min_total_price"),
        "currency_code": hit.get("currency_code"),
        "free_cancellation": bool(hit.get("is_free_cancellable")),
        "no_prepayment": bool(hit.get("is_no_prepayment_block")),
        "checkin": hit.get("checkin"),
        "checkout": hit.get("checkout"),
    }


def _parse_meta_result(meta: dict) -> dict:
    return {
        "hotel_name": meta.get("hotel_name"),
        "location": meta.get("location"),
        "links": meta.get("links", {}),
        "platform_count": meta.get("platform_count"),
    }


def _parse_hotel_details(details: dict) -> dict:
    coords = details.get("coordinates") or {}
    return {
        "hotel_id": details.get("hotel_id"),
        "name": details.get("name"),
        "address": details.get("address"),
        "latitude": coords.get("lat"),
        "longitude": coords.get("lng"),
        "star_rating": details.get("star_rating"),
        "review_score": details.get("review_score"),
        "amenities": details.get("amenities", []),
    }


def _parse_review(review: dict) -> dict:
    return {
        "author": review.get("author"),
        "country": review.get("country"),
        "rating": review.get("rating"),
        "title": review.get("title"),
        "date": review.get("date"),
        "trip_type": review.get("trip_type"),  # present on TripAdvisor reviews
    }


def _parse_reviews_response(raw: dict) -> dict:
    data = raw.get("data", {})
    return {
        "hotel_id": data.get("hotel_id") or data.get("location_id"),
        "total_reviews": data.get("total_reviews"),
        "average_rating": data.get("average_rating"),
        "reviews": [_parse_review(r) for r in data.get("reviews", [])],
    }


def _parse_calendar_day(day: dict) -> dict:
    return {
        "date": day.get("date"),
        "available": day.get("available"),
        "min_nights": day.get("min_nights"),
        "price": day.get("price"),
    }


# --------------------------------------------------------------------------- #
# 1. Resolve a place name to a Booking.com dest_id
#
# search_hotels needs a dest_id, not a free-text place name, so this is
# normally the first call in the flow.
# --------------------------------------------------------------------------- #

def lookup_destination(query: str, language: str = "en-us") -> dict:
    """
    Resolve a free-text city/region name (e.g. "Paris", "Bali") to the
    dest_id / dest_type search_hotels expects.

    Unlike most StayAPI endpoints, this one is NOT wrapped in a `data` key —
    dest_id/dest_type/suggestions sit at the top level of the response.
    `dest_id`/`dest_type` here are already the top (default) match; if the
    query was ambiguous (e.g. "Paris" -> France vs Texas), `suggestions`
    holds the alternatives so you can ask the user to disambiguate instead
    of silently booking the wrong city.
    """
    raw = _request("/v1/booking/destinations/lookup", params={"query": query, "language": language})
    return {
        "query": raw.get("query"),
        "normalized_query": raw.get("normalized_query"),
        "dest_id": raw.get("dest_id"),
        "dest_type": raw.get("dest_type"),
        "suggestions": [_parse_destination_suggestion(s) for s in raw.get("suggestions", [])],
        "message": raw.get("message"),
    }


# --------------------------------------------------------------------------- #
# 2. Search hotels in a destination across given dates
# --------------------------------------------------------------------------- #

class HotelSearchInput(BaseModel):
    dest_id: str = Field(..., description="From lookup_destination")
    dest_type: str = Field("CITY", description="From lookup_destination, e.g. CITY/DISTRICT/AIRPORT/LANDMARK")
    checkin: str = Field(..., description="YYYY-MM-DD")
    checkout: str = Field(..., description="YYYY-MM-DD")
    adults: int = Field(2, ge=1)
    rooms: int = Field(1, ge=1, le=10)
    children: int = Field(0, ge=0)
    children_ages: Optional[list[int]] = Field(None, description="One age per child, only sent if children > 0")
    rows_per_page: int = Field(25, ge=1, le=100)
    offset: int = Field(0, ge=0)
    currency: str = Field("USD")


def search_hotels(params: HotelSearchInput) -> dict:
    """
    Search Booking.com hotels in a destination across the given dates.
    Returns a parsed list of hits (hotel_id, name, rating, price, address,
    cancellation flags) plus pagination info — use hotel_id on
    get_hotel_details / get_hotel_reviews for more detail on any specific
    result, and offset/rows_per_page to page through the rest.
    """
    query_params: dict = {
        "dest_id": params.dest_id,
        "dest_type": params.dest_type,
        "checkin": params.checkin,
        "checkout": params.checkout,
        "adults": params.adults,
        "rooms": params.rooms,
        "children": params.children,
        "rows_per_page": params.rows_per_page,
        "offset": params.offset,
        "currency": params.currency,
    }
    if params.children and params.children_ages:
        query_params["children_ages"] = ",".join(str(a) for a in params.children_ages)

    raw = _request("/v1/booking/search", params=query_params)
    hits = raw.get("data", [])
    pagination = raw.get("pagination", {})
    return {
        "results": [_parse_search_hit(h) for h in hits],
        "total_count": pagination.get("total_count_with_filters"),
        "rows_per_page": pagination.get("rows_per_page"),
        "current_offset": pagination.get("current_offset"),
    }


# --------------------------------------------------------------------------- #
# 3. Find a hotel's IDs/links across every OTA StayAPI covers
# --------------------------------------------------------------------------- #

def meta_search(hotel_name: str, location: str) -> dict:
    """
    Find booking links and IDs for a hotel across every OTA StayAPI covers
    (Booking.com, Expedia, TripAdvisor, Agoda, ...), matched by name +
    location. Useful for cross-referencing a hotel_id you got from one
    source against another source's numbering.
    """
    raw = _request("/v1/meta/search", params={"hotel_name": hotel_name, "location": location})
    return _parse_meta_result(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 4. Rich hotel details (location, amenities, scores)
# --------------------------------------------------------------------------- #

def get_hotel_details(hotel_id: str) -> dict:
    """
    Fetch rich Booking.com hotel details: address, coordinates, amenities,
    star rating, review score. Use this to justify a recommendation beyond
    just price. Returns a parsed dict.
    """
    raw = _request("/v2/booking/hotel/details", params={"hotel_id": hotel_id})
    return _parse_hotel_details(raw.get("data", {}))


# --------------------------------------------------------------------------- #
# 5. Paginated guest reviews for a hotel
# --------------------------------------------------------------------------- #

def get_hotel_reviews(hotel_id: str, per_page: int = 10, language: str = "en") -> dict:
    """
    Paginated Booking.com guest reviews for a hotel_id. Returns parsed
    reviews (author, country, rating, title, date), not the raw payload.
    """
    raw = _request(
        "/v1/booking/hotel/reviews",
        params={"hotel_id": hotel_id, "per_page": per_page, "language": language},
    )
    return _parse_reviews_response(raw)


# --------------------------------------------------------------------------- #
# 6. TripAdvisor reviews by location_id
# --------------------------------------------------------------------------- #

def get_tripadvisor_reviews(location_id: str, page: int = 1) -> dict:
    """
    Paginated TripAdvisor-native reviews for a property's location_id.
    Returns parsed reviews, same shape as get_hotel_reviews.
    """
    raw = _request("/v1/tripadvisor/hotel/reviews", params={"location_id": location_id, "page": page})
    return _parse_reviews_response(raw)


# --------------------------------------------------------------------------- #
# 7. Airbnb listing availability calendar
# --------------------------------------------------------------------------- #

def get_airbnb_calendar(listing_id: str, months: int = 3) -> dict:
    """
    Multi-month availability + nightly price calendar for an Airbnb
    listing_id. Returns a parsed list of {date, available, min_nights,
    price} entries — useful for showing an availability grid rather than a
    single search result.
    """
    raw = _request("/v1/airbnb/calendar", params={"id": listing_id, "months": months})
    data = raw.get("data", {})
    return {
        "listing_id": data.get("id"),
        "calendar": [_parse_calendar_day(d) for d in data.get("calendar", [])],
    }


# --------------------------------------------------------------------------- #
# Demo — runs the real API against your free-tier key and prints each
# step's output.
#
# Requires STAYAPI_KEY to be set (50 free requests on signup, no card).
#
#     python hotel_module.py
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    def _print_section(title: str) -> None:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    if not os.getenv("STAYAPI_KEY"):
        print("Set STAYAPI_KEY before running this. Sign up at https://stayapi.com/users/sign_up")
        sys.exit(1)

    _print_section("1. lookup_destination — 'Paris'")
    dest_result = lookup_destination(query="Paris")
    print(dest_result)

    dest_id = dest_result["dest_id"]
    dest_type = dest_result["dest_type"] or "CITY"
    if not dest_id:
        print(f"No destination resolved: {dest_result['message']}")
        sys.exit(1)
    if dest_result["suggestions"]:
        print(f"(Ambiguous query — {len(dest_result['suggestions'])} suggestions available, using the first.)")

    _print_section(f"2. search_hotels — dest_id={dest_id}, dest_type={dest_type}")
    search_result = search_hotels(
        HotelSearchInput(
            dest_id=dest_id, dest_type=dest_type, checkin="2026-12-10", checkout="2026-12-13", adults=2, rooms=1
        )
    )
    print(search_result)

    hits = search_result["results"]
    if not hits:
        print("No hotels returned for that destination/date range.")
        sys.exit(0)
    first_hotel_id = hits[0]["hotel_id"]

    _print_section(f"3. get_hotel_details — hotel_id={first_hotel_id}")
    print(get_hotel_details(hotel_id=first_hotel_id))

    _print_section(f"4. get_hotel_reviews — hotel_id={first_hotel_id}")
    print(get_hotel_reviews(hotel_id=first_hotel_id, per_page=5))

    _print_section("5. meta_search — cross-OTA links for the same hotel")
    print(meta_search(hotel_name=hits[0]["hotel_name"], location="Paris"))