"""Parsing helpers that turn Atlas Flight API responses into frontend-ready
FlightOption objects. Mirrors hotels.py's role for StayAPI: tools.py owns
the raw HTTP call plus the agent-facing (@tool, pre-formatted-string) card
builder; this module owns the typed/numeric shape used by /flight/change,
so the frontend formats prices/dates itself instead of parsing display
strings back out.
"""
from datetime import datetime
from typing import Optional

from schemas import FlightOption, FlightSegmentInfo
from tools import search_flights_raw


def _parse_atlas_datetime(dt_str: str) -> tuple[str, str, str]:
    """Returns (HH:MM, YYYY-MM-DD, pretty_date_label) from Atlas's raw
    YYYYMMDDHHMM-prefixed datetime string."""
    if not dt_str or len(dt_str) < 12:
        return "--:--", "", "Unknown Date"
    dt = datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
    return dt.strftime("%H:%M"), dt.strftime("%Y-%m-%d"), f"{dt.day} {dt.strftime('%b, %A')}"


def _first_float(*values) -> Optional[float]:
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


def extract_flight_options(res_data: dict) -> list[FlightOption]:
    """Flatten Atlas routings into one FlightOption per LEG.

    A round-trip routing carries both directions (fromSegments + retSegments)
    and a single combined fare. This used to read only `fromSegments or
    segments`, so every return leg was silently dropped and the frontend's
    flight picker could never offer one. Both legs of a round-trip routing now
    come back, share the same `id` prefix, and carry price_is_round_trip=True
    with the same `price` — do not sum them.
    """
    routings = res_data.get("routings", [])
    options: list[FlightOption] = []

    for item in routings:
        outbound = item.get("fromSegments") or item.get("segments") or []
        inbound = item.get("retSegments") or item.get("returnSegments") or []
        is_round_trip = bool(outbound and inbound)

        for direction, segments in (("outbound", outbound), ("return", inbound)):
            if not segments:
                continue
            option = _option_from_segments(item, segments, direction, is_round_trip)
            if option is not None:
                options.append(option)

    return options


def _option_from_segments(
    item: dict, segments: list[dict], direction: str, is_round_trip: bool,
) -> Optional[FlightOption]:
    first_seg, last_seg = segments[0], segments[-1]

    dep_time, dep_iso, dep_label = _parse_atlas_datetime(
        first_seg.get("depTime") or first_seg.get("departureTime", ""))
    arr_time, arr_iso, arr_label = _parse_atlas_datetime(
        last_seg.get("arrTime") or last_seg.get("arrivalTime", ""))

    dep_airport = first_seg.get("depAirport") or first_seg.get("departureAirport", "")
    arr_airport = last_seg.get("arrAirport") or last_seg.get("arrivalAirport", "")

    carrier_val = first_seg.get("carrier")
    carrier_code = None
    if isinstance(carrier_val, dict):
        carrier_name = carrier_val.get("name") or carrier_val.get("code") or "Airline"
        carrier_code = carrier_val.get("code")
    elif isinstance(carrier_val, str) and carrier_val.strip():
        carrier_name = carrier_val
    else:
        carrier_name = (
            first_seg.get("carrierName")
            or first_seg.get("marketingAirline")
            or first_seg.get("operatingAirline")
            or "Airline"
        )

    flight_number = str(first_seg.get("flightNumber") or "")
    price = _first_float(item.get("adultPrice"), item.get("price"), item.get("totalPrice")) or 0.0
    num_stops = max(len(segments) - 1, 0)
    layover_text = "Direct"
    if num_stops > 0:
        transfer_airport = first_seg.get("arrAirport") or first_seg.get("arrivalAirport", "")
        layover_text = f"{num_stops} stop in {transfer_airport}"

    duration_minutes = None
    if dep_iso and arr_iso:
        try:
            dep_dt = datetime.strptime(f"{dep_iso} {dep_time}", "%Y-%m-%d %H:%M")
            arr_dt = datetime.strptime(f"{arr_iso} {arr_time}", "%Y-%m-%d %H:%M")
            duration_minutes = max(int((arr_dt - dep_dt).total_seconds() // 60), 0)
        except ValueError:
            pass

    seats_raw = item.get("seats") or first_seg.get("seatCount")
    try:
        seats_left = int(seats_raw) if seats_raw is not None else None
    except (ValueError, TypeError):
        seats_left = None

    routing_id = str(item.get("routingIdentifier") or item.get("id") or f"{flight_number}-{dep_iso}")

    return FlightOption(
        # Suffixed so the two legs of one routing don't collide as React keys.
        id=f"{routing_id}-{direction}",
        airline=carrier_name,
        airline_code=carrier_code,
        flight_number=flight_number,
        departure=FlightSegmentInfo(
            time=dep_time, date=dep_iso, date_label=dep_label,
            airport_code=dep_airport, airport_name=first_seg.get("depAirportName"),
        ),
        arrival=FlightSegmentInfo(
            time=arr_time, date=arr_iso, date_label=arr_label,
            airport_code=arr_airport, airport_name=last_seg.get("arrAirportName"),
        ),
        duration_minutes=duration_minutes,
        stops=num_stops,
        layover_text=layover_text,
        price=round(price, 2),
        currency=item.get("currency") or "USD",
        seats_left=seats_left,
        is_refundable=bool(item.get("refundable")),
        direction=direction,
        price_is_round_trip=is_round_trip,
    )


def get_flight_options(
    origin: str, destination: str, depart_date: str, return_date: Optional[str],
    adults: int, children: int, infants: int,
) -> list[FlightOption]:
    """REST-facing entry point — search_flights_raw() + extract_flight_options(),
    mirroring hotels.py's get_hotel_ui_cards()."""
    raw = search_flights_raw(origin, destination, depart_date, return_date, adults, children, infants)
    if "error" in raw:
        print("Error fetching flight options:", raw.get("details", raw["error"]))
        return []
    return extract_flight_options(raw)

if __name__ == "__main__":
    # Example usage
    flight_options = get_flight_options(
        origin="KUL",
        destination="HND",
        depart_date="2026-09-21",
        return_date="2026-09-22",
        adults=1,
        children=0,
        infants=0,
    )
    for option in flight_options:
        print(option)