import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator, Field
from typing import List, Optional, Literal


load_dotenv()

BASE_URL = os.getenv(
    "ATLAS_SANDBOX_BASE_URL", "https://sandbox.atriptech.com"
).rstrip("/")
CLIENT_ID = os.getenv("ATLAS_CLIENT_ID")
CLIENT_SECRET = os.getenv("ATLAS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        f"Missing credentials! CLIENT_ID: {CLIENT_ID}, CLIENT_SECRET: {CLIENT_SECRET}\n"
        "Ensure your .env file is in the project root where you execute the script."
    )
    
HEADERS = {
    "Content-Type": "application/json",
    "Accept":"application/json",
    "Accept-Encoding": "gzip",
    "x-atlas-client-id": CLIENT_ID,
    "x-atlas-client-secret": CLIENT_SECRET,
}


def search_atlas_flights(
    origin: str,
    destination: str,
    departure_date: str,  # Accepts "YYYY-MM-DD" or "YYYYMMDD"
    adults: int = 1,
    trip_type: str = "1",  # "1" = One-Way, "2" = Round-Trip
    airlines: list = None,
) -> dict:
    url = f"{BASE_URL}/search.do"

    # Automatically sanitize date to YYYYMMDD
    clean_date = departure_date.replace("-", "")

    payload = {
        "tripType": str(trip_type),
        "adultNum": adults,
        "childNum": 0,
        "infantNum": 0,
        "fromCity": origin.upper(),
        "fromAirport": "",
        "toCity": destination.upper(),
        "toAirport": "",
        "fromDate": clean_date,
        "retDate": "",
        "airlines": airlines or ["OD"],  # Test carrier in sandbox
        "includeMultipleFareFamily": False,
        "currency": "USD",
        "requestSource": None,
    }

    response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def parse_datetime(dt_str: str) -> tuple[str, str]:
    """Converts 'YYYYMMDDHHMM' (e.g.

    '202610221200') to ('12:00', '22 Oct, Thursday')
    """
    if not dt_str or len(dt_str) < 12:
        return "--:--", "Unknown Date"
    dt = datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
    # dt.day formats the day without leading zeros cross-platform
    date_formatted = f"{dt.day} {dt.strftime('%b, %A')}"
    return dt.strftime("%H:%M"), date_formatted


def extract_flight_ui_cards(res_data: dict) -> list[dict]:
    routings = res_data.get("routings", [])
    cards = []

    for item in routings:
        segments = item.get("fromSegments") or item.get("segments", [])
        if not segments:
            continue

        first_seg = segments[0]
        last_seg = segments[-1]

        # 1. Times & Dates
        dep_raw = first_seg.get("depTime") or first_seg.get("departureTime", "")
        arr_raw = last_seg.get("arrTime") or last_seg.get("arrivalTime", "")
        dep_time, dep_date = parse_datetime(dep_raw)
        arr_time, arr_date = parse_datetime(arr_raw)

        # 2. Airports & Route
        dep_airport = first_seg.get("depAirport") or first_seg.get("departureAirport", "")
        arr_airport = last_seg.get("arrAirport") or last_seg.get("arrivalAirport", "")
        route_str = f"{dep_airport} - {arr_airport}"

        # 3. Safe Airline Carrier Extraction
        carrier_val = first_seg.get("carrier")
        if isinstance(carrier_val, dict):
            carrier_name = carrier_val.get("name") or carrier_val.get("code", "Airline")
        elif isinstance(carrier_val, str) and carrier_val.strip():
            carrier_name = carrier_val
        else:
            carrier_name = (
                first_seg.get("carrierName")
                or first_seg.get("marketingAirline")
                or first_seg.get("operatingAirline")
                or "Airline"
            )

        flight_number = first_seg.get("flightNumber", "")

        # 4. Total Price
        price_info = (
            item.get("adultPrice")
            or item.get("price")
            or item.get("totalPrice")
            or 0
        )
        currency = item.get("currency") or "USD"
        formatted_price = f"${float(price_info):.2f}" if currency == "USD" else f"{currency} {price_info}"

        # 5. Layover / Stop Details
        num_stops = len(segments) - 1
        layover_text = "Direct"
        if num_stops > 0:
            transfer_airport = first_seg.get("arrAirport") or first_seg.get("arrivalAirport", "")
            layover_text = f"{num_stops} stop in {transfer_airport}"

        # 6. Card Record
        card = {
            "routingIdentifier": item.get("routingIdentifier") or item.get("id"),
            "airline": f"{carrier_name} {flight_number}".strip(),
            "route": route_str,
            "departure": {
                "time": dep_time,
                "date": dep_date,
                "airport": first_seg.get("depAirportName") or dep_airport,
            },
            "arrival": {
                "time": arr_time,
                "date": arr_date,
                "airport": last_seg.get("arrAirportName") or arr_airport,
            },
            "layover": layover_text,
            "price": formatted_price,
            "seats_available": f"{item.get('seats') or first_seg.get('seatCount') or 9} Seats Available",
            "refundable": "Refundable" if item.get("refundable") else "Non-refundable",
        }
        cards.append(card)

    return cards

# brenna tried
def _post(path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    if not resp.ok:
        raise AtlasAPIError(f"Atlas API error {resp.status_code} on POST {path}: {resp.text}")
    return resp.json()
    # Note: HTTP 200 does NOT mean success — Atlas returns a business
    # `status` field inside the JSON body (0 = success). Callers must check
    # `status` themselves; this helper only guards transport-level failures.

class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="Origin city/airport IATA code, e.g. 'KUL'")
    destination: str = Field(..., description="Destination city/airport IATA code, e.g. 'NRT'")
    departure_date: str = Field(..., description="'YYYY-MM-DD' or 'YYYYMMDD'")
    adults: int = Field(1, ge=1)
    child_num: int = Field(0, ge=0)
    infant_num: int = Field(0, ge=0)
    trip_type: Literal["1", "2"] = Field("1", description="'1' = one-way, '2' = round-trip")
    return_date: Optional[str] = Field(None, description="Required if trip_type='2'")
    airlines: Optional[list[str]] = Field(None, description="Airline IATA filter, e.g. ['OD']")
    currency: str = Field("USD")
 
    @model_validator(mode="before")
    @classmethod
    def _normalize_dates(cls, data):
        # Accept "YYYY-MM-DD" or "YYYYMMDD" for both dates, always send
        # Atlas the digits-only form it actually expects.
        if isinstance(data, dict):
            data = dict(data)
            if isinstance(data.get("departure_date"), str):
                data["departure_date"] = data["departure_date"].replace("-", "")
            if isinstance(data.get("return_date"), str):
                data["return_date"] = data["return_date"].replace("-", "")
        return data

def search_flights(params: FlightSearchInput) -> dict:
    """
    Search live flight offers (search.do). Each result ("routing") includes
    a routingIdentifier — pass this into verify_flight_offer before booking,
    since prices here may be cached and are not guaranteed.
    """
    payload = {
        "tripType": params.trip_type,
        "adultNum": params.adults,
        "childNum": params.child_num,
        "infantNum": params.infant_num,
        "fromCity": params.origin.upper(),
        "fromAirport": "",
        "toCity": params.destination.upper(),
        "toAirport": "",
        "fromDate": params.departure_date,
        "retDate": params.return_date or "",
        "airlines": params.airlines or ["OD"],  # OD = test carrier in sandbox
        "includeMultipleFareFamily": False,
        "currency": params.currency,
        "requestSource": None,
    }
    raw = _post("/search.do", payload)
    return json.loads(json.dumps(raw, default=str))


# --------------------------------------------------------------------------- #
# 2. Verify
# --------------------------------------------------------------------------- #
 
def verify_flight_offer(routing_identifier: str, max_response_time: int) -> dict:
    """
    Re-verify live fare, routing, and passenger requirements (verify.do).
    Must be called after search_flights and before create_flight_order.
 
    routingIdentifier is valid up to 6 hours but should be verified as early
    as possible — price-change risk rises with elapsed time.
 
    Response's `bookingRequirement` tells you which passenger fields
    (birthday, passport, nationality, etc.) are actually mandatory for THIS
    route — read it dynamically rather than assuming a fixed schema.
    Check `priceChange.isPriceChange` before proceeding to order.
 
    Status codes to branch on: 202 (routingIdentifier expired), 213 (flight
    info changed), 207/210 (sold out) — all mean re-search, not retry.
    """
    payload = {
        "routingIdentifier": routing_identifier,
        "maxResponseTime": max_response_time,
    }
    return _post("/verify.do", payload)
 
 
# --------------------------------------------------------------------------- #
# 3. Create order
# --------------------------------------------------------------------------- #
 
class PassengerInput(BaseModel):
    name: str = Field(..., description="'FamilyName/GivenName' — exact slash format required")
    passenger_type: Literal[0, 1, 2] = Field(0, description="0=Adult, 1=Child, 2=Infant")
    gender: Literal["M", "F"]
    birthday: Optional[str] = Field(None, description="YYYYMMDD, if required by bookingRequirement")
    card_type: Optional[Literal["PP", "GA", "TW", "TB", "HY"]] = None
    card_num: Optional[str] = None
    card_issue_place: Optional[str] = Field(None, description="ISO 3166-1 alpha-2")
    card_expired: Optional[str] = Field(None, description="YYYYMMDD")
    nationality: Optional[str] = Field(None, description="ISO 3166-1 alpha-2")
 
 
class CreateOrderInput(BaseModel):
    session_id: str = Field(..., description="sessionId from verify_flight_offer")
    passengers: list[PassengerInput]
    contact_name: str = Field(..., description="'FamilyName/GivenName', Latin letters only")
    contact_email: str
    contact_mobile: str = Field(..., description="'<country_code>-<number>', e.g. '0060-123456789'")
    card_type: Optional[Literal["Amex", "Visa", "MasterCard", "JCB", "Discover", "DinersClub"]] = Field(
        None, description="Required only for MoR payment method"
    )
 
 
def create_flight_order(params: CreateOrderInput) -> dict:
    """
    Create a booking order from a verified session (order.do). Must be
    called after verify_flight_offer.
 
    Only send passenger fields that bookingRequirement (from verify) flags
    as required for this route — don't hardcode a fixed field set.
 
    On success: orderNo (needed for pay), pnrCode (Atlas's own reference),
    tktLimitTime (payment deadline, SGT/GMT+8) — a flat 30-minute window,
    no separate hold/extend call.
 
    Status codes to branch on: 308 (price changed since verify — re-verify
    and re-order), 318 (duplicate booking — check duplicateOrders),
    302/315 (sold out between verify and order).
    """
    payload = {
        "sessionId": params.session_id,
        "passengers": [
            {
                "name": p.name,
                "passengerType": p.passenger_type,
                "gender": p.gender,
                **({"birthday": p.birthday} if p.birthday else {}),
                **({"cardType": p.card_type} if p.card_type else {}),
                **({"cardNum": p.card_num} if p.card_num else {}),
                **({"cardIssuePlace": p.card_issue_place} if p.card_issue_place else {}),
                **({"cardExpired": p.card_expired} if p.card_expired else {}),
                **({"nationality": p.nationality} if p.nationality else {}),
            }
            for p in params.passengers
        ],
        "contact": {
            "name": params.contact_name,
            "email": params.contact_email,
            "mobile": params.contact_mobile,
        },
    }
    if params.card_type:
        payload["payment"] = {"cardType": params.card_type}
 
    return _post("/order.do", payload)
 
 
# --------------------------------------------------------------------------- #
# 4. Confirm order (FR / AirFrance integration only — conditional)
# --------------------------------------------------------------------------- #
def confirm_flight_order(order_no: str, redirect_uri: Optional[str] = None, iframe: Optional[bool] = None) -> dict:
    """
    Get a confirmation-page link for the order (orderCommit.do).
 
    CONDITIONAL — only required for FR (Air France/KLM) integrations. For
    other airlines, skip this and go straight from create_flight_order to
    pay_flight_order.
    """
    payload = {"orderNo": order_no}
    if redirect_uri:
        payload["redirectUri"] = redirect_uri
    if iframe is not None:
        payload["iframe"] = str(iframe).lower()
    return _post("/orderCommit.do", payload)
 
 
# --------------------------------------------------------------------------- #
# 5. Pay
# --------------------------------------------------------------------------- #
 
class CreditCardInput(BaseModel):
    card_number: str
    card_cvv: str
    card_expire_month: str = Field(..., description="'01'-'12'")
    card_expire_year: str = Field(..., description="Two digits, e.g. '26' for 2026")
    card_holder_last_name: str
    card_holder_first_name: str
    card_holder_country: str = Field(..., description="ISO 3166-1 alpha-2")
    card_holder_province: str = Field(..., description="Two-letter code, e.g. 'CA'")
    card_holder_city: str
    card_holder_post_code: str
    card_holder_address: str
    reusable: bool = Field(False, description="true = multi-use card, false = single-use")
 

def pay_flight_order(order_no: str, payment_method: Literal[1, 3, 4, 5], credit_card: Optional[CreditCardInput] = None, client_order_no: Optional[str] = None) -> dict:
    """
    Pay for a created order (pay.do). Must be called after
    create_flight_order (and confirm_flight_order, if required for this
    airline).
 
    Payment success does NOT guarantee ticketing is complete — always
    follow up with query_flight_order to confirm final ticketing state.
 
    Sandbox test cards: Visa 4532015112830366 | Mastercard 5555555555554444 |
    Amex 378282246310005 | Discover 6011111111111117 | JCB 3566002020360505
 
    Status codes: 401 (past payment deadline), 404 (already paid), 406
    (payment already in progress — wait, don't retry), 414 (card brand
    mismatch vs cardType sent at order creation).
    """
    payload: dict = {
        "orderNo": order_no,
        "paymentMethod": payment_method,
    }
    if credit_card:
        cc = credit_card
        payload["creditCard"] = {
            "cardNumber": cc.card_number,
            "cardCVV": cc.card_cvv,
            "cardExpireMonth": cc.card_expire_month,
            "cardExpireYear": cc.card_expire_year,
            "cardHolderLastName": cc.card_holder_last_name,
            "cardHolderFirstName": cc.card_holder_first_name,
            "cardHolderCountry": cc.card_holder_country,
            "cardHolderProvince": cc.card_holder_province,
            "cardHolderCity": cc.card_holder_city,
            "cardHolderPostCode": cc.card_holder_post_code,
            "cardHolderAddress": cc.card_holder_address,
            "reusable": cc.reusable,
        }
    if params.client_order_no:
        payload["clientOrderNo"] = client_order_no
 
    return _post("/pay.do", payload)
 
 
# --------------------------------------------------------------------------- #
# 6. Query order
# --------------------------------------------------------------------------- #
 
class QueryOrderInput(BaseModel):
    order_no: Optional[str] = Field(None, description="Required if pnr_code not given")
    pnr_code: Optional[str] = Field(None, description="Atlas's own PNR; required if order_no not given")
 
    @model_validator(mode="after")
    def _require_one_identifier(self):
        if not self.order_no and not self.pnr_code:
            raise ValueError("query_flight_order requires order_no or pnr_code")
        return self
 
 
def query_flight_order(params: QueryOrderInput) -> dict:
    """
    Check order/ticketing status (queryOrderDetails.do).
 
    orderStatus: 0=unpaid, 1=ticketing-in-process, 2=ticketed, -3=cancelled.
    ticketStatus: 0=not issued, 1=issued.
 
    A failed/cancelled order (errorCode/errorMessage/airlineMessage explain
    why) should be escalated to a human rather than retried automatically —
    Atlas routes genuine failures to an internal exception queue.
    """
    payload = {}
    if params.order_no:
        payload["orderNo"] = params.order_no
    if params.pnr_code:
        payload["pnrCode"] = params.pnr_code
    return _post("/queryOrderDetails.do", payload)


if __name__ == "__main__":
    # res = search_atlas_flights("KUL", "BKI", "2026-10-22")
    # print(f"Status: {res.get('status')} | Message: {res.get('msg')}")
    # print(f"Found {len(res.get('routings', []))} flights.")

    # flight_cards = extract_flight_ui_cards(res)
    # print(f"Extracted {len(flight_cards)} flight cards for UI:")
    # print(json.dumps(flight_cards, indent=2))
    
    # search_result = search_flights(
    #     FlightSearchInput(
    #         origin="KUL",
    #         destination="BKI",
    #         departure_date="2026-10-22",
    #         adults=1,
    #         trip_type="1",
    #         currency="USD",
    #     )
    # )
    # print(json.dumps(search_result, indent=2)[:8000])
    
    
    # --------------------------------------------------------------------------- #
    # Step 2 — Verify
    # Copy a routingIdentifier from the search_result printed above and paste it
    # in below before uncommenting.
    # --------------------------------------------------------------------------- #
    # verify_result = verify_flight_offer(
    #     routing_identifier="S1VMX0JLSV8xXzIwMjYxMDIyX18xXzBfMHxaWUUyNDc5OV9hcGlfMXwxfDE4NS42OF8xODUuNjhfMTM1LjA1XzAuMDBfNTA2LjQxX1VTRHxLVUxfQktJXzFfMjAyNjEwMjJfXzFfMF8wXktVTC1PRDEwMDQtUS1CS0ktMjAyNjEwMjIxMzMwLTIwMjYxMDIyMTYwMC1TdXBlciBTYXZlci0xLV4xODUuNjhfMTg1LjY4XzEzNS4wNV8wLjAwXzUwNi40MV5BT0RfQU9EXl5BT0QxS1VMQktJMjAwMjAyNjEwMjJeU0dEXjc0OS4xMl43NDkuMTJeNTQ0LjgzXjF8MHwyMDI2MDgyNjA4MzEzNHwwfDE3ODc3MDQyOTQzODdiYjg0MGE4M3x8fHx8MC4wMHwzfDB8fG5vcm1hbHxmYWxzZXwyMDI2LTA4LTI2IDA4OjIzOjAwfA==.WFOhDOmsPTxDuufzvnEzgpLJYbD+Z7Lza67mj3JItXQ=",
    #     max_response_time=15000,
    # )
    # print(json.dumps(verify_result, indent=2)[:8000])
    
    
    # # --------------------------------------------------------------------------- #
    # # Step 3 — Create order
    # # Copy sessionId from verify_result above. Check verify_result["bookingRequirement"]
    # # to see which passenger fields are actually mandatory for this route before
    # # filling in PassengerInput — the fields below are a reasonable starting
    # # guess (birthday/nationality), not guaranteed to be what this route needs.
    # # --------------------------------------------------------------------------- #
    
    # order_result = create_flight_order(
    #     CreateOrderInput(
    #         session_id="556a0085-a98f-4236-866a-22d6a9a3638d",
    #         passengers=[
    #             PassengerInput(
    #                 name="Tan/Ben",
    #                 passenger_type=0,
    #                 gender="M",
    #                 birthday="19950101",
    #                 nationality="MY",
    #             )
    #         ],
    #         contact_name="Tan/Ben",
    #         contact_email="ben@example.com",
    #         contact_mobile="0060-123456789",
    #     )
    # )
    # print("=== create_flight_order ===")
    # print(json.dumps(order_result, indent=2)[:8000])
    
    
    # # --------------------------------------------------------------------------- #
    # # Step 3b — Confirm order (SKIP unless this route is FR/AirFrance — check the
    # # routing's carrier from search_result/verify_result first)
    # # --------------------------------------------------------------------------- #
    
    confirm_result = confirm_flight_order(order_no="TESTA20260826084623110")
    print("=== confirm_flight_order ===")
    print(json.dumps(confirm_result, indent=2)[:8000])
    
    
    # # --------------------------------------------------------------------------- #
    # # Step 4 — Pay
    # # Copy orderNo from order_result above. Using paymentMethod=1 (balance) here
    # # since it needs no card details — switch to 3/5 + CreditCardInput only once
    # # you've confirmed balance payment works.
    # # --------------------------------------------------------------------------- #
    
    # pay_result = pay_flight_order(
    #         order_no="TESTA20260826084623110",
    #         payment_method=1,
    #     )
    # print("=== pay_flight_order ===")
    # print(pay_result)
    
    # #Example with a sandbox test card instead (paymentMethod=3, VCC passthrough):
    # pay_result = pay_flight_order(
    #     PayOrderInput(
    #         order_no="TESTA20260826084623110",
    #         payment_method=3,
    #         credit_card=CreditCardInput(
    #             card_number="4532015112830366",  # Atlas sandbox Visa test card
    #             card_cvv="123",
    #             card_expire_month="12",
    #             card_expire_year="28",
    #             card_holder_last_name="Tan",
    #             card_holder_first_name="Ben",
    #             card_holder_country="MY",
    #             card_holder_province="KL",
    #             card_holder_city="Kuala Lumpur",
    #             card_holder_post_code="50000",
    #             card_holder_address="1 Jalan Test",
    #         ),
    #     )
    # )
    # print("=== pay_flight_order (VCC) ===")
    # print(pay_result)
    
    
    # # --------------------------------------------------------------------------- #
    # # Step 5 — Query order status
    # # --------------------------------------------------------------------------- #
    
    query_result = query_flight_order( order_no = "TESTA20260826084623110")
    print("=== query_flight_order ===")
    print(json.dumps(query_result, indent=2)[:8000])
