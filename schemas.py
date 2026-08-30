from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# Allowed values shared by itineraryPlanner.py (TRAVEL_STYLE_CAPACITY / _SPEED /
# BUDGET_PRICE_LEVELS keys) and by agent.py when it fills a TripConfig. The
# planner falls back to a default on any other string, so keep these in sync.
TravelStyle   = Literal["relaxed", "moderate", "packed"]
TransportMode = Literal["DRIVE", "TRANSIT", "WALK", "BICYCLE"]
BudgetLevel   = Literal["low", "medium", "high"]


# --- Overview Models ---
class Budget(BaseModel):
    amount: float = Field(description="Total amount")
    currency: str = Field(description="Currency code (e.g., USD)")

class TripOverview(BaseModel):
    title: str = Field(description="Catchy title for the trip")
    origin_city: str = Field(description="Origin city code/name")
    destination_city: str = Field(description="Destination city code/name")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")
    total_days: int = Field(description="Total number of days")
    summary: str = Field(description="Brief summary of the trip")
    total_estimated_budget: Budget

# --- Flight Models ---

class FlightPrice(BaseModel):
    adult_price: float
    adult_tax: float
    total: float
    currency: str

class Flight(BaseModel):
    direction: str = Field(description="outbound or return")
    routing_id: str
    carrier: str
    flight_number: str
    cabin: str
    fare_family: str
    dep_airport: str
    dep_time: str
    arr_airport: str
    arr_time: str
    duration_minutes: int
    price: FlightPrice

# --- Itinerary Planner Working Structures ---
# Plain dataclasses used by itineraryPlanner.py while it builds the plan.
# They are deliberately NOT pydantic models: during the pipeline they carry live
# datetime and Place objects, and are only converted to the pydantic models
# below when build_itinerary() emits its JSON.

MIN_PER_DAY          = 3
MAX_PER_DAY          = 8

@dataclass
class TripConfig:
    destination: str
    start_date: str                     
    end_date: str
    arrival_datetime: str               
    departure_datetime: str
    hotel: dict                         # Location-shaped: {"name", "latitude", "longitude"}
    # Everything below has a default so agent.py can build a config from just
    # destination, dates and a hotel, and let the planner supply the rest.
    airport: dict = field(default_factory=dict)   # Location-shaped, unused by the pipeline today
    travel_style: TravelStyle = "packed"
    transport_mode: TransportMode = "TRANSIT"
    group_size: int = 2
    budget: BudgetLevel = "medium"
    check_in_time: str = "15:00"        # 24h HH:MM — parsed with strptime("%H:%M")
    check_out_time: str = "11:00"       # 24h HH:MM — parsed with strptime("%H:%M")
    selected_preferences: list = field(default_factory=list)  # keys of THEME_TO_TYPES
    preferences: dict = field(default_factory=dict)           # filled by the planner
    custom_vibe: str = ""

@dataclass
class Place:
    id: str
    name: str
    location: dict                      
    types: list[str]
    primary_type: str = ""
    rating: float = 0.0
    user_rating_count: int = 0
    price_level: str = ""
    opening_hours: list = field(default_factory=list)
    visit_duration_min: int = 60
    score: float = 0.0
    source: str = ""

@dataclass
class DayPlan:
    day_index: int
    date: str
    day_type: str                       
    start_time: datetime = field(default_factory=lambda: datetime(2000, 1, 1, 9))
    end_time: datetime   = field(default_factory=lambda: datetime(2000, 1, 1, 22))
    base_location: dict  = field(default_factory=dict)
    attractions: list    = field(default_factory=list)
    meals: dict          = field(default_factory=dict)
    sequence: list       = field(default_factory=list)
    schedule: list       = field(default_factory=list)
    valid: bool          = True
    violations: list     = field(default_factory=list)
    
    # ── FIX: Add the missing capacity tracking ──
    capacity_min: int    = MIN_PER_DAY
    capacity_max: int    = MAX_PER_DAY


# --- Places, Activities & Route Models ---
# Pydantic mirror of the JSON returned by itineraryPlanner.build_itinerary(),
# so a planner result can be handed straight to ItineraryPlan.daily_itinerary.

class Location(BaseModel):
    latitude: float = Field(description="Geographical latitude coordinate")
    longitude: float = Field(description="Geographical longitude coordinate")
    name: Optional[str] = Field(default=None, description="Place label, present on hotel stops")
    address: Optional[str] = Field(default=None, description="Physical address of the location")

class ScheduleEntry(BaseModel):
    time: str = Field(description="Arrival time HH:MM")
    name: str = Field(description="Stop name, e.g. 'Lunch: Ichiran' or an attraction name")
    kind: str = Field(description="Stop kind: hotel, attraction or meal")
    duration_min: int = Field(description="Minutes spent at this stop")
    location: Location
    rating: Optional[float] = Field(default=None, description="Google rating, null for hotel stops")
    price_level: Optional[str] = Field(default=None, description="Google price level enum (e.g., PRICE_LEVEL_MODERATE)")
    opening_hours: List[str] = Field(default=[], description="Weekday opening-hour descriptions")
    transit_to_next: Optional[dict] = Field(default=None, description="Transportation to next activity")

class DayMeals(BaseModel):
    breakfast: Optional[str] = Field(default=None, description="Breakfast venue name")
    lunch: Optional[str] = Field(default=None, description="Lunch venue name")
    dinner: Optional[str] = Field(default=None, description="Dinner venue name")

class DailyItinerary(BaseModel):
    day: int = Field(description="Zero-based day index")
    date: str = Field(description="Date YYYY-MM-DD")
    type: str = Field(description="Day type: arrival, normal or departure")
    valid: bool = Field(default=True, description="Whether the day passed constraint validation")
    schedule: List[ScheduleEntry] = Field(default=[], description="Ordered stops for the day")
    attractions: List[str] = Field(default=[], description="Attraction names assigned to this day")
    meals: DayMeals = Field(default_factory=DayMeals, description="Chosen meal venue per slot")

class PlannerItinerary(BaseModel):
    """Full return value of itineraryPlanner.build_itinerary().

    Lets agent.py validate a planner run before slotting `days` straight into
    ItineraryPlan.daily_itinerary. On failure the planner returns only `error`.
    """
    destination: str = Field(default="", description="Destination as passed in TripConfig")
    start: str = Field(default="", description="Trip start date YYYY-MM-DD")
    end: str = Field(default="", description="Trip end date YYYY-MM-DD")
    days: List[DailyItinerary] = Field(default=[], description="Day-by-day plan")
    error: Optional[str] = Field(default=None, description="Set when discovery produced no candidates")


# --- Hotel & Room Models ---

class RoomOption(BaseModel):
    room_name: str = Field(description="Room type name (e.g., Superior Double Room)")
    max_occupancy: int = Field(description="Number of people suitable for this room (e.g., 2)")
    price_per_night: float = Field(description="Nightly price numeric value")
    total_price: float = Field(description="Total price for the full stay")
    currency: str = Field(default="USD", description="Currency code (e.g., USD)")
    breakfast_included: bool = Field(default=False, description="Breakfast included flag")
    is_refundable: bool = Field(default=False, description="Refundable policy status")
    cancellation_policy: Optional[str] = Field(default=None, description="Cancellation details")

class StaySchedule(BaseModel):
    check_in_date: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_in_time: str = Field(default="15:00", description="Earliest check-in time, 24-hour HH:MM (e.g., 15:00)")
    check_out_date: str = Field(description="Check-out date (YYYY-MM-DD)")
    check_out_time: str = Field(default="11:00", description="Standard check-out time, 24-hour HH:MM (e.g., 11:00)")
    total_nights: int = Field(description="Total number of nights")

class Hotel(BaseModel):
    hotel_id: str = Field(description="Unique hotel identifier")
    name: str = Field(description="Name of the hotel")
    address: Optional[str] = Field(default=None, description="Hotel street address")
    city: Optional[str] = Field(default=None, description="City location")
    latitude: Optional[float] = Field(default=None, description="Geographical latitude coordinate")
    longitude: Optional[float] = Field(default=None, description="Geographical longitude coordinate")
    star_rating: Optional[int] = Field(default=None, description="Hotel star rating (1-5)")
    rating: Optional[float] = Field(default=None, description="Guest review score, e.g. 8.7")
    review_count: Optional[int] = Field(default=None, description="Number of guest reviews")
    image_url: Optional[str] = Field(default=None, description="Main hotel photo URL")
    is_sponsored: bool = Field(default=False, description="True if this came from the sponsored/featured list")
    dest_id: Optional[str] = Field(default=None, description="StayAPI destination ID this hotel was searched under - needed so the frontend can re-search this destination later (e.g. Change Accommodation)")
    dest_type: Optional[str] = Field(default=None, description="StayAPI destination type (e.g. CITY, DISTRICT) matching dest_id")
    stay_schedule: StaySchedule = Field(description="Check-in/out dates, times, and duration")
    selected_room: RoomOption = Field(description="Selected room type with price and occupancy capacity")
    available_rooms: List[RoomOption] = Field(default=[], description="Other available room options")

# --- Master Itinerary Output Schema ---
class CostBreakdown(BaseModel):
    flights: float
    hotels: float = 0.0
    activities: float
    food_dining: float
    transportation: float
    currency: str

class ItineraryPlan(BaseModel):
    trip_overview: TripOverview
    flights: List[Flight]
    hotels: List[Hotel] = Field(description="Recommended hotels with schedule, room pricing, and capacity")
    daily_itinerary: PlannerItinerary = Field(description="Day-by-day plan, containing destination, dates, and the 'days' array matching the planner output.")
    cost_breakdown: CostBreakdown
    travel_tips: List[str]

class HotelSearchInput(BaseModel):
    dest_id: str = Field(..., description="From lookup_destination")

    @model_validator(mode="before")
    @classmethod
    def _coerce_dest_id(cls, data):
        # lookup_destination returns dest_id as an int (e.g. -372490);
        # the search endpoint just wants it as a query string either way.
        if isinstance(data, dict) and isinstance(data.get("dest_id"), int):
            data = {**data, "dest_id": str(data["dest_id"])}
        return data

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


class HotelChangeRequest(BaseModel):
    dest_id: str
    dest_type: str = "CITY"
    checkin: str
    checkout: str
    adults: int
    rooms: int
    children: int
    children_ages: Optional[list[int]] = None
    # How many of the top (cheapest-first, after sort) searched hotels to
    # backfill with real per-room pricing via get_hotel_prices. Each one is
    # an extra StayAPI call, so keep this small.
    price_lookup_limit: int = 5


class TripMemberInput(BaseModel):
    id: str
    name: str
    isCurrentUser: bool = False

class CreateCheckoutSessionsRequest(BaseModel):
    trip_id: str
    destination: str
    total_cost: float
    members: list[TripMemberInput]
    split: bool = True
    currency: str = "usd"
    # Where the browser lands after paying/cancelling. The frontend should
    # point these at its own Finalize & Pay page (Stripe appends
    # ?session_id=... to success_url automatically).
    success_url: str
    cancel_url: str

class FlightSegmentInfo(BaseModel):
    time: str
    date: str            # ISO YYYY-MM-DD
    date_label: str       # "6 Jan, Tuesday" for display
    airport_code: str
    airport_name: Optional[str] = None


class FlightOption(BaseModel):
    id: str
    airline: str
    airline_code: Optional[str] = None
    flight_number: str
    departure: FlightSegmentInfo
    arrival: FlightSegmentInfo
    duration_minutes: Optional[int] = None
    stops: int = 0
    layover_text: str = "Direct"
    price: float
    currency: str = "USD"
    seats_left: Optional[int] = None
    is_refundable: bool = False


class FlightSearchRequest(BaseModel):
    origin: str
    destination: str
    depart_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    infants: int = 0


class FlightChangeApplyRequest(BaseModel):
    session_id: str = "testing"
    output_path: Optional[str] = None
    flight: FlightOption
    trip_config: Optional[dict] = None


# --- Activity Models ---
# Field names are deliberately camelCase (not this file's usual snake_case)
# to match the ActivityOption interface already defined in the frontend's
# types.ts, which mirrors its existing camelCase mock data shape rather
# than the snake_case convention used by RoomOption/FlightOption above.
class ActivityOption(BaseModel):
    id: str = Field(description="Google Place ID")
    title: str
    category: str = Field(description="Coarse category bucket: Dining, Cafe, Culture, Nature, Shopping, or Activity")
    categoryIcon: str = ""
    rating: float = 0.0
    reviewsCount: Optional[int] = None
    distance: str = Field(default="—", description="Human-readable distance from the search origin, e.g. '450m' or '2.3km'")
    priceLabel: str = ""
    image: str = Field(default="", description="Places API photo URL; empty string if the place has no photo")
    isSponsored: bool = False
    description: str = ""
    timeSlot: Optional[str] = None
    isFavorite: bool = False
    latitude: Optional[float] = Field(default=None, description="Needed for mapCoords/location when the activity is added to the itinerary")
    longitude: Optional[float] = None
    address: Optional[str] = None


class ActivitySearchRequest(BaseModel):
    query: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None