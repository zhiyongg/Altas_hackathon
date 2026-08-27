from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


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
    travel_style: TravelStyle = "moderate"
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
    end_time: datetime   = field(default_factory=lambda: datetime(2000, 1, 1, 21))
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

class ScheduleEntry(BaseModel):
    time: str = Field(description="Arrival time HH:MM")
    name: str = Field(description="Stop name, e.g. 'Lunch: Ichiran' or an attraction name")
    kind: str = Field(description="Stop kind: hotel, attraction or meal")
    duration_min: int = Field(description="Minutes spent at this stop")
    location: Location
    rating: Optional[float] = Field(default=None, description="Google rating, null for hotel stops")
    price_level: Optional[str] = Field(default=None, description="Google price level enum (e.g., PRICE_LEVEL_MODERATE)")
    opening_hours: List[str] = Field(default=[], description="Weekday opening-hour descriptions")

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
    stay_schedule: StaySchedule = Field(description="Check-in/out dates, times, and duration")
    selected_room: RoomOption = Field(description="Selected room type with price and occupancy capacity")
    available_rooms: List[RoomOption] = Field(default=[], description="Other available room options")

# --- Master Itinerary Output Schema ---
class CostBreakdown(BaseModel):
    flights: float
    activities: float
    food_dining: float
    transportation: float
    currency: str

class ItineraryPlan(BaseModel):
    trip_overview: TripOverview
    flights: List[Flight]
    hotels: List[Hotel] = Field(description="Recommended hotels with schedule, room pricing, and capacity")
    daily_itinerary: List[DailyItinerary] = Field(description="Day-by-day plan, same shape as itineraryPlanner.build_itinerary()['days']")
    cost_breakdown: CostBreakdown
    travel_tips: List[str]
