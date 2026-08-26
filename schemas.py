from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

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

class Location(BaseModel):
    latitude: float
    longitude: float

class Place(BaseModel):
    display_name: str
    primary_type: str
    formatted_address: str
    rating: float
    price_level: str
    location: Location
    opening_hours: List[str]

class RouteToNext(BaseModel):
    distance_meters: int
    duration: str
    duration_minutes: int
    travel_mode: str
    encoded_polyline: str

class Activity(BaseModel):
    order: int
    time_slot: str
    start_time: str
    end_time: str
    activity_name: str
    description: str
    place: Place
    route_to_next: Optional[RouteToNext] = None

class DailyItinerary(BaseModel):
    day_number: int
    date: str
    theme: str
    activities: List[Activity]

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
    check_in_time: str = Field(default="15:00", description="Earliest check-in time (e.g., 15:00 or 03:00 PM)")
    check_out_date: str = Field(description="Check-out date (YYYY-MM-DD)")
    check_out_time: str = Field(default="11:00", description="Standard check-out time (e.g., 11:00 or 11:00 AM)")
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
    hotels: List[Hotel] = Field(default_factory=list, description="Hotels booked for the trip")
    daily_itinerary: List[DailyItinerary]
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
