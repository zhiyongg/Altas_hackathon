from typing import List, Optional
from pydantic import BaseModel, Field

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

class CostBreakdown(BaseModel):
    flights: float
    activities: float
    food_dining: float
    transportation: float
    currency: str

class ItineraryPlan(BaseModel):
    trip_overview: TripOverview
    flights: List[Flight]
    daily_itinerary: List[DailyItinerary]
    cost_breakdown: CostBreakdown
    travel_tips: List[str]
