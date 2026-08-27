import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# ==========================================
# 1. Define Output Schemas using Pydantic
# ==========================================
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

# ==========================================
# 2. Define Agent Tools
# ==========================================
@tool
def search_flights_atlas(origin: str, destination: str, date: str) -> str:
    """
    Search for flights using the Atlas Flight API.
    Args:
        origin: Origin airport code (e.g., KUL)
        destination: Destination airport code (e.g., BKI)
        date: Flight date (YYYY-MM-DD)
    """
    # Mock response to simulate Atlas Flight API output
    mock_data = [{
        "direction": "outbound",
        "routing_id": "BN_FSPYtfoNbfpPuNcGwqFFGobcZzNkOQT1OWCpMFk",
        "carrier": "OD",
        "flight_number": "OD1004",
        "cabin": "Q",
        "fare_family": "Value",
        "dep_airport": origin,
        "dep_time": f"{date}T13:30:00",
        "arr_airport": destination,
        "arr_time": f"{date}T16:00:00",
        "duration_minutes": 150,
        "price": {
            "adult_price": 66.76,
            "adult_tax": 15.79,
            "total": 82.55,
            "currency": "USD"
        }
    }]
    return json.dumps(mock_data)

@tool
def nearby_search(location: str, keyword: str) -> str:
    """
    Search for nearby places using Google Places Nearby Search API.
    Args:
        location: The center location (e.g., city name or coordinates)
        keyword: The type of place (e.g., cafe, restaurant)
    """
    # Mock response
    mock_data = [{
        "display_name": "Second Chapter Cafe",
        "primary_type": "cafe",
        "formatted_address": "Level 4, Aurora Place Bandar Bukit Jalil, 1, Persiaran Jalil 1, Bukit Jalil, Kuala Lumpur",
        "rating": 4.9,
        "price_level": "PRICE_LEVEL_MODERATE",
        "location": {"latitude": 3.0528241, "longitude": 101.6691816},
        "opening_hours": ["Monday: 4:00 PM – 12:00 AM", "Tuesday: 4:00 PM – 12:00 AM"]
    }]
    return json.dumps(mock_data)

@tool
def text_search(query: str) -> str:
    """
    Search for places using Google Places Text Search API.
    Args:
        query: The search query (e.g., "best coastal cafes in Kota Kinabalu")
    """
    # Mock response
    mock_data = [{
        "display_name": "Sunset Coast Coffee",
        "primary_type": "cafe",
        "formatted_address": "Jalan Tun Fuad Stephens, Kota Kinabalu",
        "rating": 4.7,
        "price_level": "PRICE_LEVEL_MODERATE",
        "location": {"latitude": 5.9804, "longitude": 116.0735},
        "opening_hours": ["Daily: 8:00 AM – 10:00 PM"]
    }]
    return json.dumps(mock_data)

# ==========================================
# 3. Agent Setup and Execution
# ==========================================
def run_itinerary_agent(user_request: str, custom_messages: str = "") -> str:
    """
    Runs the agent and ensures the output is strictly structured as JSON.
    """
    # Initialize the LLM (Gemini Flash-Lite)
    try:
        # Note: The model name could be 'gemini-2.0-flash-lite-preview-02-05' or 'gemini-1.5-flash' depending on Langchain GenAI SDK state.
        # Fallback to standard gemini-1.5-flash if needed.
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite-preview-02-05", 
            temperature=0.2
        )
    except Exception as e:
        print("Note: If 'gemini-2.0-flash-lite-preview-02-05' is not available, you can change the model string to 'gemini-1.5-flash'.")
        raise e

    tools = [search_flights_atlas, nearby_search, text_search]

    # Use a system message to guide the agent
    system_prompt = '''You are an expert travel planner AI agent.
Your goal is to build an itinerary based on user input. 
You have access to tools for flights (Atlas Flight API) and places (Nearby Search, Text Search).

IMPORTANT:
Once you have gathered the necessary information, formulate your final answer.
Your final answer MUST be ONLY a JSON object that strictly adheres to the requested output format, with no conversational text before or after the JSON.
Do NOT use markdown formatting (like ```json) in your final output, just output a raw JSON string.'''

    agent = create_react_agent(llm, tools, state_modifier=SystemMessage(content=system_prompt))

    # Combine input and custom constraints
    prompt = f"""
    Please create a travel itinerary plan for the following request:
    {user_request}
    
    User Custom Messages/Preferences:
    {custom_messages}
    
    Ensure the JSON structure exactly matches this schema:
    {{
      "trip_overview": {{ "title": "", "origin_city": "", "destination_city": "", "start_date": "", "end_date": "", "total_days": 0, "summary": "", "total_estimated_budget": {{ "amount": 0, "currency": "" }} }},
      "flights": [
        {{ "direction": "", "routing_id": "", "carrier": "", "flight_number": "", "cabin": "", "fare_family": "", "dep_airport": "", "dep_time": "", "arr_airport": "", "arr_time": "", "duration_minutes": 0, "price": {{ "adult_price": 0, "adult_tax": 0, "total": 0, "currency": "" }} }}
      ],
      "daily_itinerary": [
        {{
          "day_number": 0, "date": "", "theme": "",
          "activities": [
            {{
              "order": 0, "time_slot": "", "start_time": "", "end_time": "", "activity_name": "", "description": "",
              "place": {{ "display_name": "", "primary_type": "", "formatted_address": "", "rating": 0, "price_level": "", "location": {{ "latitude": 0, "longitude": 0 }}, "opening_hours": [] }},
              "route_to_next": {{ "distance_meters": 0, "duration": "", "duration_minutes": 0, "travel_mode": "", "encoded_polyline": "" }}
            }}
          ]
        }}
      ],
      "cost_breakdown": {{ "flights": 0, "activities": 0, "food_dining": 0, "transportation": 0, "currency": "" }},
      "travel_tips": []
    }}
    """
    
    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_output = result["messages"][-1].content
    
    # Optional: Clean up Markdown block if the model outputs it anyway
    if final_output.startswith("```json"):
        final_output = final_output.replace("```json", "", 1)
    if final_output.endswith("```"):
        final_output = final_output.rsplit("```", 1)[0]
        
    return final_output.strip()

if __name__ == "__main__":
    import sys
    # Sample usage
    print("Initializing Agent...")
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("\nWARNING: GOOGLE_API_KEY environment variable is not set. Please set it to use Gemini.")
        print("Example: set GOOGLE_API_KEY=your_api_key (Windows) or export GOOGLE_API_KEY=your_api_key (Mac/Linux)\n")
        
    user_req = "Plan a 4-Day Kota Kinabalu Adventure & Dining from KUL to BKI. Start date: 2026-10-22, End date: 2026-10-25."
    user_pref = "Please make sure to add an activity to visit a cafe to unwind after the flight on day 1."

    print("User Request:", user_req)
    print("User Preferences:", user_pref)
    print("\nPlanning itinerary (this may take a few seconds)...")
    
    try:
        final_json = run_itinerary_agent(user_req, user_pref)
        
        # Verify JSON
        try:
            parsed = json.loads(final_json)
            # Validates using the Pydantic schema
            valid_itinerary = ItineraryPlan(**parsed)
            print("\n✅ Successfully generated itinerary JSON:\n")
            print(valid_itinerary.model_dump_json(indent=2))
        except json.JSONDecodeError:
            print("\n❌ Failed to parse output as JSON. Raw output:")
            print(final_json)
        except Exception as validation_err:
            print("\n❌ JSON does not strictly match the Pydantic Schema. Raw output:")
            print(final_json)
            print(f"\nValidation Error: {validation_err}")
    except Exception as e:
        print("\n❌ An error occurred:")
        print(e)
