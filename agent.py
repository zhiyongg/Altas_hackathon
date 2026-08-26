import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Load .env from the script's directory
load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Import schemas and tools from new modules
from schemas import ItineraryPlan
from tools import (
    search_flights_atlas, nearby_search, text_search,
    lookup_destination, search_hotels, meta_search,
    get_hotel_details, get_hotel_prices,
)

# ==========================================
# Agent Setup and Execution
# ==========================================
def run_itinerary_agent(user_request: str, custom_messages: str = "") -> str:
    """
    Runs the agent and ensures the output is strictly structured as JSON.
    """
    # Initialize the LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            temperature=0.2
        )
    except Exception as e:
        raise e

    tools = [
        search_flights_atlas, nearby_search, text_search,
        lookup_destination, search_hotels, meta_search,
        get_hotel_details, get_hotel_prices,
    ]

    # Use a system message to guide the agent
    system_prompt = '''You are an expert travel planner AI agent.
        Your goal is to build a rich, detailed itinerary based on user input.
        You have access to tools for flights (Atlas Flight API), places/attractions (Google Nearby Search, Text Search),
        and hotels/accommodation (StayAPI).

        CRITICAL — ITINERARY DENSITY:
        Every single day MUST contain AT LEAST 5 activities following this pattern:

        Each day structure:
        1. Breakfast restaurant (morning)
        2. Attraction / activity (late morning)
        3. Attraction / activity (midday)
        4. Lunch restaurant (early afternoon)
        5. Attraction / activity (afternoon)
        6. Attraction / activity (late afternoon)
        7. Dinner restaurant (evening)

        That is 7 activities minimum per full day. Arrival/departure days may have fewer
        but still need at least 3 activities if there is time.

        You MUST call text_search or nearby_search MULTIPLE times per day to find
        different restaurants and attractions. Do NOT reuse the same tool result for
        multiple activities. Each activity needs its own unique real place.

        ANTI-PATTERNS — NEVER do these:
        - A day with only 1 or 2 activities
        - Repeating the same place across different time slots
        - Listing a restaurant without actually searching for one

        FLIGHT SEARCH:
        1. Use search_flights_atlas to find real outbound and return flights matching the trip dates.
        2. Include flight details, pricing, and timing in the final output.

        HOTEL SEARCH (do this AFTER planning activities, keep it efficient):
        1. Use lookup_destination to get dest_id and dest_type for the destination city.
        2. Use search_hotels with the trip dates to find a suitable hotel.
        3. Pick ONE best hotel based on rating, price, and proximity to planned activities.
        4. Use get_hotel_prices to fetch all room types. Select the best room as "selected_room"
           and include other room types detail in "available_rooms".
        5. Populate the hotel\'s "stay_schedule" with check-in/out dates, times, and total nights.

        IMPORTANT:
        - All prices and currencies must come from real tool responses.
        - The final answer must include flights, a full multi-activity daily itinerary,
          one hotel with room details, other available rooms, a cost breakdown, and travel tips.
        '''

    agent = create_react_agent(llm, tools, prompt=SystemMessage(content=system_prompt))

    # Combine input and custom constraints
    prompt = f"""
    Please create a travel itinerary plan for the following request:
    {user_request}
    
    User Custom Messages/Preferences:
    {custom_messages}
    """
    
    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    final_output = result["messages"][-1].content
    
    # Enforce the Pydantic schema using with_structured_output.
    # Pass the full conversation (including tool results) so the parser
    # has access to all the activity data the agent gathered.
    structured_llm = llm.with_structured_output(ItineraryPlan)
    all_messages = result["messages"]
    try:
        extraction_prompt = (
            "Below is the complete conversation including all tool results. "
            "Extract the full itinerary — every single activity for every day — "
            "into the schema. Do NOT skip or summarize any activities.\n\n"
            + "\n".join(m.content for m in all_messages if hasattr(m, "content") and m.content)
        )
        structured_result = structured_llm.invoke(extraction_prompt)
        return structured_result.model_dump_json(indent=2)
    except Exception as e:
        # Fallback
        print(f"Failed to parse into Pydantic schema: {e}")
        return final_output

if __name__ == "__main__":
    print("Initializing Agent...")
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("\nWARNING: GOOGLE_API_KEY environment variable is not set. Please set it to use Gemini.")
    if "STAYAPI_KEY" not in os.environ:
        print("\nWARNING: STAYAPI_KEY environment variable is not set. Hotel search tools will not work.")
        
    user_req = "Plan a 4-Day Kota Kinabalu Adventure & Dining from KUL to BKI. Start date: 2026-10-22, End date: 2026-10-25."
    user_pref = "Please make sure to add an activity to visit a cafe to unwind after the flight on day 1."

    print("User Request:", user_req)
    print("User Preferences:", user_pref)
    print("\nPlanning itinerary (this may take a few seconds)...")
    
    try:
        final_json = run_itinerary_agent(user_req, user_pref)
        print("\n✅ Successfully generated itinerary JSON:\n")
        print(final_json)
    except Exception as e:
        print("\n❌ An error occurred:")
        print(e)
