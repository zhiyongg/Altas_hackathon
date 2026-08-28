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
    search_flights_atlas,
    nearby_search,
    text_search,
    plan_itinerary,
    get_hotel_prices,
    search_hotels,
    lookup_destination,
)

# ==========================================
# Agent Setup and Execution
# ==========================================
def _content_to_text(content) -> str:
    """Normalize a LangChain message's .content into plain text.

    langchain_google_genai (Gemini) can return content as a list of parts
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string,
    especially on messages that involved a tool call. That was crashing
    the "\n".join(...) below with "expected str instance, list found",
    and — once caught — the fallback final_output was also a list,
    crashing json.loads() upstream in api.py with
    "the JSON object must be str, bytes or bytearray, not list".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content else ""


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
        search_flights_atlas,
        nearby_search,
        text_search,
        plan_itinerary,
        lookup_destination,
        get_hotel_prices,
        search_hotels,
    ]

    # Use a system message to guide the agent
    system_prompt = '''You are an expert travel planner AI agent.
        Your goal is to build an itinerary based on user input. 
        You have access to tools for flights (Atlas Flight API), hotels (StayAPI), places (Nearby Search, Text Search), and plan_itinerary to generate a daily itinerary.

        IMPORTANT WORKFLOW:
        1. Search for flights using the flight search tool.
        2. For hotels: call `lookup_destination` first to get a dest_id, then `search_hotels` with that dest_id,
           then `get_hotel_prices` for the specific hotel(s) you want to recommend so selected_room has real pricing.
           When you finalize a hotel choice, keep the exact dest_id and dest_type you used to find it — the final
           output schema needs both stamped onto that hotel (dest_id/dest_type fields) so the frontend can search
           that same destination again later if the user wants to change hotels.
        3. Search for places using the places search tools.
        4. Call the `plan_itinerary` tool, passing in the flight and hotel parameters, as well as user preferences.
        5. Combine the generated itinerary with the flights and hotel data to structure the output.
        Once you have gathered the necessary information, formulate your final answer so it can be parsed into the `ItineraryPlan` schema.
        Ensure you look up real flights and real places using your tools. The price and currency must follow back the tool responses.
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
    final_output = _content_to_text(result["messages"][-1].content)
    
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
            + "\n".join(_content_to_text(m.content) for m in all_messages if hasattr(m, "content") and m.content)
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