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
from schemas import ItineraryPlan, PlannerItinerary
from tools import search_flights_atlas, nearby_search, text_search, plan_itinerary

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

    tools = [search_flights_atlas, nearby_search, text_search, plan_itinerary]

    # Use a system message to guide the agent
    system_prompt = '''You are an expert travel planner AI agent.
        Your goal is to build an itinerary based on user input. 
        You have access to tools for flights (Atlas Flight API), places (Nearby Search, Text Search), and plan_itinerary to generate a daily itinerary.

        IMPORTANT WORKFLOW:
        1. Search for flights and hotels using the flight and places search tools.
        2. Call the `plan_itinerary` tool, passing in the flight and hotel parameters, as well as user preferences.
        3. Combine the generated itinerary with the flights and hotel data to structure the output.
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
    
    # 1. Find the raw tool output from plan_itinerary in the message stream
    raw_planner_json = None
    for msg in reversed(result["messages"]):
        if hasattr(msg, "name") and msg.name == "plan_itinerary":
            raw_planner_json = json.loads(msg.content)
            break

    # 2. Extract structured summary for overview, flights, hotels, and costs
    structured_llm = llm.with_structured_output(ItineraryPlan)
    final_output = result["messages"][-1].content
    structured_result = structured_llm.invoke(f"Extract overview, flights, hotels, cost breakdown, and tips from:\n{final_output}")

    # 3. Direct Inject: Preserve the deterministic itinerary without LLM loss
    if raw_planner_json and "days" in raw_planner_json:
        structured_result.daily_itinerary = PlannerItinerary.model_validate(raw_planner_json)

    return structured_result.model_dump_json(indent=2)

if __name__ == "__main__":
    print("Initializing Agent...")
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("\nWARNING: GOOGLE_API_KEY environment variable is not set. Please set it to use Gemini.")
        
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
