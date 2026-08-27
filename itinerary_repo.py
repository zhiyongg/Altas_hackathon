from database import itineraries_collection
import json

def save_itinerary(itinerary: dict):
    result = itineraries_collection.insert_one(itinerary)

    return str(result.inserted_id)


if __name__ == "__main__":
    
    # 1. Read the itinerary data from the JSON file
    with open("itinerary_output.json", "r") as file:
        itinerary = json.load(file)
        
    # 2. Save the itinerary to MongoDB and get the ID
    itinerary_id = save_itinerary(itinerary)

    # 3. Print the confirmation
    print("Saved itinerary:", itinerary_id)