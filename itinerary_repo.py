import json
from pathlib import Path
from typing import Any

from database import has_mongo, itineraries_collection

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "itinerary_output.json"


def save_itinerary(itinerary: dict[str, Any], *, output_path: str | Path | None = None) -> str:
    """Persist an itinerary to MongoDB when configured; otherwise save locally as JSON."""
    if has_mongo() and itineraries_collection is not None:
        result = itineraries_collection.insert_one(itinerary)
        return str(result.inserted_id)

    path = Path(output_path) if output_path else OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(itinerary, file, indent=2, ensure_ascii=False)
    return str(path)


def get_latest_itinerary() -> dict[str, Any] | None:
    """Return the most recently created itinerary from MongoDB, or from the local JSON fallback."""
    if has_mongo() and itineraries_collection is not None:
        record = itineraries_collection.find_one(sort=[("_id", -1)])
        if record is None:
            return None
        record["_id"] = str(record["_id"])
        return record

    if not OUTPUT_PATH.exists():
        return None

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    with open("itinerary_output.json", "r", encoding="utf-8") as file:
        itinerary = json.load(file)

    itinerary_id = save_itinerary(itinerary)
    print("Saved itinerary:", itinerary_id)