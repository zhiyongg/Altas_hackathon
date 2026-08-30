"""
itinerary_planner.py — Agentic Trip Itinerary Planner
======================================================
Pipeline:
  User Input → Day Classification → Candidate Discovery → Deterministic Filter
  → Geographic Clustering → LLM Selection → Daily Allocation
  → Meal Candidates → Temporal Route Optimization (meals = first-class nodes)
  → Schedule (opening hours + meal windows) → Validation → Worst-Offender Repair
  → Output JSON
"""

import os, json, math, re, logging
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from dataclasses import dataclass, field
from typing import Optional
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
from sklearn.cluster import KMeans

load_dotenv()

# ── Keys ──────────────────────────────────────────────────────────────────────
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")

if not GOOGLE_MAPS_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys in .env file.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-flash-lite"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("planner")

# ── Simple in-run memoization for network/LLM calls ─────────────────────────
_cache: dict = {}

def _cache_key(*args):
    def norm(x):
        if isinstance(x, float): return round(x, 5)
        if isinstance(x, (list, tuple)): return tuple(norm(i) for i in x)
        if isinstance(x, dict): return tuple(sorted((k, norm(v)) for k, v in x.items()))
        return x
    return tuple(norm(a) for a in args)

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_cache_lock = threading.Lock()

def cached(fn):
    def wrapper(*args, **kwargs):
        key = (fn.__name__, _cache_key(*args), _cache_key(*sorted(kwargs.items())))
        with _cache_lock:
            if key in _cache:
                print(f"[CACHE HIT] {fn.__name__}")
                return _cache[key]
        print(f"[API CALL]  {fn.__name__}")
        result = fn(*args, **kwargs)
        with _cache_lock:
            _cache[key] = result
        return result
    return wrapper

# ── Configuration ─────────────────────────────────────────────────────────────
DISCOVERY_RADIUS_M   = 15_000
MIN_PER_DAY          = 3
MAX_PER_DAY          = 7

# Meal scheduling policy

MEAL_TARGETS = {
    "breakfast": dtime(8, 0),
    "lunch": dtime(12, 30),
    "dinner": dtime(18, 30),
}

MEAL_WINDOWS = {
    "breakfast": (dtime(6, 30), dtime(10, 30)),
    "lunch":     (dtime(11, 0), dtime(15, 0)),
    "dinner":    (dtime(18, 30), dtime(21, 30)),
}

# How far we are willing to move from the preferred meal time
# before accepting that the day structure itself needs repair.
MEAL_SOFT_TOLERANCE_MIN = {
    "breakfast": 60,
    "lunch": 90,
    "dinner": 120,
}

# Streamlined list of universal Google Places Table A search types
ATTRACTION_TYPES = [
    # The Catch-Alls (Catches monuments, castles, plazas, observation decks, etc.)
    "tourist_attraction",
    "historical_landmark",
    "cultural_landmark",

    # Arts & Education (Catches history/science/art museums and planetariums)
    "museum",
    "art_gallery",

    # Nature (Catches gardens, preserves, hiking areas, and city parks)
    "park",
    "national_park",

    # Entertainment & Family (Catches water parks, theme parks, animal sanctuaries)
    "amusement_park",
    "zoo",
    "aquarium"
]

TRAVEL_STYLE_CAPACITY = {
    "relaxed":  {"normal_min": 2, "normal_max": 4, "arrival": 1, "departure": 1},
    "moderate": {"normal_min": 3, "normal_max": 6, "arrival": 2, "departure": 2},
    "packed":   {"normal_min": 4, "normal_max": 8, "arrival": 3, "departure": 2},
}

# FOR LOCAL SCORING (Maps Google's returned types to your UI categories)
THEME_TO_TYPES = {
    "culture":       ["museum", "history_museum", "art_museum", "historical_place",
                      "historical_landmark", "cultural_landmark", "castle",
                      "monument", "art_gallery", "shinto_shrine", "buddhist_temple",
                      "church", "mosque"],
    "scenery":       ["national_park", "state_park", "park", "botanical_garden",
                      "nature_preserve", "beach", "lake", "mountain_peak",
                      "wildlife_park", "zoo", "aquarium"],
    "food":          ["restaurant", "cafe", "bakery", "food_court",
                      "fine_dining_restaurant", "family_restaurant", "market"],
    "shopping":      ["shopping_mall", "market", "department_store", "gift_shop",
                      "book_store", "flea_market", "farmers_market"],
    "entertainment": ["amusement_park", "water_park", "movie_theater", "bowling_alley",
                      "karaoke", "concert_hall", "live_music_venue", "comedy_club",
                      "arcade", "night_club", "bar", "cocktail_bar", "pub", "wine_bar"],
    "adventure":     ["hiking_area", "adventure_sports_center", "cycling_park",
                      "ski_resort", "golf_course", "sports_activity_location"],
    "wellness":      ["spa", "massage", "sauna", "wellness_center", "yoga_studio"],
    "city":          ["tourist_attraction", "observation_deck", "plaza", "landmark",
                      "point_of_interest"]
}

# All available preference categories
AVAILABLE_PREFERENCES = list(THEME_TO_TYPES.keys())

SCORE_W = {"pref": 0.45, "rating": 0.20, "popularity": 0.15, "convenience": 0.10, "budget": 0.10}

BUDGET_PRICE_LEVELS = {
    "low":    {"PRICE_LEVEL_FREE", "PRICE_LEVEL_INEXPENSIVE"},
    "medium": {"PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"},
    "high":   {"PRICE_LEVEL_MODERATE", "PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE"},
}

def calculate_preference_scores(preferences: list, selected_preferences: list) -> dict:
    total_preferences = len(preferences)
    selected_count = len(selected_preferences)
    scores = {}
    for preference in preferences:
        if preference in selected_preferences:
            score = 0.50 + 0.50 * (1 - selected_count / total_preferences)
        else:
            score = 0.50 - 0.20 * (selected_count / total_preferences)
        score = max(0.0, min(1.0, score))
        scores[preference] = round(score, 4)
    return scores

def compute_max_candidates(n_days: int, n_preferences: int) -> int:
    # Base: enough for target selection (~4/day) + 2x buffer for backups
    per_day_target = 4
    base = n_days * per_day_target * 3  # 3x = selection pool + repair backups
    # Wider preference sets need a slightly larger pool for diversity
    pref_bonus = n_preferences * 5
    return max(20, min(base + pref_bonus, 120))  # sane floor/ceiling

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_dur(s) -> int:
    if s is None: return 0
    if isinstance(s, (int, float)): return int(s)
    return int(re.sub(r"[^\d]", "", str(s)) or 0)

def parse_dt(s: str) -> datetime:
    if not s:
        # Fallback to prevent crash if the LLM passed None/empty
        return datetime.now()
        
    # Fix common LLM string hallucinations (e.g., replacing spaces with 'T')
    clean_s = str(s).strip().replace(" ", "T")
    
    # Remove trailing 'Z' if the LLM appended a UTC marker
    if clean_s.endswith("Z"):
        clean_s = clean_s[:-1]
        
    try:
        return datetime.fromisoformat(clean_s)
    except ValueError:
        # Fallback if the LLM only provided 'YYYY-MM-DD' and dropped the time
        match = re.search(r'\d{4}-\d{2}-\d{2}', clean_s)
        if match:
            return datetime.strptime(match.group(), "%Y-%m-%d")
            
        # Absolute fallback to prevent a hard pipeline crash
        return datetime.now()

def fmt_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def haversine_km(a: dict, b: dict) -> float:
    lat1, lon1 = math.radians(a["latitude"]), math.radians(a["longitude"])
    lat2, lon2 = math.radians(b["latitude"]), math.radians(b["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    c = 2 * math.asin(math.sqrt(
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2))
    return 6371 * c


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE API WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def _places_headers():
    fields = [
        "places.id", "places.displayName", "places.location",
        "places.formattedAddress",
        "places.rating",
        "places.regularOpeningHours.weekdayDescriptions",
        "places.priceLevel", "places.primaryType", "places.types",
        "places.userRatingCount",
    ]
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(fields),
    }

def _safe_post(url: str, headers: dict, json_payload: dict, max_retries: int = 3):
    import time
    for attempt in range(max_retries):
        try:
            return requests.post(url, headers=headers, json=json_payload, timeout=15)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                log.error("API call failed after %d retries: %s", max_retries, e)
                return None
            time.sleep(1 + attempt)
    return None

@cached
def search_nearby(lat: float, lng: float, radius: float = DISCOVERY_RADIUS_M,
                  included_types: list[str] | None = None, max_results: int = 20) -> list[dict]:
    url = "https://places.googleapis.com/v1/places:searchNearby"
    payload = {
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": radius}
        },
    }
    if included_types: payload["includedTypes"] = included_types
    resp = _safe_post(url, headers=_places_headers(), json_payload=payload)
    if not resp or resp.status_code != 200:
        log.warning("Nearby Search %s: %s", resp.status_code if resp else "Failed", resp.text if resp else "No Response")
        return []
    return resp.json().get("places", [])

@cached
def search_text(query: str, max_results: int = 10) -> list[dict]:
    url = "https://places.googleapis.com/v1/places:searchText"
    payload = {"textQuery": query, "pageSize": max_results}
    resp = _safe_post(url, headers=_places_headers(), json_payload=payload)
    if not resp or resp.status_code != 200:
        log.warning("Text Search %s: %s", resp.status_code if resp else "Failed", resp.text if resp else "No Response")
        return []
    return resp.json().get("places", [])

# ── Pairwise EDGE cache for route-matrix elements ────────────────────────────
# The generic @cached decorator keys on the exact origin/destination arrays, so
# it misses every time the repair loop shrinks/reorders the waypoint list --
# even though every A→B edge of the smaller matrix was already paid for inside
# the bigger one. Caching individual edges lets any subset matrix be
# reconstructed locally without touching the network.
_EDGE_CACHE: dict = {}

def compute_route_matrix(origins: list[dict], destinations: list[dict],
                         travel_mode="DRIVE", routing_pref="TRAFFIC_UNAWARE") -> list[dict]:
    def _ll(loc): return (round(loc["latitude"], 5), round(loc["longitude"], 5))

    # 1. SMART EDGE CACHE: every requested edge already known -> rebuild locally
    fully_cached = True
    cached_results = []
    for oi, o in enumerate(origins):
        for di, d in enumerate(destinations):
            key = (travel_mode, routing_pref, _ll(o), _ll(d))
            if key in _EDGE_CACHE:
                cached_results.append({
                    "originIndex": oi,
                    "destinationIndex": di,
                    "duration": f"{_EDGE_CACHE[key]}s",
                })
            else:
                fully_cached = False
                break
        if not fully_cached:
            break

    if fully_cached:
        print("[CACHE HIT] compute_route_matrix (subset reconstructed, 0 elements)")
        return cached_results

    # 2. At least one edge is new -> Fetch the matrix, but safely CHUNK the payload
    # to guarantee we never exceed Google's 100-element limit per API call.
    print("[API CALL]  compute_route_matrix (dynamically chunked)")
    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    mask = ["originIndex", "destinationIndex", "duration"]
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(mask),
    }

    def _clean_ll(loc): return {"latitude": loc["latitude"], "longitude": loc["longitude"]}

    combined_results = []

    # Determine the maximum safe chunk size based on destinations
    # e.g., 100 limit // 13 destinations = 7 origins max per API call
    max_elements = 100
    dest_len = max(1, len(destinations))
    chunk_size = max(1, max_elements // dest_len)

    for i in range(0, len(origins), chunk_size):
        orig_chunk = origins[i:i + chunk_size]

        payload = {
            "origins":  [{"waypoint": {"location": {"latLng": _clean_ll(o)}}} for o in orig_chunk],
            "destinations": [{"waypoint": {"location": {"latLng": _clean_ll(d)}}} for d in destinations],
            "travelMode": travel_mode,
        }

        if travel_mode == "TRANSIT":
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT09:00:00Z")
            payload["departureTime"] = tomorrow
        elif travel_mode == "DRIVE":
            payload["routingPreference"] = routing_pref

        resp = _safe_post(url, headers=headers, json_payload=payload)
        if not resp or resp.status_code != 200:
            log.warning("Route Matrix 400: %s", resp.text if resp else "Failed")
            continue

        data = resp.json()

        # 3. Store EVERY returned edge, and adjust the originIndex so the
        # local chunks seamlessly map back to the global routing array.
        for e in data:
            local_oi = e.get("originIndex")
            di = e.get("destinationIndex")

            if local_oi is not None and di is not None:
                global_oi = i + local_oi  # Remap relative chunk index back to absolute

                # Update the JSON payload with the true global origin
                e["originIndex"] = global_oi
                combined_results.append(e)

                # Save to cache
                key = (travel_mode, routing_pref, _ll(origins[global_oi]), _ll(destinations[di]))
                _EDGE_CACHE[key] = parse_dur(e.get("duration"))

    return combined_results


# ══════════════════════════════════════════════════════════════════════════════
# LLM WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

@cached
def call_llm(system_prompt: str, user_prompt: str) -> dict:
    combined = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=combined,
            config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
        )
        text = response.text
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TripConfig:
    destination: str
    start_date: str
    end_date: str
    arrival_datetime: str
    departure_datetime: str
    hotel: dict
    airport: dict
    travel_style: str
    transport_mode: str
    group_size: int
    budget: str
    check_in_time: str
    check_out_time: str
    selected_preferences: list = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
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
    base_location: dict
    # Temporal bounds are REQUIRED at construction: a DayPlan must never
    # exist without a concrete start/end window, because temporal scheduling,
    # meal-gap detection and validation all depend on them.
    start_time: datetime = None
    end_time: datetime = None
    attractions: list = field(default_factory=list)
    meals: dict = field(default_factory=dict)
    sequence: list = field(default_factory=list)
    schedule: list = field(default_factory=list)
    capacity_min: int = 0
    capacity_max: int = 0
    valid: bool = True
    violations: list = field(default_factory=list)
    # Meals the repair stage has deliberately dropped for this day so they are
    # not re-inserted on the next rebuild.
    dropped_meals: set = field(default_factory=set)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DAY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_days(cfg: TripConfig) -> list[DayPlan]:
    arrival_dt = parse_dt(cfg.arrival_datetime)
    departure_dt = parse_dt(cfg.departure_datetime)
    start_date = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(cfg.end_date, "%Y-%m-%d").date()

    style_caps = TRAVEL_STYLE_CAPACITY.get(cfg.travel_style, TRAVEL_STYLE_CAPACITY["moderate"])

    days: list[DayPlan] = []
    cur = start_date

    while cur <= end_date:
        dtype = "arrival" if cur == start_date else ("departure" if cur == end_date else "normal")
        base = datetime.strptime(cur.isoformat(), "%Y-%m-%d")

        # 1. Compute the exact temporal window FIRST, then hand it to the
        #    constructor so the DayPlan is never created without its bounds.
        if dtype == "arrival":
            # If the flight lands after standard check-in time (15:00 by
            # default), the guest can check in almost right away — give them
            # 1 hour to clear immigration/baggage and get to the hotel, then
            # start the check-in beat there, instead of the flat 2-hour
            # touring buffer used below. If they land before check-in opens,
            # keep the 2-hour buffer for bags/immigration/transit and let
            # the mid-day check-in insertion further down handle the actual
            # 15:00 check-in once it's reached.
            cin_time = datetime.strptime(cfg.check_in_time, "%H:%M").time()
            arrival_check_in_dt = arrival_dt.replace(hour=cin_time.hour, minute=cin_time.minute, second=0)
            if arrival_dt >= arrival_check_in_dt:
                start_time = arrival_dt + timedelta(hours=1)
            else:
                start_time = arrival_dt + timedelta(hours=2)
            end_time = arrival_dt.replace(hour=21, minute=0, second=0)
        elif dtype == "departure":
            cout_time = datetime.strptime(cfg.check_out_time, "%H:%M").time()
            default_start = base.replace(hour=8, minute=0)
            checkout_dt = base.replace(hour=cout_time.hour, minute=cout_time.minute)

            # Start touring at 8 AM, OR the check-out time (whichever is earlier)
            start_time = min(default_start, checkout_dt)
            end_time = departure_dt - timedelta(hours=3)
            
        else:
            start_time = base.replace(hour=8, minute=0)
            end_time = base.replace(hour=21, minute=0)

        # Guard against a degenerate/inverted window (e.g. very early flight).
        if end_time <= start_time:
            end_time = start_time

        d = DayPlan(
            day_index=len(days),
            date=cur.isoformat(),
            day_type=dtype,
            base_location=dict(cfg.hotel),
            start_time=start_time,
            end_time=end_time,
        )

        # 2. Dynamic Capacity Math (Replaces the LLM)
        if dtype == "normal":
            d.capacity_min = style_caps["normal_min"]
            d.capacity_max = style_caps["normal_max"]
        else:
            # Calculate raw hours available
            total_hours = (d.end_time - d.start_time).total_seconds() / 3600.0

            # Subtract time spent eating (approx 1.5 hours per scheduled meal)
            meal_count = len(get_meal_slots(d))
            active_touring_hours = max(0.0, total_hours - (meal_count * 1.5))

            # Determine how many hours a single attraction takes based on travel style
            if cfg.travel_style == "relaxed":
                hrs_per_stop = 3.0
            elif cfg.travel_style == "packed":
                hrs_per_stop = 1.5
            else: # moderate
                hrs_per_stop = 2.0

            # Calculate max places mathematically
            calculated_max = int(active_touring_hours / hrs_per_stop)

            # ── FIX: Respect the dictionary limits for arrival/departure ──
            # Fetch the strict cap from your TRAVEL_STYLE_CAPACITY (e.g., 2)
            dict_limit = style_caps.get(dtype, 2)

            # Force the engine to take the lowest number (Math vs Dictionary)
            d.capacity_max = min(dict_limit, max(0, calculated_max))

            # Require at least 1 attraction if there is time, otherwise 0
            d.capacity_min = min(1, d.capacity_max)

        days.append(d)
        cur += timedelta(days=1)

    return days


def get_meal_slots(dp: DayPlan) -> list[tuple]:
    slots = []
    base = datetime.strptime(dp.date, "%Y-%m-%d")

    start_t = dp.start_time.time()
    end_t = dp.end_time.time()

    b_time = base.replace(hour=8, minute=0)
    l_time = base.replace(hour=12, minute=30)
    d_time = base.replace(hour=18, minute=30)

    # 1. Breakfast: Included if the day starts by 9:00 AM
    if start_t <= dtime(9, 0):
        slots.append(("breakfast", b_time, "hotel"))

    # 2. Lunch: Included if active through midday
    if start_t <= dtime(12, 30) and end_t >= dtime(13, 30):
        slots.append(("lunch", l_time, "attraction"))

    # 3. Dinner: Force included on normal days or if day extends past 19:30
    # ── FIX: Changed from 18:00 to 19:30 to prevent impossible overlapping on departure days ──
    if end_t >= dtime(19, 30) or dp.day_type == "normal":
        context = "hotel" if dp.day_type == "arrival" else "attraction"
        slots.append(("dinner", d_time, context))

    return slots
    

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CANDIDATE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def build_discovery_pool(cfg: TripConfig, max_candidates: int) -> dict:
    """Translate the final candidate target into discovery-phase search sizing."""
    n_prefs = max(len(cfg.selected_preferences), 1)
    return {
        # aim wider than max_candidates since filter_candidates trims down after
        "discovery_target": min(max_candidates * 2, 100),
        "n_theme_searches": min(n_prefs, 5),
        "text_results_per_theme": 10,
        "nearby_results_per_chunk": 20,
    }

def _get_dynamic_duration(place_types: list[str], primary: str, travel_style: str) -> int:
    long_visits = {
        "amusement_park", "theme_park", "national_park", "zoo",
        "aquarium", "ski_resort", "golf_course", "hiking_area", "water_park"
    }
    medium_visits = {
        "museum", "history_museum", "art_museum", "park", "shopping_mall",
        "botanical_garden", "art_gallery", "castle", "historical_place",
        "performing_arts_theater", "department_store", "spa", "market"
    }

    base = 60
    for t in [primary] + place_types:
        if t in long_visits:
            base = 180
            break
        if t in medium_visits:
            base = 120
            break

    # Apply pacing multiplier
    if travel_style == "packed":
        return int(base * 0.75)  # e.g., 120 mins -> 90 mins
    elif travel_style == "relaxed":
        return int(base * 1.25)  # e.g., 120 mins -> 150 mins

    return base

def _raw_to_place(raw: dict, source="api", travel_style: str = "packed") -> Place | None:
    loc = raw.get("location", {})
    if not loc.get("latitude"): return None

    types_list = raw.get("types", [])
    primary = raw.get("primaryType", "")

    # Keep meals at a strict 60 min minimum so we don't trigger validation errors,
    # but allow attractions to scale freely based on travel style.
    if source == "meal":
        dur = 75 if travel_style == "relaxed" else 60
    else:
        dur = _get_dynamic_duration(types_list, primary, travel_style)

    return Place(
        id=raw.get("id", ""), name=raw.get("displayName", {}).get("text", "Unknown"),
        location={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "name": raw.get("displayName", {}).get("text", "Unknown"),
            "address": raw.get("formattedAddress", "")
        },
        types=types_list, primary_type=primary,
        rating=raw.get("rating", 0.0), user_rating_count=raw.get("userRatingCount", 0),
        price_level=raw.get("priceLevel", ""),
        opening_hours=raw.get("regularOpeningHours", {}).get("weekdayDescriptions", []),
        visit_duration_min=dur,
        source=source,
    )

def discover_candidates(cfg: TripConfig, pool: dict) -> list[Place]:
    seen: set[str] = set()
    places: list[Place] = []
    hlat, hlng = cfg.hotel["latitude"], cfg.hotel["longitude"]
    banned_primary_types = ["restaurant", "cafe", "bar", "lodging", "hotel"]
    discovery_target = pool["discovery_target"]

    def add_results(raws):
        for raw in raws:
            p = _raw_to_place(raw, "text", cfg.travel_style)
            if p and p.id not in seen and p.primary_type not in banned_primary_types:
                seen.add(p.id)
                places.append(p)

    with ThreadPoolExecutor(max_workers=8) as ex:
        # 1. Theme searches + vibe searches, all fired at once
        futures = []
        themes = sorted(cfg.preferences.keys(), key=lambda t: cfg.preferences[t], reverse=True)
        for theme in themes[:pool["n_theme_searches"]]:
            futures.append(ex.submit(search_text, f"{theme} attractions in {cfg.destination}",
                                      pool["text_results_per_theme"]))

        vibe_queries = expand_vibe_to_queries(cfg.destination, cfg.custom_vibe) if cfg.custom_vibe else []
        for q in vibe_queries:
            futures.append(ex.submit(search_text, q, 5))

        for fut in as_completed(futures):
            add_results(fut.result())

        # 2. Nearby search passes — each pass's chunks run in parallel;
        #    passes stay sequential since later passes only run if still short.
        radius = DISCOVERY_RADIUS_M
        max_passes = 3 if discovery_target > 80 else (2 if discovery_target > 40 else 1)
        for pass_i in range(max_passes):
            if len(places) >= discovery_target:
                break
            chunk_futures = [
                ex.submit(search_nearby, hlat, hlng, radius, ATTRACTION_TYPES[i:i + 5],
                          pool["nearby_results_per_chunk"])
                for i in range(0, len(ATTRACTION_TYPES), 5)
            ]
            for fut in as_completed(chunk_futures):
                add_results(fut.result())
            radius = int(radius * 1.5)

    log.info("Discovered %d unique candidates (target %d, %d pass(es)).",
              len(places), discovery_target, max_passes)
    return places

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DETERMINISTIC FILTER + SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _budget_score(p: Place, budget: str) -> float:
    # ── FIX: Safely convert to string in case the LLM passed an integer ──
    safe_budget = str(budget).lower() if budget else ""
    allowed = BUDGET_PRICE_LEVELS.get(safe_budget, set())
    
    if not p.price_level:
        # unknown price → neutral, don't punish
        return 0.5
    return 1.0 if p.price_level in allowed else 0.2

def _score_place(p: Place, prefs: dict, hotel_loc: dict, budget: str) -> float:
    pref_s = 0.0
    for theme, w in prefs.items():
        if any(t in p.types for t in THEME_TO_TYPES.get(theme, [])):
            pref_s = max(pref_s, w)
    rating_s = p.rating / 5.0 if p.rating else 0.0
    pop_s    = min(math.log(p.user_rating_count + 1) / 10.0, 1.0)
    dist     = haversine_km(p.location, hotel_loc)
    conv_s   = max(1.0 - dist / (DISCOVERY_RADIUS_M / 1000), 0.0)

    # Apply new budget score
    budget_s = _budget_score(p, budget)

    return (SCORE_W["pref"] * pref_s + SCORE_W["rating"] * rating_s +
            SCORE_W["popularity"] * pop_s + SCORE_W["convenience"] * conv_s +
            SCORE_W["budget"] * budget_s)

def _normalize_name(name: str) -> set[str]:
    clean = re.sub(r"[^\w\s]", " ", name.lower())
    return set(clean.split())


def _name_similarity(name1: str, name2: str) -> float:
    generic = {
        "the", "of", "and", "tokyo", "japan",
        "park", "museum", "shrine", "zoo",
        "temple", "garden", "national", "city",
        "tower", "station", "building"
    }

    w1 = _normalize_name(name1) - generic
    w2 = _normalize_name(name2) - generic

    if not w1 or not w2:
        return 0.0

    intersection = len(w1 & w2)
    union = len(w1 | w2)

    return intersection / union


def _is_duplicate_place(p1: Place, p2: Place) -> bool:
    # 1. Exact Google Place ID match
    if p1.id and p2.id and p1.id == p2.id:
        return True

    # 2. Name similarity
    similarity = _name_similarity(p1.name, p2.name)

    # Names aren't enough by themselves.
    if similarity < 0.8:
        return False

    # 3. Geographic confirmation
    distance_km = haversine_km(p1.location, p2.location)

    # Same/similar name AND extremely close together
    if distance_km <= 0.3:
        return True

    return False

def filter_candidates(places: list[Place], cfg: TripConfig, max_candidates: int) -> list[Place]:
    # 1. Score all places
    for p in places:
        p.score = _score_place(p, cfg.preferences, cfg.hotel, cfg.budget)

    # 2. Sort highest to lowest
    places.sort(key=lambda x: x.score, reverse=True)

    # ── FIX: Deduplicate Similar Names ──
    unique_places = []
    for p in places:
        is_dup = False
        for u in unique_places:
            if _is_duplicate_place(p, u):
                # We already have a higher-scored place with a very similar name!
                log.info(f"Deduplication: Dropped '{p.name}' because it is too similar to '{u.name}'")
                is_dup = True
                break

        if not is_dup:
            unique_places.append(p)
    # ────────────────────────────────────

    # 3. Cut to max candidates
    out = unique_places[:max_candidates]
    log.info("Filtered → %d candidates (min score %.2f).", len(out), out[-1].score if out else 0)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — GEOGRAPHIC CLUSTERING (K-MEANS)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_places(places: list[Place], n_clusters: int) -> list[list[Place]]:
    if not places: return []

    # If we have fewer places than target clusters, just put each place in its own cluster
    if len(places) <= n_clusters:
        return [[p] for p in places]

    # Extract coordinates for the ML model
    coords = np.array([[p.location["latitude"], p.location["longitude"]] for p in places])

    # Suppress Windows thread leak warning for KMeans
    os.environ["OMP_NUM_THREADS"] = "1"

    # Run K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)

    # Group the places into their new mathematically optimized zones
    clusters = [[] for _ in range(n_clusters)]
    for place, label in zip(places, labels):
        clusters[label].append(place)

    # Sort zones by density (largest cluster first)
    return sorted(clusters, key=len, reverse=True)

def build_selected_clusters(selected: list[Place], clusters: list[list[Place]]) -> list[list[Place]]:
    """Filters the geo-clusters down to only the LLM-selected places."""
    sel_ids = {p.id for p in selected}
    sel_clusters = []
    for cl in clusters:
        m = [p for p in cl if p.id in sel_ids]
        if m: sel_clusters.append(m)
    return sel_clusters

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — LLM SEMANTIC SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_places_llm(candidates: list[Place], cfg: TripConfig, n_days: int) -> list[Place]:
    # Count how many normal vs partial days we have
    normal_days = sum(1 for d in classify_days(cfg) if d.day_type == "normal")
    arrival_departure_days = n_days - normal_days

    # DYNAMIC TARGET: e.g., 4 per normal day, 1-2 for arrival/departure
    target = (normal_days * 4) + (arrival_departure_days * 1)
    target = max(MIN_PER_DAY, target) # Ensure we always pick at least a minimum

    if len(candidates) <= target:
        return candidates

    summaries = [
        {"i": i, "name": p.name, "types": p.types[:4], "rating": p.rating, "popularity": p.user_rating_count}
        for i, p in enumerate(candidates)
    ]
    sys_p = '''You are a travel planner AI. Select attractions matching user preferences and also matched 
            with general tourist places with high popularity . JSON only: {"selected_indices": [0, 2, 5, ...]}
            '''
    usr_p = (
        f"Trip: {n_days} days ({normal_days} normal touring days)\n"
        f"Selected Preferences: {cfg.selected_preferences}\n"
        f"Preference Scores: {json.dumps(cfg.preferences, indent=2)}\n"
        f"Candidates:\n{json.dumps(summaries, indent=2)}\n\n"
        f"Select the best {target} indices for this duration. JSON only.")

    result = call_llm(sys_p, usr_p)
    idxs = result.get("selected_indices", [])

    if not idxs:
        log.warning("LLM returned empty — fallback to top-scored.")
        return candidates[:target]

    sel = [candidates[i] for i in idxs if 0 <= i < len(candidates)]

    # Padding safety rail if LLM picks too few
    min_required = (normal_days * MIN_PER_DAY)
    if len(sel) < min_required:
        log.info(f"LLM only picked {len(sel)} attractions. Padding to {min_required}...")
        for c in candidates:
            if c not in sel:
                sel.append(c)
            if len(sel) >= min_required:
                break

    return sel[:target]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — DAILY ALLOCATION + MEAL PLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def allocate_to_days(selected: list[Place], days: list[DayPlan], sel_clusters: list[list[Place]],
                      cfg: TripConfig, filtered_candidates: list[Place], cluster_order: list[int] | None = None) -> None:
    normal  = [d for d in days if d.day_type == "normal"]
    partial = [d for d in days if d.day_type in ("arrival", "departure")]

    used_ids = set()

    if cluster_order and sorted(cluster_order) == list(range(len(sel_clusters))):
        ordered_clusters = [sel_clusters[i] for i in cluster_order]
    else:
        ordered_clusters = sorted(sel_clusters, key=len, reverse=True)

    # 1. Primary Geo-Cluster Allocation
    for i, day in enumerate(normal):
        cap_min, cap_max = day.capacity_min, day.capacity_max
        if i < len(ordered_clusters):
            day.attractions = [p for p in ordered_clusters[i] if p.id not in used_ids][:cap_max]

        if len(day.attractions) < cap_min:
            for p in selected:
                if p.id not in used_ids and p not in day.attractions:
                    day.attractions.append(p)
                if len(day.attractions) >= cap_min:
                    break
        used_ids.update({p.id for p in day.attractions})

    # 2. Partial Days Allocation
    for day in partial:
        n = day.capacity_max
        day.attractions = [p for p in selected if p.id not in used_ids][:n]
        used_ids.update({p.id for p in day.attractions})

        # ── FIX: Top up empty departure days with high-scored backups ──
        if len(day.attractions) < n:
            for p in filtered_candidates:
                if p.id not in used_ids:
                    day.attractions.append(p)
                    used_ids.add(p.id)
                if len(day.attractions) >= n:
                    break

def _combine_date_time(date_str: str, t: dtime) -> datetime:
    base = datetime.strptime(date_str, "%Y-%m-%d")
    return base.replace(
        hour=t.hour,
        minute=t.minute,
        second=0,
        microsecond=0,
    )


def _meal_target_datetime(day: DayPlan, meal_name: str) -> datetime:
    return _combine_date_time(
        day.date,
        MEAL_TARGETS[meal_name],
    )


def _meal_window_datetimes(day: DayPlan, meal_name: str) -> tuple[datetime, datetime]:
    lo, hi = MEAL_WINDOWS[meal_name]
    return (
        _combine_date_time(day.date, lo),
        _combine_date_time(day.date, hi),
    )

# ── STAGE 6/7: meals as FIRST-CLASS route nodes + temporal optimization ──────

NIGHTLIFE_TYPES = {"bar", "night_club", "pub", "wine_bar", "cocktail_bar", "karaoke"}

# Virtual evening window so nightlife venues are deferred to the end of the day.
NIGHTLIFE_VIRTUAL_OPEN = dtime(18, 0)
NIGHTLIFE_VIRTUAL_CLOSE = dtime(23, 59)

MEAL_LABELS = {"breakfast": "Breakfast", "lunch": "Lunch", "dinner": "Dinner"}


class DayRouteMatrix:
    """ONE pairwise route-matrix API call over every day waypoint, then O(1)
    travel-time lookups afterwards.

    The old code issued a separate API request per leg (the removed
    update_sequence_travel_times did N-1 calls). Fetching the full
    origin x destination matrix once keeps API usage controlled even though we
    now re-run route optimization after inserting meals.
    """

    def __init__(self, cfg: TripConfig, waypoints: list[dict]):
        self.waypoints = waypoints
        self.n = len(waypoints)
        self.spd = _SPEED.get(cfg.transport_mode, 30)
        self.tt = [[0.0] * self.n for _ in range(self.n)]
        if self.n > 1:
            locs = [w["location"] for w in waypoints]
            for e in compute_route_matrix(locs, locs, travel_mode=cfg.transport_mode):
                oi = e.get("originIndex", 0)
                di = e.get("destinationIndex", 0)
                self.tt[oi][di] = parse_dur(e.get("duration"))

    def travel(self, i: int, j: int) -> int:
        d = self.tt[i][j]
        if not d:
            d = int(haversine_km(self.waypoints[i]["location"],
                                 self.waypoints[j]["location"]) / self.spd * 3600)
        return int(d)


def _simulate_attraction_timeline(day: DayPlan, cfg: TripConfig) -> list[dict]:
    """Cheap haversine forward simulation (no API calls). Used only to locate
    feasible lunch/dinner gaps BEFORE committing to real restaurants."""
    timeline = []
    cur = day.start_time
    prev = dict(cfg.hotel)
    spd = _SPEED.get(cfg.transport_mode, 30)
    for p in day.attractions:
        travel_s = int(haversine_km(prev, p.location) / spd * 3600)
        arrival = cur + timedelta(seconds=travel_s)
        win = get_opening_window(p, day.date)
        if win:
            open_dt = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
            if arrival < open_dt:
                arrival = open_dt
        depart = arrival + timedelta(minutes=p.visit_duration_min)
        timeline.append({"place": p, "arrival": arrival, "depart": depart, "location": p.location})
        cur, prev = depart, p.location
    return timeline


def _meal_gap_location(day: DayPlan, cfg: TripConfig, timeline: list[dict], meal_name: str) -> dict:
    """Pick the geographic anchor for a meal: breakfast near the hotel,
    lunch/dinner near the attraction being visited around the target time."""
    if meal_name == "breakfast" or not timeline:
        return dict(cfg.hotel)
    target = _meal_target_datetime(day, meal_name)
    best, best_gap = None, float("inf")
    for t in timeline:
        if t["arrival"] <= target <= t["depart"]:
            gap = 0.0
        elif target < t["arrival"]:
            gap = (t["arrival"] - target).total_seconds()
        else:
            gap = (target - t["depart"]).total_seconds()
        if gap < best_gap:
            best_gap, best = gap, t
    return dict(best["location"]) if best else dict(cfg.hotel)


def _is_valid_restaurant(r: Optional[Place], used_restaurants: set) -> bool:
    if not r:
        return False

    # 1. Ban non-restaurant place types
    banned_types = {
        "lodging", "bar", "night_club", "movie_theater",
        "convenience_store", "supermarket", "grocery_or_supermarket",
        "shopping_mall", "department_store", "gas_station", "gym", "hospital", "clinic"
    }

    # 2. Ban names that indicate it's not a real sit-down restaurant
    banned_words = {
        # Lodging
        "hotel", "resort", "hostel", "ryokan", "inn", "guest house",

        # Entertainment & General Services
        "cinema", "theater", "theatre", "stadium", "clinic", "hospital", "bank", "atm", "supermarket"
    }

    name = r.name.lower()

    # Check Types
    if any(t in r.types for t in banned_types):
        return False

    # Check Words
    if any(w in name for w in banned_words):
        return False

    # Check Exact Duplicates
    if name in used_restaurants:
        return False

    # 3. Avoid repeating obvious chains (e.g., catching Ichiran twice)
    words = name.split()
    if words:
        brand = words[0]
        generic_words = {"cafe", "restaurant", "the", "bistro", "brasserie", "pizzeria", "la", "le"}
        if len(brand) > 3 and brand not in generic_words:
            for used in used_restaurants:
                if used.startswith(brand):
                    return False # Brand already eaten this trip!

    return True


def find_meal_restaurant(meal_name: str, location: dict, cfg: TripConfig,
                         used_restaurants: set, date_str: str = "") -> Optional[Place]:
    if meal_name == "breakfast":
        meal_types = ["cafe", "bakery"]
    elif meal_name == "lunch":
        meal_types = ["restaurant", "cafe"]
    else:
        meal_types = ["restaurant", "steak_house"]
        
    raw = search_nearby(location["latitude"], location["longitude"], radius=1500,
                        included_types=meal_types, max_results=10)
    
    valid_candidates = []
    for x in raw:
        r = _raw_to_place(x, "meal", cfg.travel_style)
        if not _is_valid_restaurant(r, used_restaurants):
            continue
            
        if date_str:
            win = get_opening_window(r, date_str)
            if win == "CLOSED":
                continue
            if win:
                open_hour = int(win[0].split(":")[0])
                close_hour = int(win[1].split(":")[0])
                close_min = int(win[1].split(":")[1])
                
                # Prevent Breakfast Delay Bug
                if meal_name == "breakfast" and open_hour >= 10:
                    continue
                    
                # ── FIX: Prevent Closed-on-Arrival Bug ──
                lo_time = MEAL_WINDOWS[meal_name][0]
                close_t = dtime(close_hour, close_min)
                open_t = dtime(open_hour, int(win[0].split(":")[1]))
                
                # If it closes on the same day, reject it if it closes before/at window start
                if close_t > open_t and close_t <= lo_time:
                    continue
                    
        valid_candidates.append(r)

    if not valid_candidates:
        return None
    return max(valid_candidates, key=lambda r: (r.rating or 0, r.user_rating_count or 0))


def _entry_window(day: DayPlan, wp: dict):
    if wp["kind"] == "meal":
        return _meal_window_datetimes(day, wp.get("meal_name"))
    
    # Allow hotel check-in from check-in time onward
    if wp["kind"] == "hotel_checkin":
        cin_time = dtime(15, 0)
        return (_combine_date_time(day.date, cin_time), _combine_date_time(day.date, dtime(22, 0)))

    place = wp.get("place")
    if not place:
        return None
    if wp["kind"] == "attraction" and any(t in NIGHTLIFE_TYPES for t in place.types):
        return (_combine_date_time(day.date, NIGHTLIFE_VIRTUAL_OPEN),
                _combine_date_time(day.date, NIGHTLIFE_VIRTUAL_CLOSE))
    win = get_opening_window(place, day.date)
    if not win:
        return None
    open_dt = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
    close_dt = datetime.strptime(day.date + " " + win[1], "%Y-%m-%d %H:%M")
    if close_dt <= open_dt:
        close_dt += timedelta(days=1)
    return (open_dt, close_dt)


def _optimize_route_temporal(waypoints: list[dict], matrix: DayRouteMatrix, day: DayPlan) -> list[int]:
    n = len(waypoints)
    if n <= 2:
        return list(range(n))
    unvisited = set(range(1, n - 1))
    order = [0]
    cur = 0
    cur_time = day.start_time

    while unvisited:
        best_i, best_cost, best_entry = None, float("inf"), None
        for j in unvisited:
            wp = waypoints[j]
            
            # 1. STRICT CHRONOLOGICAL MEAL PRECEDENCE
            if wp.get("kind") == "meal":
                meal_name = wp.get("meal_name")
                if meal_name == "lunch":
                    bf_idx = next((i for i, w in enumerate(waypoints) if w.get("meal_name") == "breakfast"), None)
                    if bf_idx is not None and bf_idx in unvisited:
                        continue
                if meal_name == "dinner":
                    prior_meals = [i for i, w in enumerate(waypoints) if w.get("meal_name") in ("breakfast", "lunch")]
                    if any(m_idx in unvisited for m_idx in prior_meals):
                        continue

            arrival = cur_time + timedelta(seconds=matrix.travel(cur, j))
            entry = arrival
            penalty = 0.0
            
            # 2. PENALIZE BACK-TO-BACK MEALS
            prev_wp = waypoints[cur]
            has_unvisited_attractions = any(waypoints[u].get("kind") == "attraction" for u in unvisited)
            if prev_wp.get("kind") == "meal" and wp.get("kind") == "meal" and has_unvisited_attractions:
                penalty += 1e5 

            window = _entry_window(day, wp)
            if window:
                lo, hi = window
                if entry < lo:
                    entry = lo
                if entry > hi:
                    penalty += 1e7  # Missed window
                
                # 3. MEAL GRAVITY: Force optimizer to eat if naturally inside the window
                # ── FIX: Use 'arrival' instead of 'entry' so it doesn't pull meals from the future ──
                if wp.get("kind") == "meal" and lo <= arrival <= hi:
                    time_left = (hi - arrival).total_seconds()
                    if time_left < 7200: # Less than 2 hours left -> Extreme Priority
                        penalty -= 1e6 
                    else: # Inside window -> High Priority
                        penalty -= 50000

            cost = (entry - cur_time).total_seconds() + penalty
            if cost < best_cost:
                best_cost, best_i, best_entry = cost, j, entry

        if best_i is None:
            break

        order.append(best_i)
        unvisited.discard(best_i)
        cur = best_i
        place = waypoints[best_i].get("place")
        dur = wp.get("duration_min") if wp.get("duration_min") else (place.visit_duration_min if place else 60)
        cur_time = best_entry + timedelta(minutes=dur)

    order.append(n - 1)
    return order


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — ROUTE OPTIMIZATION WITH MEALS AS FIRST-CLASS NODES
# ══════════════════════════════════════════════════════════════════════════════

def build_day_sequence(day: DayPlan, cfg: TripConfig, used_restaurants: set) -> None:
    if len(day.attractions) > 8:
        log.warning("Day %d has %d attractions. Truncating to 8.", day.day_index, len(day.attractions))
        day.attractions = day.attractions[:8]

    # ── REPLACE FROM HERE ──
    if day.day_type == "departure":
        end_name = f"Pick Up Bags & Depart for Airport"
    else:
        end_name = f"Return to Hotel ({cfg.hotel.get('name', '')})"

    # Free any restaurants reserved on a previous pass for this day.
    for m in day.meals.values():
        used_restaurants.discard(m.name.lower())
    day.meals.clear()

    # 1. Locate feasible meal gaps via cheap simulation (no API calls)
    timeline = _simulate_attraction_timeline(day, cfg)
    slots = get_meal_slots(day)

    # 2. Find meal restaurant candidates at each gap location (meals become nodes)
    meal_waypoints = []
    for meal_name, _mt, _ctx in slots:
        if meal_name in day.dropped_meals:
            continue  # repair stage removed this meal -> do not re-insert
        loc = _meal_gap_location(day, cfg, timeline, meal_name)
        restaurant = find_meal_restaurant(meal_name, loc, cfg, used_restaurants, date_str=day.date)
        if restaurant is None:
            continue
        day.meals[meal_name] = restaurant
        used_restaurants.add(restaurant.name.lower())
        meal_waypoints.append({
            "name": f"{MEAL_LABELS[meal_name]}: {restaurant.name}",
            "location": dict(restaurant.location),
            "kind": "meal",
            "place": restaurant,
            "meal_name": meal_name,
        })

    # 3. Build route destinations: hotel + attractions + meals + hotel
    cin_time = datetime.strptime(cfg.check_in_time, "%H:%M").time()
    base_d = datetime.strptime(day.date, "%Y-%m-%d")
    check_in_dt = base_d.replace(hour=cin_time.hour, minute=cin_time.minute)

    start_duration = 0
    has_explicit_checkin = False

    if day.day_type == "arrival":
        end_name = f"Return to Hotel ({cfg.hotel.get('name', '')})"
        if day.start_time >= check_in_dt:
            start_name = f"Arrive & Check-in ({cfg.hotel.get('name', '')})"
            start_duration = 45 
            has_explicit_checkin = True
        else:
            start_name = f"Arrive & Drop Luggage ({cfg.hotel.get('name', '')})"
            start_duration = 15
    elif day.day_type == "departure":
        start_name = f"Leave Hotel & Drop Bags ({cfg.hotel.get('name', '')})"
        end_name = "Pick Up Bags & Depart for Airport"
    else:
        start_name = f"Leave Hotel ({cfg.hotel.get('name', '')})"
        end_name = f"Return to Hotel ({cfg.hotel.get('name', '')})"

    waypoints = [{"name": start_name, "location": dict(cfg.hotel), "kind": "hotel", "duration_min": start_duration}]
    
    for p in day.attractions:
        waypoints.append({"name": p.name, "location": dict(p.location), "kind": "attraction", "place": p})

    # Handle mid-day check-in or evening check-in fallback
    if day.day_type == "arrival" and not has_explicit_checkin:
        if day.start_time < check_in_dt < day.end_time:
            waypoints.append({
                "name": f"Hotel Check-in ({cfg.hotel.get('name', '')})",
                "location": dict(cfg.hotel),
                "kind": "hotel_checkin",
                "duration_min": 45,
                "place": None
            })
            has_explicit_checkin = True
        
        # If no mid-day check-in was added, force the final node to reflect the check-in
        if not has_explicit_checkin:
            end_name = f"Return & Check-in to Hotel ({cfg.hotel.get('name', '')})"

    waypoints.extend(meal_waypoints)
    waypoints.append({"name": end_name, "location": dict(cfg.hotel), "kind": "hotel"})
    
    # 4. ONE route matrix over all waypoints (controls API usage)
    matrix = DayRouteMatrix(cfg, waypoints)
    order = _optimize_route_temporal(waypoints, matrix, day)
    day.sequence = [waypoints[i] for i in order]

    for k in range(1, len(order)):
        day.sequence[k]["travel_sec_from_prev"] = matrix.travel(order[k - 1], order[k])

    calculate_schedule(day, cfg)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SCHEDULE CALCULATION + OPENING-HOUR CHECK
# ══════════════════════════════════════════════════════════════════════════════

_SPEED = {"DRIVE": 30, "TRANSIT": 20, "WALK": 5, "BICYCLE": 15}

def _parse_hours_range(desc: str):
    m = re.search(r"(\d{1,2}:\d{2}\s*[AP]M)\s*[–\-]\s*(\d{1,2}:\d{2}\s*[AP]M)", desc, re.I)
    if not m: return None
    cv = lambda s: datetime.strptime(s.strip(), "%I:%M %p").strftime("%H:%M")
    return cv(m.group(1)), cv(m.group(2))

def get_opening_window(place: Place, date_str: str):
    if not place.opening_hours: return None
    day_name = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    for desc in place.opening_hours:
        if day_name in desc: return _parse_hours_range(desc)
    return None

def calculate_schedule(day: DayPlan, cfg: TripConfig) -> list[dict]:
    schedule: list[dict] = []
    cur = day.start_time
    prev = dict(cfg.hotel)
    spd = _SPEED.get(cfg.transport_mode, 30)

    target_date = datetime.strptime(day.date, "%Y-%m-%d")

    # FIX: Loop from 0 to capture the start waypoint (Check-in / Leave Hotel)
    for i, wp in enumerate(day.sequence[:-1]):
        loc = wp["location"]
        
        # At index 0, you are already at the hotel; no travel time needed.
        if i == 0:
            travel_s = 0
            arrival = cur
        else:
            travel_s = wp.get("travel_sec_from_prev", int(haversine_km(prev, loc) / spd * 3600))
            arrival = cur + timedelta(seconds=travel_s)

        if wp["kind"] == "meal":
            last_meal_depart = next(
                (e["depart"] for e in reversed(schedule) if e["kind"] == "meal" and e.get("depart")),
                None
            )

            meal_name = wp.get("meal_name")
            if meal_name is None:
                if "Breakfast" in wp["name"]: meal_name = "breakfast"
                elif "Lunch" in wp["name"]: meal_name = "lunch"
                elif "Dinner" in wp["name"]: meal_name = "dinner"

            if meal_name:
                window_start, window_end = _meal_window_datetimes(day, meal_name)
                target = _meal_target_datetime(day, meal_name)

                if last_meal_depart:
                    if meal_name == "lunch": earliest = last_meal_depart + timedelta(hours=2, minutes=30)
                    elif meal_name == "dinner": earliest = last_meal_depart + timedelta(hours=3, minutes=30)
                    else: earliest = last_meal_depart + timedelta(hours=1)
                    window_start = max(window_start, earliest)

                if arrival < window_start:
                    arrival = window_start
                elif arrival > window_end:
                    log.warning("%s scheduled at %s, outside meal window %s-%s", wp["name"], fmt_time(arrival), fmt_time(window_start), fmt_time(window_end))

        place = wp.get("place")
        dur = wp.get("duration_min") if wp.get("duration_min") is not None else (place.visit_duration_min if place else 60)
        
        # ── FIX: DYNAMIC TIME COMPRESSION (ATTRACTIONS ONLY) ──
        if wp["kind"] == "attraction":
            # Calculate time left in day (reserving 30 mins for the drive home)
            time_left_in_day = (day.end_time - arrival).total_seconds() / 60.0 - 30
            
            if dur > time_left_in_day:
                if time_left_in_day < 20:
                    dur = 0 # Too late to visit. Set to 0 so validation catches and deletes it.
                else:
                    dur = int(time_left_in_day) # Compress the visit to fit perfectly

        if place:
            win = get_opening_window(place, day.date)
            if win:
                open_dt  = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
                close_dt = datetime.strptime(day.date + " " + win[1], "%Y-%m-%d %H:%M")

                if close_dt <= open_dt: close_dt += timedelta(days=1)
                if arrival < open_dt: arrival = open_dt
                if arrival >= close_dt:
                    dur = 0
                elif arrival + timedelta(minutes=dur) > close_dt:
                    # Compress the visit to fit before closing time (min 30 mins) instead of 0
                    dur = max(int((close_dt - arrival).total_seconds() / 60), 30)

        depart = arrival + timedelta(minutes=dur)
        
        if schedule and travel_s > 0:
            schedule[-1]["transit_to_next"] = {
                "mode": cfg.transport_mode.lower(),
                "description": f"{cfg.transport_mode.capitalize()} --- {int(travel_s/60)} mins ---> {wp['name']}"
            }

        schedule.append({
            "name": wp["name"], "kind": wp["kind"],
            "arrival": arrival, "depart": depart,
            "travel_sec": travel_s, "duration_min": dur,
            "location": loc,
            "rating": place.rating if place else None,
            "price_level": place.price_level if place else None,
            "opening_hours": place.opening_hours if place else []
        })
        cur, prev = depart, loc

    # Process Final Node cleanly (DRY implementation)
    last_wp = day.sequence[-1]
    last_name = last_wp["name"]
    last_loc = last_wp["location"]

    # Calculate return time
    ret_s = last_wp.get("travel_sec_from_prev", int(haversine_km(prev, last_loc) / spd * 3600))

    # Robust duplicate check: validates against exact name matches or specific keywords
    is_duplicate_end = False
    if schedule:
        prev_name = schedule[-1]["name"].lower()
        is_duplicate_end = (
            schedule[-1]["name"] == last_name or 
            any(sub in prev_name for sub in ["check-in", "check-out", "depart"])
        )

    if not is_duplicate_end:
        if schedule and ret_s > 0:
            schedule[-1]["transit_to_next"] = {
                "mode": cfg.transport_mode.lower(),
                "description": f"{cfg.transport_mode.capitalize()} --- {int(ret_s/60)} mins ---> {last_name}"
            }
        
    schedule.append({
        "name": last_name, 
        "kind": last_wp.get("kind", "hotel"), # Dynamically retains the kind, fallbacks safely
        "arrival": cur + timedelta(seconds=ret_s), "depart": None,
        "travel_sec": ret_s, "duration_min": 0, "location": last_loc,
        "rating": None, "price_level": None, "opening_hours": []
    })

    if day.day_type == "normal" and schedule and schedule[0].get("kind") == "hotel":
        schedule.pop(0)

    if day.day_type == "departure" and schedule:
        first = schedule[0]
        if first.get("kind") == "hotel" and any(token in first["name"].lower() for token in ["leave hotel", "drop bags"]):
            schedule.pop(0)

    day.schedule = schedule
    return schedule



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — CONSTRAINT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_day(day: DayPlan) -> tuple[bool, list[str]]:
    v: list[str] = []
    day_date = datetime.strptime(day.date, "%Y-%m-%d").date()

    # ── Per-entry checks: past-midnight, duration, opening hours ──
    for entry in day.schedule:
        if entry["kind"] == "hotel":
            continue  # the hotel return is validated separately below

        # HARD: nothing may spill into the next calendar day (catches the
        # "Return 01:26" / midnight-wraparound failures).
        if entry["arrival"].date() != day_date:
            v.append(f"{entry['name']}: scheduled at {fmt_time(entry['arrival'])} on the following day")
            continue

        # Enforce 60 min for meals, 20 min for attractions
        min_req = 60 if entry["kind"] == "meal" else 20
        if entry.get("duration_min", 60) < min_req:
            v.append(f"{entry['name']}: visit duration truncated to {entry.get('duration_min')} min (minimum {min_req} required)")

        place = next((wp.get("place") for wp in day.sequence if wp.get("name") == entry["name"]), None)
        if not place:
            continue

        win = get_opening_window(place, day.date)
        if not win:
            continue

        open_dt = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
        close_dt = datetime.strptime(day.date + " " + win[1], "%Y-%m-%d %H:%M")
        if close_dt <= open_dt:
            close_dt += timedelta(days=1)

        if entry["arrival"] >= close_dt:
            v.append(f"{place.name}: arrives after closing {win[1]}")
            continue

        dep = entry.get("depart") or entry["arrival"]
        if dep and dep > close_dt:
            v.append(f"{place.name}: depart {fmt_time(dep)} > close {win[1]}")

    # ── Return-time guard (catches late / next-day returns) ──
    if day.schedule:
        last = day.schedule[-1]["arrival"]
        if last and last > day.end_time:
            v.append(f"Return {fmt_time(last)} > day end {fmt_time(day.end_time)}")

    # ── Meal-window sanity: breakfast, lunch AND dinner ──
    meal_windows = {
        "Breakfast": MEAL_WINDOWS["breakfast"],
        "Lunch": MEAL_WINDOWS["lunch"],
        "Dinner": MEAL_WINDOWS["dinner"],
    }
    meal_times: dict[str, datetime] = {}
    for entry in day.schedule:
        if entry["kind"] != "meal":
            continue
        if entry["arrival"].date() != day_date:
            continue  # already flagged above

        for label, (lo, hi) in meal_windows.items():
            if label in entry["name"]:
                t = entry["arrival"].time()
                meal_times[label] = entry["arrival"]
                if not (lo <= t <= hi):
                    v.append(
                        f"{entry['name']}: served at {fmt_time(entry['arrival'])}, "
                        f"outside acceptable {label.lower()} window "
                        f"{lo.strftime('%H:%M')}-{hi.strftime('%H:%M')}"
                    )

    # ── Chronological meal order: lunch must precede dinner ──
    if "Lunch" in meal_times and "Dinner" in meal_times and meal_times["Lunch"] >= meal_times["Dinner"]:
        v.append("Meals out of order: lunch must precede dinner")

    day.valid = len(v) == 0
    day.violations = v
    return day.valid, v

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — DETERMINISTIC REPAIR → LLM REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def _meal_name_from_entry(entry: dict) -> Optional[str]:
    if "Breakfast" in entry["name"]:
        return "breakfast"
    if "Lunch" in entry["name"]:
        return "lunch"
    if "Dinner" in entry["name"]:
        return "dinner"
    return None


def _drop_bad_meal(day: DayPlan, used_restaurants: set) -> bool:
    day_date = datetime.strptime(day.date, "%Y-%m-%d").date()
    for entry in day.schedule:
        if entry["kind"] != "meal":
            continue
        meal_name = _meal_name_from_entry(entry)
        if not meal_name:
            continue
            
        lo, hi = MEAL_WINDOWS[meal_name]
        t = entry["arrival"].time()
        
        out_of_window = not (lo <= t <= hi)
        next_day = entry["arrival"].date() != day_date
        
        # ── FIX: Check if restaurant closed and truncated the meal ──
        truncated = entry.get("duration_min", 60) < 60
        
        if out_of_window or next_day or truncated:
            place = day.meals.pop(meal_name, None)
            if place is not None:
                used_restaurants.discard(place.name.lower())
            day.dropped_meals.add(meal_name)
            log.info("Repair: dropped invalid/closed %s (%s)", meal_name, entry["name"])
            return True
    return False


def _worst_offender_attraction(day: DayPlan) -> Optional[Place]:
    """Among attractions, find the one most responsible for the schedule
    failing, so repair removes the culprit rather than blindly the weakest."""
    entries = {e["name"]: e for e in day.schedule if e["kind"] == "attraction"}

    def cost(p: Place) -> tuple:
        e = entries.get(p.name)
        closed = 0
        lateness = 0.0
        if e is not None:
            dur = e.get("duration_min", 60)
            if dur <= 0:
                closed = 2          # arrives after closing / fully truncated
            elif dur < 20:
                closed = 1          # heavily truncated
            dep = e.get("depart")
            if dep:
                lateness = max(0.0, (dep - day.end_time).total_seconds() / 60.0)
                # later in the day = more likely to push the return past end
                lateness += (dep - day.start_time).total_seconds() / 3600.0
        # Tie-break toward removing the lower-scored attraction.
        return (closed, lateness, -p.score)

    return max(day.attractions, key=cost)


def deterministic_repair(day: DayPlan, cfg: TripConfig, backups: list[Place], used_restaurants: set) -> bool:
    if validate_day(day)[0]:
        return True

    max_attempts = 5
    for _ in range(max_attempts):
        acted = False

        meal_dropped = _drop_bad_meal(day, used_restaurants)

        if meal_dropped and len(day.attractions) > 1:
            # DO NOT clear day.dropped_meals here. Keep the broken meal slot dropped
            # so the schedule can stabilize with remaining attractions.
            victim = _worst_offender_attraction(day)
            if victim:
                day.attractions.remove(victim)
                log.info("Repair: sacrificed attraction '%s' to make time for schedule.", victim.name)
            acted = True

        elif meal_dropped:
            acted = True

        elif len(day.attractions) > 1:
            victim = _worst_offender_attraction(day)
            if victim is not None:
                day.attractions.remove(victim)
                log.info("Repair: removed worst offender '%s' (score=%.2f)", victim.name, victim.score)
                acted = True

        if not acted:
            break

        build_day_sequence(day, cfg, used_restaurants)
        if validate_day(day)[0]:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def expand_vibe_to_queries(destination: str, custom_vibe: str) -> list[str]:
    """
    Feature 2: Takes a custom user vibe and asks the LLM to generate
    highly targeted Google Search queries for that niche interest.
    """
    if not custom_vibe:
        return []

    sys_p = (
        "You are a local travel expert. Given a destination and a user's niche vibe/interest, "
        "generate 3 specific, highly-targeted search strings that can be used with the Google Places API "
        "to find matching locations. Return JSON only: {\"queries\": [\"string 1\", \"string 2\", ...]}"
    )
    usr_p = f"Destination: {destination}\nUser Vibe/Interest: {custom_vibe}\nGenerate search queries. JSON only."

    res = call_llm(sys_p, usr_p)
    queries = res.get("queries", [])
    log.info(f"Semantic Preference Expansion generated queries for vibe '{custom_vibe}': {queries}")
    return queries


def build_itinerary(cfg: TripConfig) -> dict:
    _cache.clear()
    log.info("═══ Itinerary: %s  %s → %s ═══", cfg.destination, cfg.start_date, cfg.end_date)
    days = classify_days(cfg)
    cfg.preferences = calculate_preference_scores(AVAILABLE_PREFERENCES, cfg.selected_preferences)
    max_candidates = compute_max_candidates(len(days), len(cfg.selected_preferences))
    pool = build_discovery_pool(cfg, max_candidates)
    candidates = discover_candidates(cfg, pool)
    if not candidates: return {"error": "No candidates discovered.", "days": []}

    filtered = filter_candidates(candidates, cfg, max_candidates)
    backups  = filtered[MAX_PER_DAY * 2:]
    n_normal_days = sum(1 for d in days if d.day_type == "normal")
    k_zones = max(1, n_normal_days) # Ensure at least 1 zone
    clusters = cluster_places(filtered, n_clusters=k_zones)
    selected = select_places_llm(filtered, cfg, n_days=len(days))
    sel_clusters = build_selected_clusters(selected, clusters)
    allocate_to_days(selected, days, sel_clusters, cfg, filtered)

    # ── Initialize global tracking set ──
    used_restaurants = set()

    for d in days:
        log.info("── Day %d (%s, %s) ──", d.day_index, d.date, d.day_type)

        # Meals are first-class route nodes. build_day_sequence runs meal
        # placement, temporal route optimization, travel-time recalculation and
        # the schedule in a single API-controlled pass.
        build_day_sequence(d, cfg, used_restaurants)
        ok, viols = validate_day(d)

        if ok:
            log.info("  ✓ Day %d VALID", d.day_index)
            continue

        log.warning("  ✗ Day %d: %s", d.day_index, viols)
        if deterministic_repair(d, cfg, backups, used_restaurants): # <-- PASS TRACKER
            log.info("  ✓ Day %d repaired (deterministic)", d.day_index)
        else:
            log.warning("  ✗ Day %d could not be fully repaired", d.day_index)

    # Clean Output
    out = {"destination": cfg.destination, "start": cfg.start_date, "end": cfg.end_date, "days": []}
    for d in days:
        out["days"].append({
            "day": d.day_index, "date": d.date, "type": d.day_type,
            "valid": d.valid, #"violations": d.violations,
            "schedule": [
                {
                    "time": fmt_time(e["arrival"]),
                    "name": e["name"],
                    "kind": e["kind"],
                    "duration_min": e["duration_min"],
                    "travel_time_min": round(e.get("travel_sec", 0) / 60),
                    "location": {
                        "latitude": e["location"].get("latitude", 0.0),
                        "longitude": e["location"].get("longitude", 0.0),
                        "name": e["location"].get("name") or e.get("name", ""),
                        "address": e.get("address") or e["location"].get("address") or ""
                    },
                    "rating": e.get("rating"),
                    "price_level": e.get("price_level"),
                    "opening_hours": e.get("opening_hours"),
                    "transit_to_next": e.get("transit_to_next")
                }
                for e in d.schedule],
            "attractions": [p.name for p in d.attractions],
            "meals": {k: v.name for k, v in d.meals.items()},
        })
    log.info("═══ Done: %d days ═══", len(days))
    return out


if __name__ == "__main__":
    cfg = TripConfig(
        destination="Tokyo, Japan",
        start_date="2026-09-01", end_date="2026-09-03",
        arrival_datetime="2026-09-01T15:00:00", departure_datetime="2026-09-03T18:00:00",
        hotel={"name": "Hotel Gracery Shinjuku", "latitude": 35.6938, "longitude": 139.7036},
        airport={"name": "Narita Airport", "latitude": 35.7720, "longitude": 140.3929},
        selected_preferences=["culture", "food", "scenery"],
        travel_style="packed", transport_mode="TRANSIT",   
        group_size=2, budget="medium",    # Group size havent implement or no need implement
        check_in_time="14:00",
        check_out_time="11:00",
        custom_vibe="Cyberpunk and electronic"
    )

    itinerary = build_itinerary(cfg)
    
    # Save output as JSON file
    output_file = "itinerary_output.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(itinerary, f, indent=2, ensure_ascii=False)

    print(f"Itinerary saved to: {output_file}")