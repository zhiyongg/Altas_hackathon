"""
itinerary_planner.py — Agentic Trip Itinerary Planner
======================================================
Pipeline:
  User Input → Day Classification → Candidate Discovery → Deterministic Filter
  → Geographic Clustering → LLM Selection → Daily Allocation
  → Route Matrix (Order) → Meal Placement → Route Matrix (Timings) 
  → Schedule Calculation → Validation → Repair → Output JSON
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

# ── Configuration ─────────────────────────────────────────────────────────────
DISCOVERY_RADIUS_M   = 15_000       
MIN_PER_DAY          = 3
MAX_PER_DAY          = 8

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
    "relaxed":  {"normal_min": 2, "normal_max": 5, "arrival": 1, "departure": 1},
    "moderate": {"normal_min": 3, "normal_max": 7, "arrival": 2, "departure": 2},
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
    return datetime.fromisoformat(s)

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
        "places.formattedAddress", "places.rating",
        "places.regularOpeningHours.weekdayDescriptions",
        "places.priceLevel", "places.primaryType", "places.types",
        "places.userRatingCount",
    ]
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(fields),
    }

def search_nearby(lat: float, lng: float, radius: float = DISCOVERY_RADIUS_M,
                  included_types: list[str] | None = None, max_results: int = 20) -> list[dict]:
    print("nearbySearch called")
    url = "https://places.googleapis.com/v1/places:searchNearby"
    payload = {
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": radius}
        },
    }
    if included_types: payload["includedTypes"] = included_types
    resp = requests.post(url, headers=_places_headers(), json=payload)
    if resp.status_code != 200:
        log.warning("Nearby Search %s: %s", resp.status_code, resp.text)
        return []
    return resp.json().get("places", [])

def search_text(query: str, max_results: int = 10) -> list[dict]:
    print("searchText called")
    url = "https://places.googleapis.com/v1/places:searchText"
    payload = {"textQuery": query, "pageSize": max_results}
    resp = requests.post(url, headers=_places_headers(), json=payload)
    if resp.status_code != 200:
        log.warning("Text Search %s: %s", resp.status_code, resp.text)
        return []
    return resp.json().get("places", [])

def compute_route_matrix(origins: list[dict], destinations: list[dict],
                         travel_mode="DRIVE", routing_pref="TRAFFIC_AWARE") -> list[dict]:
    print("computeRoadMatrix called")
    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    mask = ["originIndex", "destinationIndex", "duration", "distanceMeters", "condition", "status"]
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(mask),
    }
    
    def _clean_ll(loc): return {"latitude": loc["latitude"], "longitude": loc["longitude"]}

    payload = {
        "origins":  [{"waypoint": {"location": {"latLng": _clean_ll(o)}}} for o in origins],
        "destinations": [{"waypoint": {"location": {"latLng": _clean_ll(d)}}} for d in destinations],
        "travelMode": travel_mode,
    }
    
    if travel_mode == "TRANSIT":
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT09:00:00Z")
        payload["departureTime"] = tomorrow
    elif travel_mode == "DRIVE":
        payload["routingPreference"] = routing_pref

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        log.warning("Route Matrix 400: %s", resp.text)
        return []
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# LLM WRAPPER  
# ══════════════════════════════════════════════════════════════════════════════

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

from schemas import TripConfig, Place, DayPlan


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
        d = DayPlan(day_index=len(days), date=cur.isoformat(), day_type=dtype, base_location=dict(cfg.hotel))
        
        base = datetime.strptime(d.date, "%Y-%m-%d")
        
        # 1. Calculate precise start and end times
        if dtype == "arrival":
            d.start_time = arrival_dt + timedelta(hours=2)
            d.end_time = arrival_dt.replace(hour=22, minute=0, second=0)
        elif dtype == "departure":
            cout_time = datetime.strptime(cfg.check_out_time, "%H:%M").time()
            default_start = base.replace(hour=9, minute=0)
            checkout_dt = base.replace(hour=cout_time.hour, minute=cout_time.minute)
            
            # Start touring at 9 AM, OR the check-out time (whichever is earlier)
            d.start_time = min(default_start, checkout_dt)
            d.end_time = departure_dt - timedelta(hours=3)
        else:
            d.start_time = base.replace(hour=9, minute=0)
            d.end_time = base.replace(hour=22, minute=0)
            
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
            d.capacity_max = max(0, calculated_max)
            # Require at least 1 attraction if there is time, otherwise 0
            d.capacity_min = min(1, d.capacity_max) 
            
        days.append(d)
        cur += timedelta(days=1)
        
    return days


def get_meal_slots(dp: DayPlan) -> list[tuple]:
    """
    Universally calculates applicable meal slots based on the day's 
    actual start and end times (derived from arrival/departure/normal schedules).
    Returns [(meal_name, target_datetime, context_label), ...]
    """
    slots = []
    base = datetime.strptime(dp.date, "%Y-%m-%d")
    
    start_t = dp.start_time.time()
    end_t = dp.end_time.time()
    
    # Standard meal target datetimes
    b_time = base.replace(hour=8, minute=0)
    l_time = base.replace(hour=12, minute=30)
    d_time = base.replace(hour=18, minute=30)
    
    # 1. Breakfast: Included if the day starts by 9:00 AM
    if start_t <= dtime(9, 0):
        slots.append(("breakfast", b_time, "hotel"))
        
    # 2. Lunch: Included if the user is active through the midday window (12:30 - 13:30)
    if start_t <= dtime(12, 30) and end_t >= dtime(13, 30):
        slots.append(("lunch", l_time, "attraction"))
        
    # 3. Dinner: Included if the day extends past 6:30 PM
    if end_t >= dtime(18, 30):
        # If it's an arrival day ending late, eat near hotel; otherwise near attractions
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
        "discovery_target": min(max_candidates * 2, 200),
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

def _raw_to_place(raw: dict, source="api", travel_style: str = "moderate") -> Place | None:
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
        location={"latitude": loc["latitude"], "longitude": loc["longitude"]},
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

    # 1. Standard preference theme searches
    themes = sorted(cfg.preferences.keys(), key=lambda t: cfg.preferences[t], reverse=True)
    for theme in themes[:pool["n_theme_searches"]]:
        for raw in search_text(f"{theme} attractions in {cfg.destination}",
                                max_results=pool["text_results_per_theme"]):
            p = _raw_to_place(raw, "text", cfg.travel_style) 
            if p and p.id not in seen and p.primary_type not in banned_primary_types:
                seen.add(p.id)
                places.append(p)

    # ── FEATURE 2 INTEGRATION: Semantic Vibe Search Expansion ──
    if cfg.custom_vibe:
        vibe_queries = expand_vibe_to_queries(cfg.destination, cfg.custom_vibe)
        for q in vibe_queries:
            for raw in search_text(q, max_results=5):
                p = _raw_to_place(raw, "text", cfg.travel_style)
                if p and p.id not in seen and p.primary_type not in banned_primary_types:
                    seen.add(p.id)
                    places.append(p)
    # ───────────────────────────────────────────────────────────

    # 2. Nearby search expansion passes
    radius = DISCOVERY_RADIUS_M
    max_passes = 3 if discovery_target > 80 else (2 if discovery_target > 40 else 1)

    for pass_i in range(max_passes):
        if len(places) >= discovery_target:
            break
        for i in range(0, len(ATTRACTION_TYPES), 5):
            chunk = ATTRACTION_TYPES[i:i + 5]
            for raw in search_nearby(hlat, hlng, radius=radius, included_types=chunk,
                                      max_results=pool["nearby_results_per_chunk"]):
                p = _raw_to_place(raw, "nearby", cfg.travel_style)
                if p and p.id not in seen and p.primary_type not in banned_primary_types:
                    seen.add(p.id)
                    places.append(p)
            if len(places) >= discovery_target:
                break
        radius = int(radius * 1.5)

    log.info("Discovered %d unique candidates (target %d, %d pass(es)).",
              len(places), discovery_target, max_passes)
    return places

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DETERMINISTIC FILTER + SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _budget_score(p: Place, budget: str) -> float:
    allowed = BUDGET_PRICE_LEVELS.get(budget.lower(), set())
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

def _is_too_similar(name1: str, name2: str) -> bool:
    # Clean the names (lowercase, remove punctuation)
    clean1 = re.sub(r'[^\w\s]', '', name1.lower())
    clean2 = re.sub(r'[^\w\s]', '', name2.lower())
    
    w1 = set(clean1.split())
    w2 = set(clean2.split())
    
    # Ignore generic words that cause false positives (e.g., "Tokyo Park" vs "Ueno Park")
    generic = {"the", "of", "and", "tokyo", "japan", "park", "museum", "shrine", 
               "temple", "garden", "national", "city", "tower", "station", "building"}
    
    w1 = w1 - generic
    w2 = w2 - generic
    
    if not w1 or not w2: 
        return False
        
    # Count how many core words they share
    overlap = len(w1.intersection(w2))
    
    # If one name is entirely contained inside the other (e.g., "Meiji Jingu" in "Meiji Jingu Gaien")
    # Or if they share a massive amount of words
    if overlap == len(w1) or overlap == len(w2):
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
            if _is_too_similar(p.name, u.name):
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
    sys_p = 'You are a travel planner AI. Select attractions matching user preferences. JSON only: {"selected_indices": [0, 2, 5, ...]}'
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

    # ── FIX 2: THE "USE IT OR LOSE IT" SWEEP ──
    # Distribute any remaining LLM-selected attractions to fill the afternoon gaps!
    for day in normal:
        if len(day.attractions) < day.capacity_max:
            # Combine the LLM selection + the 40 deterministic backups.
            # If selected runs out, it smoothly continues grabbing high-scored backups!
            for p in selected + filtered_candidates: 
                if p.id not in used_ids:
                    day.attractions.append(p)
                    used_ids.add(p.id)
                if len(day.attractions) >= day.capacity_max:
                    break


def place_meals(day: DayPlan, cfg: TripConfig, used_restaurants: set) -> None:
    
    # ── SMART FILTER ENGINE ──
    def _is_valid_restaurant(r: Place) -> bool:
        if not r: return False
        
        # 1. Ban hotels and non-food venues explicitly
        banned_words = {"hotel", "resort", "hostel", "ryokan", "inn", "guest house"}
        banned_types = {"lodging", "bar", "night_club"}
        r_name_lower = r.name.lower()
        
        if any(bt in r.types for bt in banned_types): return False
        if any(bw in r_name_lower for bw in banned_words): return False

        # 2. Ban exact matches
        if r_name_lower in used_restaurants: return False
        
        # 3. Ban chain restaurants (e.g. catch "Ichiran Shibuya" if "Ichiran Shinjuku" was eaten)
        words = r_name_lower.split()
        if words:
            first_word = words[0]
            generic_words = {"cafe", "restaurant", "the", "bistro", "brasserie", "pizzeria", "la"}
            # If the first word is a unique brand name (like "ichiran", "mcdonalds", "starbucks")
            if len(first_word) > 3 and first_word not in generic_words:
                for used in used_restaurants:
                    if used.startswith(first_word):
                        return False # Brand already eaten this trip!
        
        return True
    # ─────────────────────────

    # Remove this day's existing meals from the global ban-list before recalculating
    for m in day.meals.values():
        used_restaurants.discard(m.name.lower())
    day.meals.clear() 
    
    slots = get_meal_slots(day)
    
    # ── Days with NO attractions (Departure Days) ──
    if not day.attractions:
        for meal_name, _mt, ctx in slots:
            meal_types = ["cafe", "bakery"] if meal_name == "breakfast" else ["restaurant"]
            raw = search_nearby(cfg.hotel["latitude"], cfg.hotel["longitude"], radius=1000, included_types=meal_types, max_results=10)
            
            # Apply the smart filter
            rests = [r for r in (_raw_to_place(x, "meal", cfg.travel_style) for x in raw) if _is_valid_restaurant(r)]
            
            if rests:
                best = max(rests, key=lambda r: r.rating or 0)
                day.meals[meal_name] = best
                used_restaurants.add(best.name.lower())
        
        new_seq = [day.sequence[0]]
        for mn in ["breakfast", "lunch", "dinner"]:
            if mn in day.meals:
                m = day.meals[mn]
                new_seq.append({"name": f"{mn.title()}: {m.name}", "location": dict(m.location), "kind": "meal", "place": m})
        new_seq.append(day.sequence[-1])
        day.sequence = new_seq
        return

    # ── Normal Days (With Attractions) ──
    prelim: list[dict] = []
    cur_time = day.start_time
    spd = _SPEED.get(cfg.transport_mode, 30)
    prev_loc = dict(cfg.hotel)

    for wp in day.sequence[1:-1]:
        loc = wp["location"]
        travel_s = wp.get("travel_sec_from_prev", int(haversine_km(prev_loc, loc) / spd * 3600))
        arrival = cur_time + timedelta(seconds=travel_s)
        place = wp.get("place")
        dur = place.visit_duration_min if place else 60
        if place:
            win = get_opening_window(place, day.date)
            if win:
                open_dt = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
                if arrival < open_dt: arrival = open_dt
        depart = arrival + timedelta(minutes=dur)
        prelim.append({"name": wp["name"], "depart": depart, "location": loc, "place": place})
        cur_time = depart
        prev_loc = loc

    noon = datetime.strptime(day.date + " 12:30", "%Y-%m-%d %H:%M")
    lunch_wp = None
    best_diff = float("inf")
    for entry in prelim:
        diff = abs((entry["depart"] - noon).total_seconds())
        if diff < best_diff:
            best_diff = diff
            lunch_wp = entry

    dinner_wp = prelim[-1] if prelim else None

    if dinner_wp and dinner_wp["depart"].hour < 16:
        dinner_wp = None 

    for meal_name, _mt, ctx in slots:
        loc = cfg.hotel
        meal_types = ["restaurant"]
        if meal_name == "breakfast": meal_types = ["cafe", "bakery"]
        elif meal_name == "lunch" and lunch_wp: loc, meal_types = lunch_wp["location"], ["restaurant", "cafe"]
        elif meal_name == "dinner" and dinner_wp: loc, meal_types = dinner_wp["location"], ["restaurant", "steak_house"]

        raw = search_nearby(loc["latitude"], loc["longitude"], radius=1000, included_types=meal_types, max_results=10)
        
        # Apply the smart filter
        rests = [r for r in (_raw_to_place(x, "meal", cfg.travel_style) for x in raw) if _is_valid_restaurant(r)]
        
        if rests: 
            best = max(rests, key=lambda r: r.rating or 0)
            day.meals[meal_name] = best
            used_restaurants.add(best.name.lower())

    lunch_wp_name = lunch_wp["name"] if lunch_wp else None
    dinner_wp_name = dinner_wp["name"] if dinner_wp else None
    
    new_seq = [day.sequence[0]] 
    if "breakfast" in day.meals:
        m = day.meals["breakfast"]
        new_seq.append({"name": f"Breakfast: {m.name}", "location": dict(m.location), "kind": "meal", "place": m})
        
    for wp in day.sequence[1:-1]:
        new_seq.append(wp)
        if lunch_wp_name and wp["name"] == lunch_wp_name and "lunch" in day.meals:
            m = day.meals["lunch"]
            new_seq.append({"name": f"Lunch: {m.name}", "location": dict(m.location), "kind": "meal", "place": m})
        if dinner_wp_name and wp["name"] == dinner_wp_name and "dinner" in day.meals:
            m = day.meals["dinner"]
            new_seq.append({"name": f"Dinner: {m.name}", "location": dict(m.location), "kind": "meal", "place": m})
            
    if not dinner_wp_name and "dinner" in day.meals:
        m = day.meals["dinner"]
        new_seq.append({"name": f"Dinner: {m.name}", "location": dict(m.location), "kind": "meal", "place": m})

    new_seq.append(day.sequence[-1]) 
    day.sequence = new_seq

    
# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — ROUTE MATRIX (Order) & UPDATE TRAVEL TIMES
# ══════════════════════════════════════════════════════════════════════════════

def route_matrix_best_order(day: DayPlan, cfg: TripConfig) -> None:
    # SAFETY RAILS: Force max 8 attractions (10 total locations)
    if len(day.attractions) > 8:
        log.warning(f"Day {day.day_index} has {len(day.attractions)} attractions. Truncating to 8.")
        day.attractions = day.attractions[:8]
        
    # ── NEW: Smart Hotel Naming Logic ──
    if day.day_type == "arrival":
        start_name = f"Drop Luggage ({cfg.hotel.get('name', '')})"
        end_name = f"Check-in & Rest ({cfg.hotel.get('name', '')})"
    elif day.day_type == "departure":
        start_name = f"Check-out & Leave Bags ({cfg.hotel.get('name', '')})"
        end_name = f"Pick Up Bags & Depart for Airport"
    else:
        start_name = f"Leave Hotel ({cfg.hotel.get('name', '')})"
        end_name = f"Return to Hotel ({cfg.hotel.get('name', '')})"

    seq = [{"name": start_name, "location": dict(cfg.hotel), "kind": "hotel"}]
    for p in day.attractions:
        seq.append({"name": p.name, "location": dict(p.location), "kind": "attraction", "place": p})
    seq.append({"name": end_name, "location": dict(cfg.hotel), "kind": "hotel"})

    if len(seq) <= 3:
        day.sequence = seq
        update_sequence_travel_times(day, cfg)
        return

    locs = [s["location"] for s in seq]
    matrix = compute_route_matrix(locs, locs, travel_mode=cfg.transport_mode)

    n = len(seq)
    tt = [[0.0] * n for _ in range(n)]
    for e in matrix:
        oi, di = e.get("originIndex", 0), e.get("destinationIndex", 0)
        tt[oi][di] = parse_dur(e.get("duration"))

    hotel_i, hotel_end = 0, n - 1
    
    # ── FIX 3: Nightlife Temporal Constraint ──
    # Separate daytime attractions from evening/nightlife attractions
    nightlife_types = {"bar", "night_club", "pub", "wine_bar", "cocktail_bar", "karaoke"}
    standard_idx = []
    night_idx = []
    for i, s in enumerate(seq):
        if s["kind"] == "attraction":
            if any(t in nightlife_types for t in s["place"].types):
                night_idx.append(i)
            else:
                standard_idx.append(i)
                
    ordered, visited, cur = [hotel_i], {hotel_i}, hotel_i
    
    # 1. TSP standard daytime attractions first
    for _ in range(len(standard_idx)):
        best_i, best_t = None, float("inf")
        for j in standard_idx:
            if j not in visited and tt[cur][j] < best_t:
                best_i, best_t = j, tt[cur][j]
        if best_i is not None:
            ordered.append(best_i)
            visited.add(best_i)
            cur = best_i
            
    # 2. TSP nightlife attractions immediately after
    for _ in range(len(night_idx)):
        best_i, best_t = None, float("inf")
        for j in night_idx:
            if j not in visited and tt[cur][j] < best_t:
                best_i, best_t = j, tt[cur][j]
        if best_i is not None:
            ordered.append(best_i)
            visited.add(best_i)
            cur = best_i

    ordered.append(hotel_end)
    day.sequence = [seq[i] for i in ordered]
    update_sequence_travel_times(day, cfg)



def update_sequence_travel_times(day: DayPlan, cfg: TripConfig) -> None:
    if len(day.sequence) < 2: return
    
    tt = {}
    chunk_size = 5
    
    for i in range(0, len(day.sequence) - 1, chunk_size):
        end_idx = min(i + chunk_size, len(day.sequence) - 1)
        
        origins = [s["location"] for s in day.sequence[i:end_idx]]
        destinations = [s["location"] for s in day.sequence[i+1:end_idx+1]]
        
        matrix = compute_route_matrix(origins, destinations, travel_mode=cfg.transport_mode)
        
        for e in matrix:
            # ── FIX: Provide safe integer defaults ──
            oi = e.get("originIndex", 0)
            di = e.get("destinationIndex", 0)
            
            if oi == di:
                global_idx = i + oi
                tt[global_idx] = parse_dur(e.get("duration"))
                
    for i in range(len(day.sequence) - 1):
        dur = tt.get(i)
        if not dur:  
            loc1 = day.sequence[i]["location"]
            loc2 = day.sequence[i+1]["location"]
            dur = int((haversine_km(loc1, loc2) / 5.0) * 3600) 
        day.sequence[i+1]["travel_sec_from_prev"] = dur


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

    # ── FIX: Baseline target date for safe meal clamping ──
    target_date = datetime.strptime(day.date, "%Y-%m-%d")

    for wp in day.sequence[1:-1]:
        loc = wp["location"]
        travel_s = wp.get("travel_sec_from_prev", int(haversine_km(prev, loc) / spd * 3600))
        arrival = cur + timedelta(seconds=travel_s)

        # ── FIX: Absolute Datetime comparisons ──
        if wp["kind"] == "meal":
            if "Lunch" in wp["name"]:
                min_lunch = target_date.replace(hour=11, minute=30)
                if arrival < min_lunch: arrival = min_lunch
            elif "Dinner" in wp["name"]:
                min_dinner = target_date.replace(hour=18, minute=0)
                if arrival < min_dinner: arrival = min_dinner

        place = wp.get("place")
        dur = wp.get("place").visit_duration_min if place else 60
        if place:
            win = get_opening_window(place, day.date)
            if win:
                open_dt  = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
                close_dt = datetime.strptime(day.date + " " + win[1], "%Y-%m-%d %H:%M")
                
                if close_dt <= open_dt: close_dt += timedelta(days=1)
                if arrival < open_dt: arrival = open_dt
                if arrival + timedelta(minutes=dur) > close_dt:
                    dur = max(int((close_dt - arrival).total_seconds() / 60), 0)

        depart = arrival + timedelta(minutes=dur)
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

    last_loc = day.sequence[-1]["location"]
    ret_s = day.sequence[-1].get("travel_sec_from_prev", int(haversine_km(prev, last_loc) / spd * 3600))
    schedule.append({
        "name": "Return to Hotel", "kind": "hotel",
        "arrival": cur + timedelta(seconds=ret_s), "depart": None,
        "travel_sec": ret_s, "duration_min": 0, "location": last_loc,
        "rating": None, "price_level": None, "opening_hours": []
    })
    
    day.schedule = schedule
    return schedule



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — CONSTRAINT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_day(day: DayPlan) -> tuple[bool, list[str]]:
    v: list[str] = []
    for entry in day.schedule:
        # Only skip the hotel return, everything else gets validated
        if entry["kind"] == "hotel": continue
        
        # Enforce 60 min for meals, 20 min for attractions
        min_req = 60 if entry["kind"] == "meal" else 20
        
        if entry.get("duration_min", 60) < min_req:
            v.append(f"{entry['name']}: visit duration truncated to {entry.get('duration_min')} min (minimum {min_req} required)")

        place = next((wp.get("place") for wp in day.sequence if wp.get("name") == entry["name"]), None)
        if not place: continue
        
        win = get_opening_window(place, day.date)
        if not win: continue
        
        open_dt  = datetime.strptime(day.date + " " + win[0], "%Y-%m-%d %H:%M")
        close_dt = datetime.strptime(day.date + " " + win[1], "%Y-%m-%d %H:%M")
        if close_dt <= open_dt: close_dt += timedelta(days=1)
            
        dep = entry.get("depart") or entry["arrival"]
        if dep and dep > close_dt:
            v.append(f"{place.name}: depart {fmt_time(dep)} > close {win[1]}")
            
    if day.schedule:
        last = day.schedule[-1]["arrival"]
        if last and last > day.end_time:
            v.append(f"Return {fmt_time(last)} > day end {fmt_time(day.end_time)}")
            
    day.valid = len(v) == 0
    day.violations = v
    return day.valid, v

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — DETERMINISTIC REPAIR → LLM REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def deterministic_repair(day: DayPlan, cfg: TripConfig, backups: list[Place], used_restaurants: set) -> bool:
    route_matrix_best_order(day, cfg)
    place_meals(day, cfg, used_restaurants)
    update_sequence_travel_times(day, cfg)
    calculate_schedule(day, cfg)
    
    if validate_day(day)[0]: 
        return True

    max_drops = 2
    drops = 0
    
    while len(day.attractions) > 1 and drops < max_drops:
        day.attractions.sort(key=lambda p: p.score)
        rm = day.attractions.pop(0)
        log.info("Repair: removed %s (score=%.2f) to free up time", rm.name, rm.score)
        
        route_matrix_best_order(day, cfg)
        place_meals(day, cfg, used_restaurants)
        update_sequence_travel_times(day, cfg)
        calculate_schedule(day, cfg)
        
        if validate_day(day)[0]: 
            return True
            
        drops += 1

    return False

def llm_repair(day: DayPlan, cfg: TripConfig, backups: list[Place], used_restaurants: set) -> bool:
    bk = [{"i": i, "name": p.name, "score": round(p.score, 2), "types": p.types[:3]} for i, p in enumerate(backups)]
    sys_p = ('You repair itinerary violations. JSON: '
             '{"action":"replace"|"remove", "remove_indices":[…], "add_indices":[…]}')
    usr_p = (
        f"Day {day.day_index} violations: {json.dumps(day.violations)}\n"
        f"Attractions: {[p.name for p in day.attractions]}\n"
        f"Backups:\n{json.dumps(bk)}\nSuggest repair. JSON only.")
        
    res = call_llm(sys_p, usr_p)
    act = res.get("action", "")
    changed = False

    if act == "replace":
        rm_idxs = sorted([i for i in res.get("remove_indices", []) if isinstance(i, int) and 0 <= i < len(day.attractions)], reverse=True)
        for ri in rm_idxs:
            day.attractions.pop(ri)
            changed = True
        for ai in res.get("add_indices", []):
            if isinstance(ai, int) and 0 <= ai < len(backups):
                day.attractions.append(backups[ai])
                changed = True
    elif act == "remove":
        rm_idxs = sorted([i for i in res.get("remove_indices", []) if isinstance(i, int) and 0 <= i < len(day.attractions)], reverse=True)
        for ri in rm_idxs:
            day.attractions.pop(ri)
            changed = True

    if not changed and day.attractions:
        day.attractions.sort(key=lambda p: p.score)
        rm = day.attractions.pop(0)
        log.info("LLM repair fallback: removed %s", rm.name)

    route_matrix_best_order(day, cfg)
    place_meals(day, cfg, used_restaurants)
    update_sequence_travel_times(day, cfg)
    calculate_schedule(day, cfg)
    return validate_day(day)[0]
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
        "generate 4 to 6 specific, highly-targeted search strings that can be used with the Google Places API "
        "to find matching locations. Return JSON only: {\"queries\": [\"string 1\", \"string 2\", ...]}"
    )
    usr_p = f"Destination: {destination}\nUser Vibe/Interest: {custom_vibe}\nGenerate search queries. JSON only."
    
    res = call_llm(sys_p, usr_p)
    queries = res.get("queries", [])
    log.info(f"Semantic Preference Expansion generated queries for vibe '{custom_vibe}': {queries}")
    return queries


def build_itinerary(cfg: TripConfig) -> dict:
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

        route_matrix_best_order(d, cfg)          
        place_meals(d, cfg, used_restaurants)     # <-- PASS TRACKER
        update_sequence_travel_times(d, cfg)
        calculate_schedule(d, cfg)                
        ok, viols = validate_day(d)               

        if ok:
            log.info("  ✓ Day %d VALID", d.day_index)
            continue
            
        log.warning("  ✗ Day %d: %s", d.day_index, viols)
        if deterministic_repair(d, cfg, backups, used_restaurants): # <-- PASS TRACKER
            log.info("  ✓ Day %d repaired (deterministic)", d.day_index)
            continue
            
        log.info("  ↳ LLM repair for day %d…", d.day_index)
        if llm_repair(d, cfg, backups, used_restaurants):           # <-- PASS TRACKER
            log.info("  ✓ Day %d repaired (LLM)", d.day_index)
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
                    "location": e["location"],
                    "rating": e.get("rating"),
                    "price_level": e.get("price_level"),
                    "opening_hours": e.get("opening_hours")
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
        travel_style="moderate", transport_mode="TRANSIT",   
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