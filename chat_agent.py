"""
chat.py — the full post-creation chatbot layer consolidated into ONE file.

This is a straight merge of the original six modules, in dependency order,
with NO behavioral changes:

    chat_schemas.py       -> Section 1: data models (ChatAction, IntentResult,
                                       ChatMessage, HandlerResult)
    intent_classifier.py  -> Section 2: LLM intent gateway (chat_llm,
                                       classify_intent)
    chat_state.py         -> Section 3: persistence, versioning, chat history
                                       (SessionStore, ChatHistoryManager,
                                       serialization helpers)
    action_handlers.py    -> Section 4: deterministic action executors
    chat_engine.py        -> Section 5: ChatEngine orchestrator + demo REPL
    test_chat_features.py -> Section 6: offline feature tests (unittest)

Why one file still behaves like six:
* Every original module name is registered in sys.modules at the bottom of
  this file (see BACKWARD COMPATIBILITY), so existing code such as
  `from chat_engine import ChatEngine`, `import chat_state` or
  `from chat_schemas import ChatAction` keeps working unchanged.
* unittest.mock.patch() targets like "intent_classifier.chat_llm" or
  "chat_state.build_state_from_output" also keep working, because the merged
  functions resolve those names through the module globals at call time.

Only external dependency: `itineraryPlanner` (the planner itself is NOT
part of this merge and stays untouched).

Run the interactive demo directly:
    python chat.py
Run the tests directly:
    python chat.py --test
"""

import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum

from google.genai import types

# In --test mode the suite is fully offline: dummy API keys are injected
# BEFORE itineraryPlanner is imported (it raises ValueError at import when
# GOOGLE_MAPS_API_KEY / GEMINI_API_KEY are missing). Real runs are unaffected
# because setdefault() never overwrites existing env vars.
if "--test" in sys.argv:
    os.environ.setdefault("GOOGLE_MAPS_API_KEY", "dummy-maps-key-for-tests")
    os.environ.setdefault("GEMINI_API_KEY", "dummy-gemini-key-for-tests")

import itineraryPlanner as ip
from itineraryPlanner import DayPlan, Place, TripConfig

# ══════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ══════════════════════════════════════════════════════════════════════════════

MAX_VERSIONS = 10

# Anchored to this module's directory so sessions never scatter across CWDs.
DEFAULT_SESSIONS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sessions")

VALID_TRANSPORT_MODES = {"DRIVE", "TRANSIT", "WALK", "BICYCLE"}
MEAL_SLOTS = ("breakfast", "lunch", "dinner")
MAX_ATTRACTIONS_PER_DAY = 8          # mirrors build_day_sequence's safety cap
NAME_MATCH_THRESHOLD = 0.45
DUPLICATE_THRESHOLD = 0.8
BANNED_PRIMARY_TYPES = {"restaurant", "cafe", "bar", "lodging", "hotel"}

# Loose synonyms so "nature" / "history" style requests map onto the planner's
# AVAILABLE_PREFERENCES vocabulary.
PREFERENCE_SYNONYMS = {
    "nature": "scenery", "outdoors": "scenery", "parks": "scenery",
    "history": "culture", "historical": "culture", "museums": "culture",
    "art": "culture", "temples": "culture", "shrines": "culture",
    "nightlife": "entertainment", "music": "entertainment",
    "dining": "food", "restaurants": "food", "eating": "food",
    "hiking": "adventure", "sports": "adventure",
    "spa": "wellness", "relaxation": "wellness",
    "sightseeing": "city", "landmarks": "city",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data models (was chat_schemas.py)
# ══════════════════════════════════════════════════════════════════════════════

class ChatAction(str, Enum):
    """Every intent the chat layer knows how to execute deterministically."""
    QUESTION           = "QUESTION"
    ADD_PLACE          = "ADD_PLACE"
    REMOVE_PLACE       = "REMOVE_PLACE"
    REPLACE_PLACE      = "REPLACE_PLACE"
    CHANGE_TRANSPORT   = "CHANGE_TRANSPORT"
    CHANGE_SCHEDULE    = "CHANGE_SCHEDULE"
    REPLAN_DAY         = "REPLAN_DAY"
    UPDATE_PREFERENCES = "UPDATE_PREFERENCES"
    REGENERATE_TRIP    = "REGENERATE_TRIP"
    UNDO               = "UNDO"


@dataclass
class IntentResult:
    """Output of the LLM intent classifier.

    action   — validated ChatAction value.
    params   — extracted parameters (place_name, day 0-based index,
               new_transport, new_start_time/new_end_time, category,
               preferences, pacing, ...).
    fallback — True when the LLM output was empty/unknown and we defaulted
               to QUESTION.
    message  — the original user message (filled in by the engine so
               handlers can quote/inspect it).
    """
    action: ChatAction
    params: dict = field(default_factory=dict)
    fallback: bool = False
    message: str = ""


@dataclass
class ChatMessage:
    role: str            # "user" | "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content,
                "timestamp": self.timestamp}


@dataclass
class HandlerResult:
    """What every action handler returns to the engine."""
    itinerary_changed: bool
    response_text: str
    updated_days: list = field(default_factory=list)  # 0-based day indexes


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LLM intent gateway (was intent_classifier.py)
#
# The LLM ONLY classifies intent / extracts parameters here. All itinerary
# mutations are performed by deterministic code in Section 4.
#
# chat_llm() is deliberately UNCACHED (unlike itineraryPlanner.call_llm which
# is @cached) so answers always reflect the freshest itinerary context.
# ══════════════════════════════════════════════════════════════════════════════

_log_intent = logging.getLogger("chat.intent")

VALID_ACTIONS = {a.value for a in ChatAction}


def chat_llm(system_prompt: str, user_prompt: str) -> dict:
    """Uncached Gemini JSON call. Returns {} on ANY failure so callers can
    always degrade gracefully (offline mode / stubbed tests)."""
    text = ""
    try:
        response = ip.gemini_client.models.generate_content(
            model=ip.GEMINI_MODEL,
            contents=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
            config=types.GenerateContentConfig(
                temperature=0.2, response_mime_type="application/json"),
        )
        text = response.text or ""
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return {}
        return {}
    except Exception as e:
        _log_intent.warning("chat_llm failed: %s", e)
        return {}


_CLASSIFIER_SYSTEM = """You are the intent classification engine of a travel-itinerary chatbot.
The user already has a generated itinerary and is asking to inspect or modify it.
Classify the LAST user message into exactly ONE action and extract its parameters.

Actions and parameter schemas:
- QUESTION            {"day": <0-based int, optional>}                  — any question / smalltalk / unclear request
- ADD_PLACE           {"place_name": str, "day": <0-based int>}         — add a specific place to a day
- REMOVE_PLACE        {"place_name": str, "day": <0-based int, optional>} — remove a place or a meal ("lunch"/"dinner"/"breakfast")
- REPLACE_PLACE       {"place_name": str, "category": str, "day": <0-based int, optional>} — swap a place for something similar; category is one of culture/scenery/food/shopping/entertainment/adventure/wellness/city
- CHANGE_TRANSPORT    {"new_transport": "DRIVE"|"TRANSIT"|"WALK"|"BICYCLE"}
- CHANGE_SCHEDULE     {"day": <0-based int>, "new_start_time": "HH:MM", "new_end_time": "HH:MM"} — either time optional
- REPLAN_DAY          {"day": <0-based int>, "theme": str optional, "pacing": str optional} — rebuild one day from scratch
- UPDATE_PREFERENCES  {"preferences": [str, ...]}                       — user states new interests
- REGENERATE_TRIP     {}                                                — rebuild the entire trip
- UNDO                {}                                                — revert the last change

Day resolution rules:
- The context lists each itinerary day as index/date/type. "day 1" or "the first day" means index 0, "day 2" means index 1, etc.
- A date like "September 2nd" must be resolved to the matching index using the context dates.

Few-shot examples:
- "Replace Meiji Jingu with another shrine" -> {"action": "REPLACE_PLACE", "params": {"place_name": "Meiji Jingu", "category": "culture"}}
- "Add Tokyo Skytree to day 2" -> {"action": "ADD_PLACE", "params": {"place_name": "Tokyo Skytree", "day": 1}}
- "Remove lunch on the last day" -> {"action": "REMOVE_PLACE", "params": {"place_name": "lunch", "day": <last index>}}
- "Let's drive instead" -> {"action": "CHANGE_TRANSPORT", "params": {"new_transport": "DRIVE"}}
- "Start day 2 at 10am" -> {"action": "CHANGE_SCHEDULE", "params": {"day": 1, "new_start_time": "10:00"}}
- "Day 2 is boring, redo it with more nature and keep it relaxed" -> {"action": "REPLAN_DAY", "params": {"day": 1, "theme": "scenery", "pacing": "relaxed"}}
- "I'm actually really into shopping" -> {"action": "UPDATE_PREFERENCES", "params": {"preferences": ["shopping"]}}
- "What am I doing on day 2?" -> {"action": "QUESTION", "params": {"day": 1}}
- "undo that" -> {"action": "UNDO", "params": {}}

Return JSON ONLY: {"action": "<ACTION>", "params": {...}}"""


def _resolve_day_param(params: dict, context: dict):
    """Coerce params['day'] to a valid 0-based int where possible: ints pass
    through, date strings are matched against the itinerary dates from the
    context. Unresolvable values are left for the handler's precondition
    checks (which reply with a clarification instead of mutating)."""
    if "day" not in params:
        return
    raw = params["day"]
    itinerary = context.get("itinerary", []) if isinstance(context, dict) else []
    if isinstance(raw, bool):
        params.pop("day")
        return
    if isinstance(raw, int):
        return
    if isinstance(raw, str):
        raw = raw.strip()
        for d in itinerary:
            if raw == d.get("date"):
                params["day"] = d.get("day")
                return
        m = re.fullmatch(r"\d+", raw)
        if m:
            params["day"] = int(raw)
            return
    params.pop("day", None)


def classify_intent(message: str, context: dict) -> IntentResult:
    """LLM classification with strict validation: any unknown/empty action
    falls back to QUESTION (fallback=True) so a misfire can never mutate."""
    context = context or {}
    user_prompt = json.dumps({
        "conversation_summary": context.get("summary", ""),
        "recent_messages": context.get("recent_messages", []),
        "itinerary": context.get("itinerary", []),
        "preferences": context.get("preferences", {}),
        "last_user_message": message,
    }, ensure_ascii=False, default=str)

    raw = chat_llm(_CLASSIFIER_SYSTEM, user_prompt)
    if not isinstance(raw, dict):
        raw = {}

    action_str = str(raw.get("action", "")).strip().upper()
    params = raw.get("params")
    if not isinstance(params, dict):
        params = {}

    if action_str not in VALID_ACTIONS:
        if action_str:
            _log_intent.info("Unknown action %r from LLM — falling back to "
                             "QUESTION.", action_str)
        return IntentResult(action=ChatAction.QUESTION, params=params,
                            fallback=True, message=message)

    _resolve_day_param(params, context)
    return IntentResult(action=ChatAction(action_str), params=params,
                        fallback=False, message=message)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Persistence, versioning and chat history (was chat_state.py)
#
# * SessionStore       — versioned rich JSON snapshots: sessions/<sid>/v{N}.json
# * ChatHistoryManager — sessions/<sid>/chat.json with rolling summary
# * Serialization helpers — Place / DayPlan <-> JSON round-trip that preserves
#   everything build_itinerary()'s display JSON drops (types, ids, scores,
#   visit durations, raw opening-hour strings).
#
# No MongoDB. Zero edits to the existing planner files.
# ══════════════════════════════════════════════════════════════════════════════

_log_state = logging.getLogger("chat.state")

# ── Serialization helpers (rich snapshots — never rebuilt from lossy display
#    JSON) ─────────────────────────────────────────────────────────────────────


def place_to_dict(p: Place) -> dict:
    return asdict(p)


def place_from_dict(d: dict) -> Place:
    return Place(**d)


def day_to_snapshot(day: DayPlan) -> dict:
    """Rich per-day snapshot: full Place dumps, ISO datetimes,
    dropped_meals as a sorted list (sets are not JSON-serializable)."""
    return {
        "day_index": day.day_index,
        "date": day.date,
        "day_type": day.day_type,
        "base_location": dict(day.base_location or {}),
        "start_time": day.start_time.isoformat() if day.start_time else None,
        "end_time": day.end_time.isoformat() if day.end_time else None,
        "attractions": [place_to_dict(p) for p in day.attractions],
        "meals": {slot: place_to_dict(p) for slot, p in day.meals.items()},
        "dropped_meals": sorted(day.dropped_meals),
        "capacity_min": day.capacity_min,
        "capacity_max": day.capacity_max,
        "valid": day.valid,
        "violations": list(day.violations),
    }


def day_from_snapshot(snap: dict) -> DayPlan:
    """Reconstruct a live itineraryPlanner.DayPlan (parsed datetimes, live
    Place objects, dropped_meals restored as a set). sequence/schedule are
    left empty — they are regenerated by build_day_sequence on rebuild;
    the rendered display lives in the session's display_itinerary."""
    return DayPlan(
        day_index=snap["day_index"],
        date=snap["date"],
        day_type=snap["day_type"],
        base_location=dict(snap.get("base_location") or {}),
        start_time=datetime.fromisoformat(snap["start_time"]) if snap.get("start_time") else None,
        end_time=datetime.fromisoformat(snap["end_time"]) if snap.get("end_time") else None,
        attractions=[place_from_dict(d) for d in snap.get("attractions", [])],
        meals={slot: place_from_dict(d) for slot, d in snap.get("meals", {}).items()},
        capacity_min=snap.get("capacity_min", 0),
        capacity_max=snap.get("capacity_max", 0),
        valid=snap.get("valid", True),
        violations=list(snap.get("violations", [])),
        dropped_meals=set(snap.get("dropped_meals", [])),
    )


def render_day_json(day: DayPlan) -> dict:
    """Replicate build_itinerary()'s per-day output block for a single day,
    so mutated days re-render identically to the original planner output."""
    return {
        "day": day.day_index, "date": day.date, "type": day.day_type,
        "valid": day.valid,
        "schedule": [
            {
                "time": ip.fmt_time(e["arrival"]),
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
            for e in day.schedule],
        "attractions": [p.name for p in day.attractions],
        "meals": {k: v.name for k, v in day.meals.items()},
    }


# ── Session state container ───────────────────────────────────────────────────

@dataclass
class SessionState:
    cfg: TripConfig
    days: list                      # list[DayPlan] (live objects)
    used_restaurants: set = field(default_factory=set)
    display_itinerary: dict = field(default_factory=dict)
    version: int = 0                # 0 = not yet persisted
    previous_version: int = None


def _bootstrap_place_from_entry(entry: dict, source: str) -> Place:
    """Best-effort Place from one display-JSON schedule entry (demo bootstrap
    path only — display JSON is lossy so types/ids are approximated)."""
    name = entry.get("name", "Unknown")
    if source == "meal" and ":" in name:
        name = name.split(":", 1)[1].strip()
    loc = dict(entry.get("location") or {})
    loc.setdefault("name", name)
    loc.setdefault("address", "")
    return Place(
        id=f"boot-{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}",
        name=name,
        location=loc,
        types=[],
        primary_type="",
        rating=entry.get("rating") or 0.0,
        user_rating_count=0,
        price_level=entry.get("price_level") or "",
        opening_hours=entry.get("opening_hours") or [],
        visit_duration_min=max(entry.get("duration_min") or 60, 60),
        source=source,
    )


def _days_from_output_json(cfg: TripConfig, output_json: dict) -> tuple:
    """Reconstruct best-effort live DayPlans from a plain itinerary_output.json
    (lossy) using classify_days for the temporal windows."""
    classified = ip.classify_days(cfg)
    used_restaurants: set = set()
    days = []
    for cd, od in zip(classified, output_json.get("days", [])):
        for entry in od.get("schedule", []):
            kind = entry.get("kind")
            if kind == "attraction":
                cd.attractions.append(_bootstrap_place_from_entry(entry, "text"))
            elif kind == "meal":
                label = entry.get("name", "").split(":", 1)[0].strip().lower()
                if label in MEAL_SLOTS:
                    place = _bootstrap_place_from_entry(entry, "meal")
                    cd.meals[label] = place
                    used_restaurants.add(place.name.lower())
        days.append(cd)
    return days, used_restaurants


def build_state_from_output(cfg: TripConfig, output_json: dict = None,
                            days: list = None,
                            used_restaurants: set = None) -> SessionState:
    """Build an in-memory SessionState either from live planner objects
    (rich path) or from a plain itinerary_output.json (demo bootstrap)."""
    if not cfg.preferences:
        cfg.preferences = ip.calculate_preference_scores(
            ip.AVAILABLE_PREFERENCES, cfg.selected_preferences)

    if days is None:
        if not output_json:
            raise ValueError("Either live days or an output JSON is required.")
        days, used_restaurants = _days_from_output_json(cfg, output_json)
    elif used_restaurants is None:
        used_restaurants = {m.name.lower() for d in days for m in d.meals.values()}

    if output_json:
        display = json.loads(json.dumps(output_json))  # deep copy
    else:
        display = {"destination": cfg.destination, "start": cfg.start_date,
                   "end": cfg.end_date,
                   "days": [render_day_json(d) for d in days]}

    return SessionState(cfg=cfg, days=days,
                        used_restaurants=set(used_restaurants or set()),
                        display_itinerary=display)


# ── SessionStore — versioned local JSON persistence ───────────────────────────

class SessionStore:
    def __init__(self, root: str = None, session_id: str = "default"):
        self.root = root or DEFAULT_SESSIONS_ROOT
        self.session_id = session_id
        self.session_dir = os.path.join(self.root, session_id)
        os.makedirs(self.session_dir, exist_ok=True)

    # ── version bookkeeping ──────────────────────────────────────────────
    def _version_path(self, n: int) -> str:
        return os.path.join(self.session_dir, f"v{n}.json")

    def list_versions(self) -> list:
        versions = []
        for fname in os.listdir(self.session_dir):
            m = re.fullmatch(r"v(\d+)\.json", fname)
            if m:
                versions.append(int(m.group(1)))
        return sorted(versions)

    def _prune(self):
        versions = self.list_versions()
        while len(versions) > MAX_VERSIONS:
            oldest = versions.pop(0)
            try:
                os.remove(self._version_path(oldest))
            except OSError:
                pass

    # ── save / load ──────────────────────────────────────────────────────
    def save_snapshot(self, state: SessionState) -> int:
        versions = self.list_versions()
        new_version = (versions[-1] + 1) if versions else 1
        snapshot = {
            "version": new_version,
            "previous_version": state.version or None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "trip_config": asdict(state.cfg),
            "days": [day_to_snapshot(d) for d in state.days],
            "used_restaurants": sorted(state.used_restaurants),
            "preferences": dict(state.cfg.preferences),
            "display_itinerary": state.display_itinerary,
        }
        with open(self._version_path(new_version), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
        state.previous_version = state.version or None
        state.version = new_version
        self._prune()
        return new_version

    def _load_version(self, n: int) -> SessionState:
        with open(self._version_path(n), "r", encoding="utf-8") as f:
            snap = json.load(f)
        return SessionState(
            cfg=TripConfig(**snap["trip_config"]),
            days=[day_from_snapshot(d) for d in snap["days"]],
            used_restaurants=set(snap.get("used_restaurants", [])),
            display_itinerary=snap.get("display_itinerary", {}),
            version=snap["version"],
            previous_version=snap.get("previous_version"),
        )

    def _try_load_version(self, n: int) -> SessionState:
        """Like _load_version but returns None (with a warning) when the
        snapshot file is unreadable/corrupt instead of raising."""
        try:
            return self._load_version(n)
        except (OSError, json.JSONDecodeError, KeyError, TypeError,
                ValueError) as e:
            _log_state.warning("Version %s is unreadable/corrupt, skipping: %s",
                               n, e)
            return None

    def load_current(self) -> SessionState:
        """Newest READABLE version: a corrupt head snapshot is skipped (with a
        warning) in favour of the next-newest readable one; None only when no
        version can be read at all."""
        for n in reversed(self.list_versions()):
            state = self._try_load_version(n)
            if state is not None:
                return state
        return None

    def restore_version(self, n: int) -> SessionState:
        """Restore version n by re-saving it as a NEW head version (keeps the
        history linear, so undo-of-undo also works)."""
        if n not in self.list_versions():
            raise ValueError(f"Version {n} does not exist for this session.")
        state = self._try_load_version(n)
        if state is None:
            raise ValueError(f"Version {n} is unreadable/corrupt.")
        versions = self.list_versions()
        state.version = versions[-1]  # so the new snapshot points back at HEAD
        self.save_snapshot(state)
        return state

    # ── bootstrap right after build_itinerary ────────────────────────────
    def create_from_planner_output(self, cfg: TripConfig,
                                   output_json: dict = None,
                                   days: list = None,
                                   used_restaurants: set = None) -> SessionState:
        state = build_state_from_output(cfg, output_json=output_json, days=days,
                                        used_restaurants=used_restaurants)
        self.save_snapshot(state)
        return state


# ── ChatHistoryManager — chat.json + rolling summary + compact context ────────

class ChatHistoryManager:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.path = os.path.join(session_dir, "chat.json")

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"summary": "", "messages": []}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"summary": "", "messages": []}

    def _save(self, data: dict):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_message(self, role: str, content: str):
        data = self._load()
        data["messages"].append({
            "role": role, "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()})
        self._save(data)

    def build_context(self, state: SessionState = None, max_recent: int = 8) -> dict:
        """Recent messages + rolling summary + compact itinerary view
        (day/date + place names only) + preferences."""
        data = self._load()
        ctx = {
            "summary": data.get("summary", ""),
            "recent_messages": [
                {"role": m["role"], "content": m["content"]}
                for m in data.get("messages", [])[-max_recent:]
            ],
        }
        if state is not None:
            ctx["itinerary"] = [
                {
                    "day": d.day_index,
                    "date": d.date,
                    "type": d.day_type,
                    "attractions": [p.name for p in d.attractions],
                    "meals": {slot: p.name for slot, p in d.meals.items()},
                }
                for d in state.days
            ]
            ctx["preferences"] = {
                "selected": list(state.cfg.selected_preferences),
                "scores": dict(state.cfg.preferences),
                "travel_style": state.cfg.travel_style,
                "transport_mode": state.cfg.transport_mode,
            }
        return ctx

    def maybe_summarize(self, llm_fn, threshold: int = 20,
                        keep_recent: int = 8) -> bool:
        """Compress older messages into the rolling summary once the log
        exceeds `threshold`. Degrades gracefully to plain truncation when
        the LLM call fails/returns nothing."""
        data = self._load()
        messages = data.get("messages", [])
        if len(messages) <= threshold:
            return False

        older = messages[:-keep_recent]
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in older)
        summary = ""
        try:
            result = llm_fn(
                "You are a conversation summarizer for a travel-itinerary "
                "chatbot. Compress the transcript into 3-5 sentences keeping "
                "every itinerary change and open request. "
                'Return JSON only: {"summary": "..."}',
                f"Existing summary:\n{data.get('summary', '')}\n\n"
                f"Transcript to fold in:\n{transcript}")
            if isinstance(result, dict):
                summary = str(result.get("summary", "")).strip()
        except Exception:
            summary = ""

        if not summary:
            # graceful fallback: keep a truncated readout of the older turns
            snippets = [f"{m['role']}: {m['content'][:60]}" for m in older[-10:]]
            summary = (data.get("summary", "") + " | " if data.get("summary") else "")
            summary += "Earlier conversation (truncated): " + " / ".join(snippets)
            summary = summary[:2000]

        data["summary"] = summary
        data["messages"] = messages[-keep_recent:]
        self._save(data)
        return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Deterministic action executors (was action_handlers.py)
#
# The LLM never mutates anything: every handler is plain code that reuses the
# planner's own functions (build_day_sequence, validate_day,
# deterministic_repair, search_text, search_nearby, _raw_to_place,
# _score_place, _name_similarity, calculate_preference_scores,
# filter_candidates, build_itinerary, ...).
#
# Handlers take (intent: IntentResult, state: SessionState,
# store: SessionStore) and return a HandlerResult. They mutate only the
# affected day(s); the engine persists a new version whenever
# itinerary_changed is True. When a precondition cannot be resolved (unknown
# place, bad day index) the handler replies with a clarification and performs
# NO mutation.
# ══════════════════════════════════════════════════════════════════════════════

_log_handlers = logging.getLogger("chat.handlers")

# ── Common helpers ────────────────────────────────────────────────────────────


def _rebuild_and_validate(day, cfg, used_restaurants, backups=None) -> bool:
    """build_day_sequence -> validate_day. 
    Auto-repair disabled so the user maintains full manual control."""
    ip.build_day_sequence(day, cfg, used_restaurants)
    ok, _viols = ip.validate_day(day)
    
    # Disable the aggressive auto-repair
    # if not ok:
    #     ip.deterministic_repair(day, cfg, backups or [], used_restaurants)
        
    return day.valid


def _refresh_display(state: SessionState, day):
    """Re-render one mutated day into the stored display itinerary."""
    days = state.display_itinerary.setdefault("days", [])
    rendered = render_day_json(day)
    for i, d in enumerate(days):
        if d.get("day") == day.day_index:
            days[i] = rendered
            return
    days.append(rendered)


def _find_day_and_entry(state: SessionState, place_name: str):
    """Fuzzy, case-insensitive lookup of a named place across all days using
    the planner's _name_similarity. Returns (day_index, kind, ref) where kind
    is "attraction" (ref = Place) or "meal" (ref = slot name), or None when
    nothing matches — the caller must then ask for clarification, NOT mutate.
    """
    if not place_name:
        return None
    query = place_name.strip().lower()

    # Direct meal-slot reference ("remove lunch")
    if query in MEAL_SLOTS:
        for day in state.days:
            if query in day.meals:
                return (day.day_index, "meal", query)
        return None

    best = None
    best_sim = 0.0
    for day in state.days:
        for p in day.attractions:
            sim = ip._name_similarity(place_name, p.name)
            if query == p.name.lower():
                sim = 1.0
            if sim > best_sim:
                best_sim, best = sim, (day.day_index, "attraction", p)
        for slot, p in day.meals.items():
            sim = ip._name_similarity(place_name, p.name)
            if query == p.name.lower():
                sim = 1.0
            if sim > best_sim:
                best_sim, best = sim, (day.day_index, "meal", slot)
    if best is not None and best_sim >= NAME_MATCH_THRESHOLD:
        return best
    return None


def _valid_day_index(state: SessionState, params: dict):
    """Return a valid 0-based day index from params, or None."""
    day = params.get("day")
    if isinstance(day, bool) or not isinstance(day, int):
        return None
    if 0 <= day < len(state.days):
        return day
    return None


def _already_in_itinerary(state: SessionState, name: str) -> bool:
    for day in state.days:
        for p in day.attractions:
            if ip._name_similarity(name, p.name) >= DUPLICATE_THRESHOLD \
                    or name.lower() == p.name.lower():
                return True
    return False


def _compact_itinerary(state: SessionState) -> list:
    display_days = {
        d.get("day"): d for d in state.display_itinerary.get("days", [])
        if isinstance(d, dict)
    }
    itinerary = []
    for day in state.days:
        display_day = display_days.get(day.day_index, {})
        schedule = [
            {
                "time": entry.get("time"),
                "name": entry.get("name"),
                "kind": entry.get("kind"),
            }
            for entry in display_day.get("schedule", [])
            if isinstance(entry, dict)
        ]
        if not schedule:
            schedule = [
                {
                    "time": ip.fmt_time(entry["arrival"]),
                    "name": entry.get("name"),
                    "kind": entry.get("kind"),
                }
                for entry in day.schedule
            ]
        itinerary.append({
            "day": day.day_index,
            "date": day.date,
            "type": day.day_type,
            "schedule": schedule,
            "attractions": [p.name for p in day.attractions],
            "meals": {slot: p.name for slot, p in day.meals.items()},
        })
    return itinerary


def _day_readout(day) -> str:
    parts = [f"On {day.date} (day {day.day_index + 1}, {day.day_type} day) "
             f"you'll visit: "
             + (", ".join(p.name for p in day.attractions) or "no attractions")
             + "."]
    if day.meals:
        parts.append("Meals: " + ", ".join(
            f"{slot} at {p.name}" for slot, p in day.meals.items()) + ".")
    return " ".join(parts)


def _validity_note(day) -> str:
    if day.valid:
        return ""
    return (f" Heads up: day {day.day_index + 1} still has scheduling issues "
            f"({'; '.join(day.violations[:2])}).")


# ── Handlers ──────────────────────────────────────────────────────────────────


def handle_question(intent: IntentResult, state: SessionState,
                    store: SessionStore) -> HandlerResult:
    """No mutation ever. Compact itinerary + question -> chat_llm for a
    natural-language answer; offline fallback = direct day readout."""
    question = intent.message or intent.params.get("question", "")
    answer = ""
    result = chat_llm(
        "You are a helpful travel assistant. Answer the user's question about "
        "their itinerary using ONLY the provided itinerary data. Be concise. "
        'Return JSON only: {"answer": "..."}',
        json.dumps({"itinerary": _compact_itinerary(state),
                    "preferences": list(state.cfg.selected_preferences),
                    "question": question}, ensure_ascii=False))
    if isinstance(result, dict):
        answer = str(result.get("answer", "")).strip()

    if not answer:
        # Offline fallback: deterministic readout.
        day_idx = _valid_day_index(state, intent.params)
        if day_idx is None:
            m = re.search(r"day\s+(\d+)", question, re.I)
            if m and 1 <= int(m.group(1)) <= len(state.days):
                day_idx = int(m.group(1)) - 1
        if day_idx is not None:
            answer = _day_readout(state.days[day_idx])
        else:
            answer = " ".join(_day_readout(d) for d in state.days)
    return HandlerResult(itinerary_changed=False, response_text=answer)


def handle_add_place(intent: IntentResult, state: SessionState,
                     store: SessionStore) -> HandlerResult:
    place_name = str(intent.params.get("place_name", "")).strip()
    if not place_name:
        return HandlerResult(False, "Which place would you like me to add?")
    day_idx = _valid_day_index(state, intent.params)
    if day_idx is None:
        return HandlerResult(
            False, f"Sure — which day should I add {place_name} to? "
                   f"(day 1 to day {len(state.days)})")
    day = state.days[day_idx]
    if _already_in_itinerary(state, place_name):
        return HandlerResult(
            False, f"{place_name} is already in your itinerary, so I left "
                   f"everything unchanged.")
    if len(day.attractions) >= MAX_ATTRACTIONS_PER_DAY:
        return HandlerResult(
            False, f"Day {day_idx + 1} already has {len(day.attractions)} "
                   f"stops (the maximum). Try removing something first.")

    raws = ip.search_text(f"{place_name} in {state.cfg.destination}")
    best_raw, best_sim = None, -1.0
    for raw in raws:
        name = raw.get("displayName", {}).get("text", "")
        sim = ip._name_similarity(place_name, name)
        if sim > best_sim:
            best_sim, best_raw = sim, raw
    place = ip._raw_to_place(best_raw, "text", state.cfg.travel_style) \
        if best_raw else None
    if place is None:
        return HandlerResult(
            False, f"I couldn't find \"{place_name}\" near "
                   f"{state.cfg.destination}. Could you check the name?")

    place.score = ip._score_place(place, state.cfg.preferences,
                                  state.cfg.hotel, state.cfg.budget)
    day.attractions.append(place)
    _rebuild_and_validate(day, state.cfg, state.used_restaurants)
    if not any(p.id == place.id for p in day.attractions):
        # Repair had to sacrifice the new place — the day simply can't fit it.
        _refresh_display(state, day)
        return HandlerResult(
            True, f"I tried adding {place.name} to day {day_idx + 1}, but the "
                  f"schedule couldn't fit it and it was dropped again during "
                  f"repair. Consider removing another stop first.",
            [day_idx])
    _refresh_display(state, day)
    return HandlerResult(
        True, f"Done — {place.name} is now on day {day_idx + 1} ({day.date}) "
              f"and the day's schedule has been rebuilt."
              + _validity_note(day),
        [day_idx])


def handle_remove_place(intent: IntentResult, state: SessionState,
                        store: SessionStore) -> HandlerResult:
    place_name = str(intent.params.get("place_name", "")).strip()
    if not place_name:
        return HandlerResult(False, "Which place should I remove?")
    hit = _find_day_and_entry(state, place_name)
    if hit is None:
        return HandlerResult(
            False, f"I couldn't find \"{place_name}\" in your itinerary. "
                   f"Could you tell me the exact place (or meal) name?")
    day_idx, kind, ref = hit
    # If the user specified a day, respect it when it disagrees with the match.
    wanted_day = _valid_day_index(state, intent.params)
    if wanted_day is not None and wanted_day != day_idx:
        alt = None
        target = state.days[wanted_day]
        q = place_name.lower()
        if q in MEAL_SLOTS and q in target.meals:
            alt = (wanted_day, "meal", q)
        else:
            for p in target.attractions:
                if ip._name_similarity(place_name, p.name) >= NAME_MATCH_THRESHOLD:
                    alt = (wanted_day, "attraction", p)
                    break
            for slot, p in target.meals.items():
                if ip._name_similarity(place_name, p.name) >= NAME_MATCH_THRESHOLD:
                    alt = (wanted_day, "meal", slot)
                    break
        if alt is None:
            return HandlerResult(
                False, f"I couldn't find \"{place_name}\" on day "
                       f"{wanted_day + 1} — it looks like it's on day "
                       f"{day_idx + 1} instead. Which one did you mean?")
        day_idx, kind, ref = alt

    day = state.days[day_idx]
    if kind == "meal":
        slot = ref
        place = day.meals.pop(slot, None)
        if place is not None:
            state.used_restaurants.discard(place.name.lower())
        day.dropped_meals.add(slot)
        removed = f"{slot} ({place.name if place else 'unknown venue'})"
    else:
        day.attractions.remove(ref)
        removed = ref.name
    _rebuild_and_validate(day, state.cfg, state.used_restaurants)
    _refresh_display(state, day)
    return HandlerResult(
        True, f"Removed {removed} from day {day_idx + 1} ({day.date}) and "
              f"rebuilt the day's schedule." + _validity_note(day),
        [day_idx])


def handle_replace_place(intent: IntentResult, state: SessionState,
                         store: SessionStore) -> HandlerResult:
    place_name = str(intent.params.get("place_name", "")).strip()
    if not place_name:
        return HandlerResult(False, "Which place should I replace?")
    hit = _find_day_and_entry(state, place_name)
    if hit is None:
        return HandlerResult(
            False, f"I couldn't find \"{place_name}\" in your itinerary, so "
                   f"nothing was changed. What's the exact name?")
    day_idx, kind, ref = hit
    day = state.days[day_idx]
    cfg = state.cfg

    if kind == "meal":
        # Replacing a meal = free the venue and let the rebuild pick another
        # (the slot is NOT added to dropped_meals so it gets refilled).
        slot = ref
        old = day.meals.pop(slot, None)
        if old is not None:
            state.used_restaurants.add(old.name.lower())  # keep it banned
        _rebuild_and_validate(day, cfg, state.used_restaurants)
        _refresh_display(state, day)
        new = day.meals.get(slot)
        return HandlerResult(
            True, f"Swapped your {slot} on day {day_idx + 1} to "
                  f"{new.name if new else 'a new venue (none found nearby)'}"
                  + _validity_note(day),
            [day_idx])

    category = str(intent.params.get("category", "")).strip().lower()
    queries = []
    if category:
        queries.append(f"best {category} attractions in {cfg.destination}")
    queries.append(f"attractions similar to {place_name} in {cfg.destination}")

    seen_ids = set()
    candidates = []
    for q in queries:
        for raw in ip.search_text(q):
            p = ip._raw_to_place(raw, "text", cfg.travel_style)
            if p is None or p.id in seen_ids:
                continue
            if p.primary_type in BANNED_PRIMARY_TYPES:
                continue
            if _already_in_itinerary(state, p.name):
                continue  # exclude names already in the itinerary
            if ip._name_similarity(p.name, ref.name) >= DUPLICATE_THRESHOLD:
                continue  # don't replace a place with itself
            seen_ids.add(p.id)
            candidates.append(p)
    if category in ip.THEME_TO_TYPES:
        for raw in ip.search_nearby(cfg.hotel["latitude"], cfg.hotel["longitude"],
                                    included_types=ip.THEME_TO_TYPES[category][:5]):
            p = ip._raw_to_place(raw, "nearby", cfg.travel_style)
            if p is None or p.id in seen_ids:
                continue
            if p.primary_type in BANNED_PRIMARY_TYPES:
                continue
            if _already_in_itinerary(state, p.name):
                continue
            if ip._name_similarity(p.name, ref.name) >= DUPLICATE_THRESHOLD:
                continue
            seen_ids.add(p.id)
            candidates.append(p)

    if not candidates:
        return HandlerResult(
            False, f"I couldn't find a good replacement for {ref.name}, so I "
                   f"left it in place.")

    for p in candidates:
        p.score = ip._score_place(p, cfg.preferences, cfg.hotel, cfg.budget)
    best = max(candidates, key=lambda p: p.score)

    day.attractions.remove(ref)
    day.attractions.append(best)
    _rebuild_and_validate(day, cfg, state.used_restaurants)
    _refresh_display(state, day)
    return HandlerResult(
        True, f"Replaced {ref.name} with {best.name} on day {day_idx + 1} "
              f"({day.date}) and rebuilt the schedule." + _validity_note(day),
        [day_idx])


# def handle_change_transport(intent: IntentResult, state: SessionState,
#                             store: SessionStore) -> HandlerResult:
#     mode = str(intent.params.get("new_transport")
#                or intent.params.get("transport")
#                or intent.params.get("mode") or "").strip().upper()
#     if mode not in VALID_TRANSPORT_MODES:
#         return HandlerResult(
#             False, f"I can switch between DRIVE, TRANSIT, WALK or BICYCLE — "
#                    f"\"{mode or '?'}\" isn't one I support.")
#     if mode == state.cfg.transport_mode:
#         return HandlerResult(
#             False, f"You're already using {mode} — nothing to change.")
#     state.cfg.transport_mode = mode
#     updated = []
#     for day in state.days:
#         _rebuild_and_validate(day, state.cfg, state.used_restaurants)
#         _refresh_display(state, day)
#         updated.append(day.day_index)
#     return HandlerResult(
#         True, f"Switched your transport mode to {mode} and rebuilt all "
#               f"{len(updated)} days' routes and schedules.",
#         updated)


def handle_change_schedule(intent: IntentResult, state: SessionState,
                           store: SessionStore) -> HandlerResult:
    day_idx = _valid_day_index(state, intent.params)
    if day_idx is None:
        return HandlerResult(
            False, f"Which day's schedule should I change? "
                   f"(day 1 to day {len(state.days)})")
    day = state.days[day_idx]

    def _parse_hhmm(val):
        if not val:
            return None
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(val).strip())
        if not m:
            return None
        hh, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        base = datetime.strptime(day.date, "%Y-%m-%d")
        return base.replace(hour=hh, minute=mm)

    new_start = _parse_hhmm(intent.params.get("new_start_time"))
    new_end = _parse_hhmm(intent.params.get("new_end_time"))
    if new_start is None and new_end is None:
        return HandlerResult(
            False, "What time should the day start or end? Please give me a "
                   "time like 10:00.")

    start = new_start or day.start_time
    end = new_end or day.end_time
    if end <= start:  # guard against an inverted window — no mutation
        return HandlerResult(
            False, f"That would make day {day_idx + 1} end "
                   f"({end.strftime('%H:%M')}) before it starts "
                   f"({start.strftime('%H:%M')}), so I left it unchanged.")

    day.start_time = start
    day.end_time = end
    _rebuild_and_validate(day, state.cfg, state.used_restaurants)
    _refresh_display(state, day)
    return HandlerResult(
        True, f"Day {day_idx + 1} ({day.date}) now runs from "
              f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')} — the "
              f"schedule has been rebuilt." + _validity_note(day),
        [day_idx])


def handle_replan_day(intent: IntentResult, state: SessionState,
                      store: SessionStore) -> HandlerResult:
    day_idx = _valid_day_index(state, intent.params)
    if day_idx is None:
        return HandlerResult(
            False, f"Which day should I replan? (day 1 to day "
                   f"{len(state.days)})")
    day = state.days[day_idx]
    cfg = state.cfg

    theme = str(intent.params.get("theme")
                or intent.params.get("category") or "").strip().lower()
    theme = PREFERENCE_SYNONYMS.get(theme, theme)
    if theme in ip.AVAILABLE_PREFERENCES:
        themes = [theme]
    else:
        themes = sorted(cfg.preferences, key=cfg.preferences.get,
                        reverse=True)[:3]

    other_day_places = [p for d in state.days if d.day_index != day_idx
                        for p in d.attractions]
    other_ids = {p.id for p in other_day_places}

    seen_ids = set()
    pool = []
    for t in themes:
        for raw in ip.search_text(f"{t} attractions in {cfg.destination}",
                                  max_results=10):
            p = ip._raw_to_place(raw, "text", cfg.travel_style)
            if p is None or p.id in seen_ids or p.id in other_ids:
                continue
            if p.primary_type in BANNED_PRIMARY_TYPES:
                continue
            # exclude anything already used on OTHER days (keep them intact)
            if any(ip._name_similarity(p.name, o.name) >= DUPLICATE_THRESHOLD
                   for o in other_day_places):
                continue
            seen_ids.add(p.id)
            pool.append(p)

    if not pool:
        return HandlerResult(
            False, f"I couldn't find fresh candidates to replan day "
                   f"{day_idx + 1}, so it stays as-is.")

    candidates = ip.filter_candidates(pool, cfg, max_candidates=20)
    target = day.capacity_max or max(len(day.attractions), 1)
    pacing = str(intent.params.get("pacing", "")).strip().lower()
    if any(w in pacing for w in ("relax", "slow", "chill", "easy", "fewer", "light")):
        target = max(1, target - 2)
    target = max(1, min(target, len(candidates)))

    day.attractions = candidates[:target]
    day.dropped_meals.clear()  # meals get reconsidered for the fresh day
    _rebuild_and_validate(day, cfg, state.used_restaurants,
                          backups=candidates[target:])
    _refresh_display(state, day)
    names = ", ".join(p.name for p in day.attractions)
    return HandlerResult(
        True, f"Replanned day {day_idx + 1} ({day.date}) from scratch — new "
              f"line-up: {names}. Other days were left untouched."
              + _validity_note(day),
        [day_idx])


# def handle_update_preferences(intent: IntentResult, state: SessionState,
#                               store: SessionStore) -> HandlerResult:
#     raw_prefs = intent.params.get("preferences", [])
#     if isinstance(raw_prefs, str):
#         raw_prefs = [raw_prefs]
#     if not isinstance(raw_prefs, list):
#         raw_prefs = []

#     cfg = state.cfg
#     added, unknown = [], []
#     for pref in raw_prefs:
#         key = PREFERENCE_SYNONYMS.get(str(pref).strip().lower(),
#                                       str(pref).strip().lower())
#         if key in ip.AVAILABLE_PREFERENCES:
#             if key not in cfg.selected_preferences:
#                 cfg.selected_preferences.append(key)
#                 added.append(key)
#         else:
#             unknown.append(str(pref))

#     if not added:
#         known = ", ".join(ip.AVAILABLE_PREFERENCES)
#         return HandlerResult(
#             False, f"I couldn't map {unknown or 'that'} to a known interest. "
#                    f"I understand: {known}.")

#     # Recompute weights and re-score every attraction deterministically.
#     cfg.preferences = ip.calculate_preference_scores(
#         ip.AVAILABLE_PREFERENCES, cfg.selected_preferences)
#     for day in state.days:
#         for p in day.attractions:
#             p.score = ip._score_place(p, cfg.preferences, cfg.hotel, cfg.budget)

#     note = f" (I couldn't map: {', '.join(unknown)}.)" if unknown else ""
#     return HandlerResult(
#         True, f"Added {', '.join(added)} to your interests and re-scored the "
#               f"itinerary.{note} The plan itself wasn't changed — say "
#               f"something like \"replan day 2\" if you'd like a day rebuilt "
#               f"around the new interests.",
#         [])


def handle_regenerate_trip(intent: IntentResult, state: SessionState,
                           store: SessionStore) -> HandlerResult:
    out = ip.build_itinerary(state.cfg)
    if not isinstance(out, dict) or out.get("error") or not out.get("days"):
        return HandlerResult(
            False, "I couldn't regenerate the trip (the planner returned no "
                   "days), so your current itinerary is untouched.")
    fresh = build_state_from_output(state.cfg, output_json=out)
    state.days = fresh.days
    state.used_restaurants = fresh.used_restaurants
    state.display_itinerary = fresh.display_itinerary
    return HandlerResult(
        True, f"Regenerated the whole trip to {state.cfg.destination} "
              f"({len(state.days)} days) from scratch.",
        [d.day_index for d in state.days])


def handle_undo(intent: IntentResult, state: SessionState,
                store: SessionStore) -> HandlerResult:
    if not state.previous_version:
        return HandlerResult(
            False, "There's nothing to undo yet — this is the earliest "
                   "version of your itinerary.")
    target = state.previous_version
    try:
        restored = store.restore_version(target)
    except (ValueError, FileNotFoundError):
        return HandlerResult(
            False, f"Version {target} is no longer available (old versions "
                   f"are pruned), so I couldn't undo.")
    # Adopt the restored state in place so the engine keeps its reference.
    state.cfg = restored.cfg
    state.days = restored.days
    state.used_restaurants = restored.used_restaurants
    state.display_itinerary = restored.display_itinerary
    state.version = restored.version
    state.previous_version = restored.previous_version
    return HandlerResult(
        False,  # restore_version already persisted the new head version
        f"Undone — restored your itinerary to version {target} (saved as "
        f"version {state.version}).",
        [d.day_index for d in state.days])


ACTION_DISPATCH = {
    ChatAction.QUESTION: handle_question,
    ChatAction.ADD_PLACE: handle_add_place,
    ChatAction.REMOVE_PLACE: handle_remove_place,
    ChatAction.REPLACE_PLACE: handle_replace_place,
    #ChatAction.CHANGE_TRANSPORT: handle_change_transport,
    ChatAction.CHANGE_SCHEDULE: handle_change_schedule,
    ChatAction.REPLAN_DAY: handle_replan_day,
    #ChatAction.UPDATE_PREFERENCES: handle_update_preferences,
    ChatAction.REGENERATE_TRIP: handle_regenerate_trip,
    ChatAction.UNDO: handle_undo,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Orchestrator (was chat_engine.py)
#
# Flow per message:
#   add user msg -> build context -> classify intent (LLM) -> dispatch handler
#   (deterministic) -> on mutation save a new version and mention it in the
#   reply -> add assistant msg -> return the response text.
#
# Handler execution is wrapped in try/except: on any exception the in-memory
# state is discarded (reloaded from the last saved version) and the user gets
# an apologetic reply — a failed action can never corrupt the session.
# ══════════════════════════════════════════════════════════════════════════════

_log_engine = logging.getLogger("chat.engine")


class ChatEngine:
    def __init__(self, session_id: str, store_root: str = None):
        # store_root=None -> DEFAULT_SESSIONS_ROOT (module-anchored,
        # CWD-independent); tests still override with explicit temp dirs.
        self.session_id = session_id
        self.store = SessionStore(root=store_root, session_id=session_id)
        self.history = ChatHistoryManager(self.store.session_dir)
        self.state = self.store.load_current()  # None until bootstrapped

    # ── bootstrap ────────────────────────────────────────────────────────
    def start_session_from_output(self, output_json_path_or_dict,
                                  trip_config_kwargs: dict = None,
                                  cfg: TripConfig = None,
                                  days: list = None,
                                  used_restaurants: set = None):
        """Bootstrap a session right after build_itinerary(): either from the
        live planner objects (cfg + days) or from a plain
        itinerary_output.json path/dict + TripConfig kwargs (demo use)."""
        output_json = output_json_path_or_dict
        if isinstance(output_json, (str, os.PathLike)):
            with open(output_json, "r", encoding="utf-8") as f:
                output_json = json.load(f)
        if cfg is None:
            if not trip_config_kwargs:
                raise ValueError(
                    "Provide either a TripConfig or trip_config_kwargs — the "
                    "planner's TripConfig has no field defaults.")
            cfg = TripConfig(**trip_config_kwargs)
        self.state = self.store.create_from_planner_output(
            cfg, output_json=output_json, days=days,
            used_restaurants=used_restaurants)
        return self.state

    # ── main loop ────────────────────────────────────────────────────────
    def process_message(self, text: str) -> str:
        if self.state is None:
            return ("No itinerary session is loaded yet — bootstrap one with "
                    "start_session_from_output() first.")

        self.history.add_message("user", text)
        context = self.history.build_context(self.state)
        intent = classify_intent(text, context)
        intent.message = text
        handler = ACTION_DISPATCH.get(intent.action, handle_question)

        try:
            result: HandlerResult = handler(intent, self.state, self.store)
        except Exception as e:
            _log_engine.exception("Handler %s failed: %s", intent.action, e)
            # Discard any partial in-memory mutation: reload the saved state.
            reloaded = self.store.load_current()
            if reloaded is not None:
                self.state = reloaded
            result = HandlerResult(
                itinerary_changed=False,
                response_text="Sorry — something went wrong while applying "
                              "that change, so I left your itinerary exactly "
                              "as it was. Could you try rephrasing?")

        response = result.response_text
        if result.itinerary_changed:
            version = self.store.save_snapshot(self.state)
            response += (f"\n[Itinerary updated — saved as version {version}."
                         f" Say \"undo\" to revert.]")

        self.history.add_message("assistant", response)
        self.history.maybe_summarize(chat_llm, threshold=20)
        return response

