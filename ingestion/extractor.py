import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

EXTRACTION_PROMPT = """
You are a structured health data extraction system.

Extract health data from the user's input and return ONLY valid JSON. No explanation, no markdown, no extra text.

Schema:
{
  "event_type": "symptom | activity | food | medication | sleep | mood | other",
  "symptom": string or null,
  "body_part": string or null,
  "body_parts": list[string] or null,
  "laterality": "left | right | bilateral | null",
  "trigger": string or null,
  "time_reference": integer or null,
  "severity_band": "Low | Moderate | High | null",
  "impact_class": "C1 | C2 | C3 | C4 | C5",
  "cyclic_candidate": true | false
}

--- FIELD RULES ---

event_type:
  symptom   - a physical or mental symptom (pain, nausea, anxiety, fatigue)
  activity  - exercise or physical action (running, gym, yoga, walk)
  food      - eating, drinking, consumption (meal, coffee, alcohol)
  medication - taking a drug, supplement, or treatment
  sleep     - sleep duration, quality, or pattern
  mood      - emotional or mental state when not a clinical symptom
  other     - anything that doesn't fit above

symptom: the clinical concept. Use concise SNOMED-style terms. e.g. "pain", "nausea", "fatigue", "palpitation", "insomnia".
body_part: primary anatomical location in standard terms. e.g. "knee", "abdomen", "chest".
body_parts: all body parts mentioned, as a list.
laterality: "left", "right", "bilateral" only. null if not mentioned.
trigger: what caused or preceded the event. e.g. "running", "coffee", "sushi", "stress".

time_reference (integer day offset from today):
  today / this morning / just now → 0
  yesterday / last night / a day ago → -1
  two days ago → -2
  last week / a week ago → -7
  two weeks ago → -14
  last month → -30
  last year → -365
  unknown / vague → null

severity_band (how notable or impactful the event was):
  Low      - mentioned casually, mild, not distressing ("had a coffee", "slept okay")
  Moderate - clearly notable, affects the day, user emphasizes it ("bad headache", "barely slept")
  High     - significant distress, limits function, alarming language ("heart racing", "can't move", "worst pain")
  null     - cannot be determined

impact_class (how long this type of event affects health - based on the NATURE of the event, not severity):
  C1 - Transient: single meals, one coffee, one night of poor sleep, a minor mood. Gone in hours to 3 days.
  C2 - Short-term: accumulated sleep debt, a cold or flu, a stressful week, consistent poor eating. Lasts 3-12 days.
  C3 - Acute: physical injury, infection requiring medication, post-surgery recovery in progress. Lasts weeks.
  C4 - Persistent: a symptom the user has experienced MULTIPLE TIMES ("second this month", "keeps happening", "third time"), long rehab, ongoing mental health pattern. Lasts months.
  C5 - Chronic: diagnosed condition, known allergy, long-term medication, past surgery. Permanent.

cyclic_candidate: true if the language implies this event happens on a regular schedule.
  true  - ANY of these signals: "usual", "every [day/week/Sunday...]", "always", "daily", "weekly",
           "as usual", "my regular", "my morning X", "again this week", "my routine",
           recurring symptom phrases like "second/third one this month", "same as always",
           long-term medications still being taken ("still on it", "every day since").
  false - one-off, first-time, or no scheduling language present.
  IMPORTANT: activity events like runs, gym sessions, walks described with scheduling words
             ("Sunday run", "my usual run", "weekly tennis") → cyclic_candidate: true.

--- RULES ---
- Be conservative. If unsure, return null.
- Do NOT guess body parts or causes not mentioned.
- impact_class is based on event TYPE, not severity. A severe headache is still C1 unless it's a known chronic condition.
- Never assign C6. Do not output C6.
- severity_band is about how bad this specific instance was, independent of impact_class.
- Keep symptom and body_part short and canonical (1-3 words).

--- EXAMPLES ---

Input: "My left knee hurts after running yesterday"
Output:
{
  "event_type": "symptom",
  "symptom": "pain",
  "body_part": "knee",
  "body_parts": ["knee"],
  "laterality": "left",
  "trigger": "running",
  "time_reference": -1,
  "severity_band": "Moderate",
  "impact_class": "C1",
  "cyclic_candidate": false
}

Input: "Had way too much coffee trying to finish work, and now even though I'm exhausted I feel jittery and my heart's kind of racing."
Output:
{
  "event_type": "symptom",
  "symptom": "palpitation",
  "body_part": "heart",
  "body_parts": ["heart"],
  "laterality": null,
  "trigger": "coffee",
  "time_reference": 0,
  "severity_band": "High",
  "impact_class": "C1",
  "cyclic_candidate": false
}

Input: "Slept maybe 5 hours, kept waking up. Feel pretty groggy."
Output:
{
  "event_type": "sleep",
  "symptom": "insomnia",
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "Moderate",
  "impact_class": "C1",
  "cyclic_candidate": false
}

Input: "Third bad night in a row. Exhausted all week, can't focus at work."
Output:
{
  "event_type": "sleep",
  "symptom": "fatigue",
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "High",
  "impact_class": "C2",
  "cyclic_candidate": false
}

Input: "Did my usual Sunday tennis, felt good."
Output:
{
  "event_type": "activity",
  "symptom": null,
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "Low",
  "impact_class": "C1",
  "cyclic_candidate": true
}

Input: "I have a peanut allergy and accidentally had something with peanuts in it at lunch."
Output:
{
  "event_type": "symptom",
  "symptom": "allergic reaction",
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": "peanuts",
  "time_reference": 0,
  "severity_band": "High",
  "impact_class": "C5",
  "cyclic_candidate": false
}

Input: "Feeling pretty anxious today, not sure why."
Output:
{
  "event_type": "mood",
  "symptom": "anxiety",
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "Moderate",
  "impact_class": "C1",
  "cyclic_candidate": false
}

Input: "Started taking metformin for my diabetes last year, still on it."
Output:
{
  "event_type": "medication",
  "symptom": null,
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": -365,
  "severity_band": null,
  "impact_class": "C5",
  "cyclic_candidate": true
}

Input: "Twisted my ankle pretty badly at football practice, it's quite swollen."
Output:
{
  "event_type": "symptom",
  "symptom": "sprain",
  "body_part": "ankle",
  "body_parts": ["ankle"],
  "laterality": null,
  "trigger": "football",
  "time_reference": 0,
  "severity_band": "High",
  "impact_class": "C3",
  "cyclic_candidate": false
}

Input: "Had a salad for lunch, nothing special."
Output:
{
  "event_type": "food",
  "symptom": null,
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "Low",
  "impact_class": "C1",
  "cyclic_candidate": false
}

Input: "Bad migraine again, second one this month. Same as always - nausea and can't stand light."
Output:
{
  "event_type": "symptom",
  "symptom": "migraine",
  "body_part": "head",
  "body_parts": ["head"],
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "High",
  "impact_class": "C4",
  "cyclic_candidate": true
}

Input: "Did my usual Tuesday gym session."
Output:
{
  "event_type": "activity",
  "symptom": null,
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "Low",
  "impact_class": "C1",
  "cyclic_candidate": true
}

Input: "Morning coffee as usual."
Output:
{
  "event_type": "food",
  "symptom": null,
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": "coffee",
  "time_reference": 0,
  "severity_band": "Low",
  "impact_class": "C1",
  "cyclic_candidate": true
}

Input: "Sunday run done. Easy 5km, felt strong."
Output:
{
  "event_type": "activity",
  "symptom": null,
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": null,
  "time_reference": 0,
  "severity_band": "Low",
  "impact_class": "C1",
  "cyclic_candidate": true
}
"""



def _extract_first_json_object(text: str) -> str:
    """
    Extracts the first complete {...} JSON object from a string.
    Handles cases where the model returns multiple JSON objects,
    or wraps the object in prose before/after it.
    Returns the original text unchanged if no object boundary is found.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return text  # fallback: return as-is, let json.loads surface the error


def extract_health_data(user_input: str) -> dict:
    response = client.chat.completions.create(
        model="gemma-4-e2b",  # or whatever your LM Studio model name is
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1  # keep low to reduce hallucination
    )

    text = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps output in them
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Extract the first complete JSON object from the response.
    # Some models output multiple objects or wrap output in extra text.
    text = _extract_first_json_object(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON. Raw output:")
        print(text)
        return None


EXPECTED_KEYS = [
    "event_type",
    "symptom",
    "body_part",
    "body_parts",
    "laterality",
    "trigger",
    "time_reference",
    "severity_band",
    "impact_class",
    "cyclic_candidate",
]

VALID_EVENT_TYPES = {"symptom", "activity", "food", "medication", "sleep", "mood", "other"}

VALID_SEVERITY_BANDS = {"Low", "Moderate", "High"}

VALID_IMPACT_CLASSES = {"C1", "C2", "C3", "C4", "C5"}

LATERALITY_VALUES = {"left", "right", "bilateral"}


def _clean_string(value):
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    if value in {"", "none", "null", "unknown", "n/a"}:
        return None
    return value


def _clean_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "none", "null", "unknown", "n/a"}:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _clean_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "1"}:
            return True
        if value.strip().lower() in {"false", "no", "0"}:
            return False
    return False


def _clean_severity_band(value):
    """Normalize severity_band to exactly Low / Moderate / High or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    # Capitalize first letter to handle "low", "LOW", "Low" etc.
    normalized = value.strip().capitalize()
    if normalized in VALID_SEVERITY_BANDS:
        return normalized
    return None


def _clean_impact_class(value):
    """Normalize impact_class to C1–C5 or fall back to C1."""
    if value is None:
        return "C1"
    if not isinstance(value, str):
        return "C1"
    normalized = value.strip().upper()
    if normalized in VALID_IMPACT_CLASSES:
        return normalized
    # If the model somehow outputs C6, clamp to C5
    return "C1"


def _split_body_parts(value):
    if value is None:
        return []
    if isinstance(value, list):
        parts = value
    else:
        text = _clean_string(value)
        if text is None:
            return []
        for separator in [",", "/", "&"]:
            text = text.replace(separator, " and ")
        parts = text.split(" and ")

    clean_parts = []
    for part in parts:
        part = _clean_string(part)
        if part and part not in clean_parts:
            clean_parts.append(part)
    return clean_parts


def sanitize_output(data: dict) -> dict:
    if data is None:
        return None

    clean = {}
    for key in EXPECTED_KEYS:
        clean[key] = data.get(key, None)

    # event_type
    raw_event_type = _clean_string(clean["event_type"])
    clean["event_type"] = raw_event_type if raw_event_type in VALID_EVENT_TYPES else "other"

    # string fields
    clean["symptom"] = _clean_string(clean["symptom"])
    clean["body_part"] = _clean_string(clean["body_part"])
    clean["laterality"] = _clean_string(clean["laterality"])
    if clean["laterality"] not in LATERALITY_VALUES:
        clean["laterality"] = None
    clean["trigger"] = _clean_string(clean["trigger"])

    # time
    clean["time_reference"] = _clean_int(clean["time_reference"])

    # body parts
    body_parts = _split_body_parts(clean.get("body_parts"))
    if not body_parts:
        body_parts = _split_body_parts(clean.get("body_part"))
    clean["body_parts"] = body_parts
    clean["body_part"] = body_parts[0] if body_parts else None

    # new fields
    clean["severity_band"] = _clean_severity_band(clean["severity_band"])
    clean["impact_class"] = _clean_impact_class(clean["impact_class"])
    clean["cyclic_candidate"] = _clean_bool(clean.get("cyclic_candidate", False))

    return clean