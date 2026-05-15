import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

EXTRACTION_PROMPT = """
Extract health data from user input into valid JSON. No prose.

Schema:
{
  "event_type": "symptom | activity | food | medication | sleep | mood | other",
  "symptom": string (clinical term),
  "body_parts": list[string],
  "laterality": "left | right | bilateral | null",
  "trigger": string (cause/precursor),
  "time_reference": integer (day offset from today: today=0, yesterday=-1),
  "severity_band": "Low | Moderate | High | null",
  "impact_class": "C1 (Transient: <3d) | C2 (Short: <2w) | C3 (Acute: injury/rehab) | C4 (Recurring/Persistent) | C5 (Chronic/Diagnosed)",
  "cyclic_candidate": boolean (true if routine, usual, daily, weekly, or recurring)
}

Rules:
- severity_band: High = alarming/limits function; Moderate = notable/distressing; Low = casual/mild.
- impact_class: based on nature, not severity. C4 if multiple occurrences mentioned.
- cyclic_candidate: true if using "usual", "every day", "again", "routine".

Examples:
Input: "My left knee hurts after running yesterday"
{"event_type": "symptom", "symptom": "pain", "body_parts": ["knee"], "laterality": "left", "trigger": "running", "time_reference": -1, "severity_band": "Moderate", "impact_class": "C1", "cyclic_candidate": false}

Input: "Third bad night in a row. Exhausted all week."
{"event_type": "sleep", "symptom": "fatigue", "body_parts": [], "laterality": null, "trigger": null, "time_reference": 0, "severity_band": "High", "impact_class": "C2", "cyclic_candidate": false}

Input: "Did my usual Sunday tennis session."
{"event_type": "activity", "symptom": null, "body_parts": [], "laterality": null, "trigger": null, "time_reference": 0, "severity_band": "Low", "impact_class": "C1", "cyclic_candidate": true}

Input: "Taking Lisinopril for my hypertension daily since last year."
{"event_type": "medication", "symptom": null, "body_parts": [], "laterality": null, "trigger": null, "time_reference": -365, "severity_band": null, "impact_class": "C5", "cyclic_candidate": true}

Input: "Had ice cream, feeling really bloated and gassy now."
{"event_type": "food", "symptom": "bloating", "body_parts": ["abdomen"], "laterality": null, "trigger": "ice cream", "time_reference": 0, "severity_band": "Moderate", "impact_class": "C1", "cyclic_candidate": false}

Input: "Migraine again, third this month. Nausea as always."
{"event_type": "symptom", "symptom": "migraine", "body_parts": ["head"], "laterality": null, "trigger": null, "time_reference": 0, "severity_band": "High", "impact_class": "C4", "cyclic_candidate": true}
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