import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

EXTRACTION_PROMPT = """
You are a medical data extraction system.

Extract structured health data from the user's input.

Return ONLY valid JSON. No explanation.

Schema:
{
  "event_type": "symptom | activity | food | medication | other",
  "symptom": string or null,
  "body_part": string or null,
  "body_parts": list[string] or null,
  "laterality": "left | right | bilateral | null",
  "trigger": string or null,
  "time_reference": integer or null,
  "severity": number (1-10) or null
}

Rules:
- Be conservative. If unsure, return null.
- Do NOT guess body parts or causes.
- Standardize symptoms and clinical concepts using SNOMED CT-style clinical terms when possible.
- Standardize body parts to concise anatomical terms.
- Keep outputs short and canonical.
- Split multiple body parts into body_parts.
- time_reference is an approximate integer day offset from today:
  - today, now, this morning → 0
  - yesterday, a day ago → -1
  - tomorrow → 1
  - a week ago, last week → -7
  - two weeks ago → -14
  - last year → -365
  - unknown or vague → null

Examples:

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
  "severity": null
}

Input: "I had sushi and now I feel nauseous"
Output:
{
  "event_type": "symptom",
  "symptom": "nausea",
  "body_part": null,
  "body_parts": null,
  "laterality": null,
  "trigger": "sushi",
  "time_reference": 0,
  "severity": null
}
"""

def extract_health_data(user_input: str) -> dict:
    response = client.chat.completions.create(
        model="gemma-4-e2b",  # or whatever your LM Studio model name is
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1  # VERY IMPORTANT (reduce hallucination)
    )

    text = response.choices[0].message.content.strip()

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
    "severity"
]

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

    clean["event_type"] = _clean_string(clean["event_type"]) or "other"
    clean["symptom"] = _clean_string(clean["symptom"])
    clean["body_part"] = _clean_string(clean["body_part"])
    clean["laterality"] = _clean_string(clean["laterality"])
    clean["trigger"] = _clean_string(clean["trigger"])
    clean["time_reference"] = _clean_int(clean["time_reference"])

    body_parts = _split_body_parts(clean.get("body_parts"))
    if not body_parts:
        body_parts = _split_body_parts(clean.get("body_part"))

    clean["body_parts"] = body_parts
    clean["body_part"] = body_parts[0] if body_parts else None

    return clean
