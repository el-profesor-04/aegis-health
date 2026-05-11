import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
from openai import OpenAI


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_EMBEDDING_THRESHOLD") != "1",
    reason="Opt-in integration test. Set RUN_LLM_EMBEDDING_THRESHOLD=1 and run with LM Studio available.",
)


BASE_URL = os.getenv("AEGIS_LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("AEGIS_LMSTUDIO_API_KEY", "lm-studio")
CHAT_MODEL = os.getenv("AEGIS_CHAT_MODEL", "gemma-4-e2b")
EMBEDDING_MODEL = os.getenv("AEGIS_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


TEST_PAIRS_LABELED = [
    # Clear positives
    ("knee pain", "pain in knee", 1),
    ("my knee hurts", "knee pain", 1),
    ("sharp pain in lower back", "lower back pain", 1),
    ("headache", "pain in head", 1),
    ("stomach ache", "abdominal pain", 1),
    ("sore throat", "throat pain", 1),
    ("feeling nauseous", "nausea", 1),
    ("dizzy", "feeling lightheaded", 1),
    ("my chest feels tight", "chest tightness", 1),
    ("heart is racing", "rapid heartbeat", 1),

    # Hard positives
    ("pressure behind my eyes", "eye strain", 1),
    ("my head feels heavy", "headache", 1),
    ("uneasy stomach", "nausea", 1),
    ("room spinning feeling", "dizziness", 1),
    ("burning feeling in chest", "acid reflux", 1),
    ("pins and needles in fingers", "tingling in fingers", 1),
    ("hard to catch my breath", "shortness of breath", 1),
    ("tight feeling in neck", "neck stiffness", 1),
    ("cramps in my belly", "abdominal cramps", 1),
    ("feeling drained and weak", "fatigue", 1),

    # Clear negatives
    ("knee pain", "headache", 0),
    ("stomach ache", "chest pain", 0),
    ("dizziness", "skin rash", 0),
    ("sore throat", "back pain", 0),
    ("nausea", "leg cramps", 0),
    ("blurred vision", "ear pain", 0),
    ("shortness of breath", "finger pain", 0),
    ("fatigue", "ankle swelling", 0),
    ("toothache", "abdominal pain", 0),
    ("itchy skin", "joint pain", 0),

    # Tricky negatives
    ("chest pain", "heartburn", 0),
    ("headache", "migraine", 0),
    ("back stiffness", "back pain", 0),
    ("fatigue", "sleepiness", 0),
    ("anxiety", "rapid heartbeat", 0),
    ("nausea", "loss of appetite", 0),
    ("joint pain", "muscle soreness", 0),
    ("fever", "feeling warm", 0),
    ("dizziness", "blurred vision", 0),
    ("tingling", "numbness", 0),

    # Body part variations
    ("left knee pain", "knee pain", 1),
    ("right shoulder hurts", "shoulder pain", 1),
    ("pain in upper back", "back pain", 1),
    ("neck pain", "upper spine pain", 1),
    ("wrist pain", "hand pain", 0),
    ("ankle pain", "foot pain", 0),
    ("lower abdominal pain", "stomach pain", 1),
    ("pain in fingers", "hand pain", 1),
    ("eye pain", "pain behind eyes", 1),
    ("jaw pain", "tooth pain", 0),
]


STANDARDIZATION_PROMPT = """
You standardize short health phrases for concept deduplication.

Return ONLY valid JSON. No explanation.

Use SNOMED CT-style clinical wording when possible, but keep the result concise.
Preserve important body part and laterality details.
Do not infer diagnoses that are not stated.

Schema:
{
  "canonical_phrase": string,
  "symptom": string or null,
  "body_parts": list[string],
  "laterality": "left | right | bilateral | null"
}

Examples:
Input: "my knee hurts"
Output: {"canonical_phrase": "knee pain", "symptom": "pain", "body_parts": ["knee"], "laterality": null}

Input: "feeling nauseous"
Output: {"canonical_phrase": "nausea", "symptom": "nausea", "body_parts": [], "laterality": null}
"""


def standardize_phrase(phrase):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": STANDARDIZATION_PROMPT},
            {"role": "user", "content": phrase},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    data = json.loads(text)
    return data["canonical_phrase"].strip().lower()


def get_embedding(text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(left, right):
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def run_threshold_experiment(pairs):
    rows = []
    phrase_cache = {}
    embedding_cache = {}

    for phrase_a, phrase_b, label in pairs:
        canonical_a = phrase_cache.setdefault(phrase_a, standardize_phrase(phrase_a))
        canonical_b = phrase_cache.setdefault(phrase_b, standardize_phrase(phrase_b))

        embedding_a = embedding_cache.setdefault(canonical_a, get_embedding(canonical_a))
        embedding_b = embedding_cache.setdefault(canonical_b, get_embedding(canonical_b))

        rows.append({
            "phrase_a": phrase_a,
            "phrase_b": phrase_b,
            "canonical_a": canonical_a,
            "canonical_b": canonical_b,
            "label": label,
            "score": cosine_similarity(embedding_a, embedding_b),
        })

    thresholds = np.linspace(0.50, 0.99, 100)
    best_threshold = None
    best_errors = float("inf")

    labels = np.array([row["label"] for row in rows])
    scores = np.array([row["score"] for row in rows])

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        errors = int(np.sum(predictions != labels))
        if errors < best_errors:
            best_errors = errors
            best_threshold = float(threshold)

    return rows, best_threshold, best_errors


def write_csv(rows, best_threshold):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    output_path = ARTIFACT_DIR / "embedding_threshold_results.csv"
    fieldnames = [
        "phrase_a",
        "phrase_b",
        "canonical_a",
        "canonical_b",
        "label",
        "score",
        "prediction",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["prediction"] = int(row["score"] >= best_threshold)
            writer.writerow(output)

    return output_path


def write_plot(rows, best_threshold):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ARTIFACT_DIR.mkdir(exist_ok=True)
    output_path = ARTIFACT_DIR / "embedding_threshold_histogram.png"

    positive_scores = [row["score"] for row in rows if row["label"] == 1]
    negative_scores = [row["score"] for row in rows if row["label"] == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(positive_scores, bins=20, alpha=0.6, label="Positive")
    plt.hist(negative_scores, bins=20, alpha=0.6, label="Negative")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Threshold = {best_threshold:.3f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("LLM-standardized phrase embedding scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def print_report(rows, best_threshold, best_errors):
    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Errors: {best_errors}/{len(rows)}")

    print("\nFalse positives")
    for row in rows:
        if row["score"] >= best_threshold and row["label"] == 0:
            print(
                f'{row["score"]:.3f} | {row["phrase_a"]} -> {row["canonical_a"]} '
                f'<-> {row["phrase_b"]} -> {row["canonical_b"]}'
            )

    print("\nFalse negatives")
    for row in rows:
        if row["score"] < best_threshold and row["label"] == 1:
            print(
                f'{row["score"]:.3f} | {row["phrase_a"]} -> {row["canonical_a"]} '
                f'<-> {row["phrase_b"]} -> {row["canonical_b"]}'
            )


def test_llm_standardized_embedding_threshold():
    rows, best_threshold, best_errors = run_threshold_experiment(TEST_PAIRS_LABELED)

    csv_path = write_csv(rows, best_threshold)
    plot_path = write_plot(rows, best_threshold)
    print_report(rows, best_threshold, best_errors)

    print(f"\nCSV: {csv_path}")
    if plot_path:
        print(f"Plot: {plot_path}")
    else:
        print("Plot skipped: matplotlib is not installed.")

    assert best_threshold is not None


if __name__ == "__main__":
    rows, best_threshold, best_errors = run_threshold_experiment(TEST_PAIRS_LABELED)
    csv_path = write_csv(rows, best_threshold)
    plot_path = write_plot(rows, best_threshold)
    print_report(rows, best_threshold, best_errors)
    print(f"\nCSV: {csv_path}")
    if plot_path:
        print(f"Plot: {plot_path}")
    else:
        print("Plot skipped: matplotlib is not installed.")
