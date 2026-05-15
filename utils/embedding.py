from openai import OpenAI
import numpy as np

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def get_embedding(text: str):
    if text is None:
        return None

    response = client.embeddings.create(
        model="text-embedding-qwen3-embedding-0.6b",
        input=text
    )

    return np.array(response.data[0].embedding)

def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    """Fetch embeddings for a list of strings in a single batch call."""
    if not texts:
        return []
        
    # Filter out None/empty
    valid_texts = [t for t in texts if t]
    if not valid_texts:
        return [None] * len(texts)
        
    response = client.embeddings.create(
        model="text-embedding-qwen3-embedding-0.6b",
        input=valid_texts
    )
    
    # Map back to original order
    embeddings_map = {t: np.array(emb.embedding) for t, emb in zip(valid_texts, response.data)}
    return [embeddings_map.get(t) for t in texts]

def cosine_similarity(a, b):
    if a is None or b is None:
        return -1
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    test_pairs_labeled = [

    # ✅ CLEAR POSITIVES
    ["knee pain", "pain in knee", 1],
    ["my knee hurts", "knee pain", 1],
    ["sharp pain in lower back", "lower back pain", 1],
    ["headache", "pain in head", 1],
    ["stomach ache", "abdominal pain", 1],
    ["sore throat", "throat pain", 1],
    ["feeling nauseous", "nausea", 1],
    ["dizzy", "feeling lightheaded", 1],
    ["my chest feels tight", "chest tightness", 1],
    ["heart is racing", "rapid heartbeat", 1],

    # ⚠️ HARD POSITIVES
    ["pressure behind my eyes", "eye strain", 1],
    ["my head feels heavy", "headache", 1],
    ["uneasy stomach", "nausea", 1],
    ["room spinning feeling", "dizziness", 1],
    ["burning feeling in chest", "acid reflux", 1],
    ["pins and needles in fingers", "tingling in fingers", 1],
    ["hard to catch my breath", "shortness of breath", 1],
    ["tight feeling in neck", "neck stiffness", 1],
    ["cramps in my belly", "abdominal cramps", 1],
    ["feeling drained and weak", "fatigue", 1],

    # ❌ CLEAR NEGATIVES
    ["knee pain", "headache", 0],
    ["stomach ache", "chest pain", 0],
    ["dizziness", "skin rash", 0],
    ["sore throat", "back pain", 0],
    ["nausea", "leg cramps", 0],
    ["blurred vision", "ear pain", 0],
    ["shortness of breath", "finger pain", 0],
    ["fatigue", "ankle swelling", 0],
    ["toothache", "abdominal pain", 0],
    ["itchy skin", "joint pain", 0],

    # ⚠️ TRICKY NEGATIVES
    ["chest pain", "heartburn", 0],
    ["headache", "migraine", 0],       # intentionally strict
    ["back stiffness", "back pain", 0],
    ["fatigue", "sleepiness", 0],
    ["anxiety", "rapid heartbeat", 0],
    ["nausea", "loss of appetite", 0],
    ["joint pain", "muscle soreness", 0],
    ["fever", "feeling warm", 0],
    ["dizziness", "blurred vision", 0],
    ["tingling", "numbness", 0],

    # ⚠️ BODY PART VARIATIONS
    ["left knee pain", "knee pain", 1],
    ["right shoulder hurts", "shoulder pain", 1],
    ["pain in upper back", "back pain", 1],
    ["neck pain", "upper spine pain", 1],
    ["wrist pain", "hand pain", 0],
    ["ankle pain", "foot pain", 0],
    ["lower abdominal pain", "stomach pain", 1],
    ["pain in fingers", "hand pain", 1],
    ["eye pain", "pain behind eyes", 1],
    ["jaw pain", "tooth pain", 0],

    # ⚠️ COMPLEX PHRASES
    ["nausea and headache", "headache", 0],
    ["chest pain and dizziness", "chest pain", 0],
    ["back pain after running", "back pain", 1],
    ["knee pain from gym", "knee pain", 1],
    ["stomach ache after eating", "stomach ache", 1],
    ["headache with blurred vision", "headache", 0],
    ["fatigue and weakness", "fatigue", 1],
    ["pain in neck when turning", "neck pain", 1],
    ["sharp knee pain", "knee pain", 1],
    ["mild headache", "headache", 1],
]
    res = []
    for n1,n2,l in test_pairs_labeled:
        e1 = get_embedding(n1)
        e2 = get_embedding(n2)
        res.append((cosine_similarity(e1, e2), l, n1, n2))
    import numpy as np
    import matplotlib.pyplot as plt

    # your results: (score, label, n1, n2)
    scores = [r[0] for r in res]
    labels = [r[1] for r in res]

    scores = np.array(scores)
    labels = np.array(labels)

    # ----------------------------
    # 1. Plot distributions
    # ----------------------------
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(pos_scores, bins=20, alpha=0.6, label="Positive (should match)")
    plt.hist(neg_scores, bins=20, alpha=0.6, label="Negative (should NOT match)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Similarity Distribution")
    plt.show()


    # ----------------------------
    # 2. Find best threshold
    # ----------------------------
    thresholds = np.linspace(0.5, 0.95, 100)

    best_t = None
    best_error = float("inf")

    for t in thresholds:
        preds = (scores >= t).astype(int)

        # errors = false positives + false negatives
        errors = np.sum(preds != labels)

        if errors < best_error:
            best_error = errors
            best_t = t

    print(f"Best threshold: {best_t:.3f} with {best_error} errors")


    # ----------------------------
    # 3. Plot threshold line
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(pos_scores, bins=20, alpha=0.6, label="Positive")
    plt.hist(neg_scores, bins=20, alpha=0.6, label="Negative")

    plt.axvline(best_t, color='red', linestyle='--', label=f"Best Threshold = {best_t:.2f}")

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Threshold Selection")
    plt.show()


    # ----------------------------
    # 4. Print mistakes
    # ----------------------------
    print("\n--- False Positives (WRONG MATCHES) ---")
    for s, l, n1, n2 in res:
        if s >= best_t and l == 0:
            print(f"{s:.3f} | {n1}  <->  {n2}")

    print("\n--- False Negatives (MISSED MATCHES) ---")
    for s, l, n1, n2 in res:
        if s < best_t and l == 1:
            print(f"{s:.3f} | {n1}  <->  {n2}")