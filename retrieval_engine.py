# retrieval_engine.py

from typing import List, Dict, Tuple
import numpy as np

# 🔹 Example: Dummy database (replace with your actual DB)
case_database = {
    "C1": {
        "features": np.array([0.1, 0.2, 0.3]),
        "diagnosis": "Arrhythmia",
        "outcome": "Recovered"
    },
    "C2": {
        "features": np.array([0.2, 0.1, 0.4]),
        "diagnosis": "Normal",
        "outcome": "Stable"
    },
    "C3": {
        "features": np.array([0.9, 0.8, 0.7]),
        "diagnosis": "Tachycardia",
        "outcome": "Critical"
    }
}

# 🔹 Cosine Similarity Function
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 🔹 MAIN FUNCTION (IMPORTANT)
def retrieve_similar_cases(
    input_case: Dict,
    top_k: int = 3
) -> List[Dict]:

    input_features = np.array(input_case["features"])

    similarity_scores: List[Tuple[str, float]] = []

    # 🔹 Step 1: Compute similarity with all cases
    for case_id, case_data in case_database.items():
        sim = cosine_similarity(input_features, case_data["features"])
        similarity_scores.append((case_id, sim))

    # 🔹 Step 2: Sort by similarity (descending)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 🔹 Step 3: Take top-K
    top_matches = similarity_scores[:top_k]

    # 🔹 Step 4: Convert to structured format (DAY-1 CONTRACT)
    results = []

    for case_id, sim in top_matches:
        case_data = case_database[case_id]

        results.append({
            "case_id": case_id,
            "similarity": float(sim),
            "features": case_data.get("features").tolist(),
            "diagnosis": case_data.get("diagnosis"),
            "outcome": case_data.get("outcome")
        })

    return results


# 🔹 TEST (Run this file directly)
if __name__ == "__main__":
    
    test_input = {
        "features": [0.1, 0.2, 0.25]
    }

    results = retrieve_similar_cases(test_input)

    print("\n🔍 Retrieved Cases:\n")
    for r in results:
        print(r)