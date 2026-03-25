# insight_generator.py

from typing import List, Dict
from collections import Counter
from retrieval_engine import retrieve_similar_cases


class InsightGenerator:

    def __init__(self):
        pass


    # 🔹 MAIN FUNCTION (UPDATED FOR DAY 2)
    def generate_insight(self, query: str, case_database: Dict) -> Dict:

        # 🔹 Step 0: Get similar cases ONLY from retrieval engine
        top_matches = retrieve_similar_cases(
            query=query,
            case_database=case_database,
            top_k=3
        )

        # 🔹 Case: No matches found
        if not top_matches:
            return {
                "prediction": None,
                "confidence": 0.0,
                "explanation": "No similar cases found."
            }

        # 🔹 Step 1: Extract diagnoses
        diagnoses = [case["diagnosis"] for case in top_matches]

        # 🔹 Step 2: Majority Voting
        diagnosis_counts = Counter(diagnoses)
        predicted_diagnosis = diagnosis_counts.most_common(1)[0][0]

        # 🔹 Step 3: Confidence Score (Average similarity)
        similarities = [case["similarity"] for case in top_matches]
        confidence = sum(similarities) / len(similarities)

        # 🔹 Step 4: Explanation Generation
        explanation = self._generate_explanation(top_matches, predicted_diagnosis)

        # 🔹 Final Output
        return {
            "prediction": predicted_diagnosis,
            "confidence": round(confidence, 3),
            "explanation": explanation
        }


    # 🔹 Explanation Logic
    def _generate_explanation(self, top_matches: List[Dict], prediction: str) -> str:

        high_sim_cases = [
            case for case in top_matches if case["similarity"] > 0.8
        ]

        return (
            f"Prediction is '{prediction}' based on {len(top_matches)} similar cases. "
            f"{len(high_sim_cases)} cases have high similarity (>0.8). "
            f"Most frequent diagnosis among retrieved cases is '{prediction}'."
        )


# 🔹 TEST BLOCK (UPDATED)
if __name__ == "__main__":

    # Dummy case database (simulating stored cases)
    case_database = {
        "C1": {
            "case_id": "C1",
            "features": [0.1, 0.2, 0.3],
            "diagnosis": "Arrhythmia",
            "outcome": "Recovered"
        },
        "C2": {
            "case_id": "C2",
            "features": [0.2, 0.1, 0.4],
            "diagnosis": "Arrhythmia",
            "outcome": "Stable"
        },
        "C3": {
            "case_id": "C3",
            "features": [0.9, 0.8, 0.7],
            "diagnosis": "Normal",
            "outcome": "Stable"
        }
    }

    # Example query
    query = "irregular heartbeat and chest discomfort"

    engine = InsightGenerator()
    result = engine.generate_insight(query, case_database)

    print("\n🧠 Insight Output:\n")
    print(result)