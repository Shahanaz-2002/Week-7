# insight_aggregator.py

from typing import List, Dict


class InsightAggregator:

    def aggregate_insights(self, top_matches: List[Dict]) -> Dict:

        # 🔹 Case: No matches
        if not top_matches:
            return {
                "diagnosis": "No similar cases found",
                "treatment": "No treatment pattern available",
                "confidence": 0.0
            }

        diagnosis_score = {}
        treatment_score = {}

        total_similarity = 0.0

        # 🔹 Aggregate weighted scores
        for case in top_matches:

            diagnosis = case.get("diagnosis", None)
            treatment = case.get("treatment", None)
            similarity = float(case.get("similarity", 0.0))

            total_similarity += similarity

            if diagnosis:
                diagnosis_score[diagnosis] = (
                    diagnosis_score.get(diagnosis, 0.0) + similarity
                )

            if treatment:
                treatment_score[treatment] = (
                    treatment_score.get(treatment, 0.0) + similarity
                )

        # 🔹 Select best diagnosis
        if diagnosis_score:
            predicted_diagnosis = max(diagnosis_score, key=diagnosis_score.get)
        else:
            predicted_diagnosis = "Unknown condition"

        # 🔹 Select best treatment
        if treatment_score:
            predicted_treatment = max(treatment_score, key=treatment_score.get)
        else:
            predicted_treatment = "No treatment pattern found"

        # 🔹 Confidence (normalized)
        confidence = total_similarity / len(top_matches)

        return {
            "diagnosis": predicted_diagnosis,
            "treatment": predicted_treatment,
            "confidence": round(confidence, 3)
        }


# 🔹 TEST BLOCK
if __name__ == "__main__":

    sample_top_matches = [
        {
            "case_id": "C1",
            "similarity": 0.92,
            "diagnosis": "Eczema",
            "treatment": "Topical steroids"
        },
        {
            "case_id": "C2",
            "similarity": 0.88,
            "diagnosis": "Eczema",
            "treatment": "Moisturizers"
        },
        {
            "case_id": "C3",
            "similarity": 0.75,
            "diagnosis": "Psoriasis",
            "treatment": "Vitamin D analogues"
        }
    ]

    aggregator = InsightAggregator()

    result = aggregator.aggregate_insights(sample_top_matches)

    print("\n📊 Aggregated Insight:\n")
    print(result)