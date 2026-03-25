# main.py

import time
from typing import Dict

from retrieval_engine import retrieve_similar_cases
from insight_aggregator import InsightAggregator
from confidence_engine import ConfidenceEngine
from explanation_generator import ExplanationGenerator

from utils import validate_case_input, format_output, log
from config import TOP_K
from database import fetch_case_database


def main():

    try:

        start_time = time.time()

        # 🔹 Load Data
        log("Loading case database...")
        case_database: Dict = fetch_case_database()

        # 🔹 Initialize Engines
        insight_aggregator = InsightAggregator()
        confidence_engine = ConfidenceEngine()
        explanation_generator = ExplanationGenerator()

        # 🔹 Test Query Cases
        test_cases = [
            {
                "case_id": "NEW_SKIN_003",
                "clinic_id": "CLINIC_001",
                "symptoms": ["pus-filled pimples on cheeks", "oily skin"],
                "duration_days": 60,
                "doctor_notes": "Inflammatory acne lesions with occasional scarring.",
                "diagnosis": "",
                "treatment": "",
                "outcome": "",
                "recovery_days": None,
                "patient": {"age": 24, "gender": "Female"}
            }
        ]

        # 🔹 Process Each Case
        for new_case in test_cases:

            log(f"Processing {new_case['case_id']}")

            # 🔹 Validate input
            validate_case_input(new_case)

            # 🔹 Retrieve similar cases (ONLY via retrieval engine)
            top_matches = retrieve_similar_cases(
                query=new_case,
                case_database=case_database,
                top_k=TOP_K
            )

            # 🔹 Safety check
            if not top_matches:
                log("No similar cases found.")
                continue

            # 🔹 retrieved_cases already contain full data
            retrieved_cases = top_matches

            # 🔹 Generate Insights
            insight = insight_aggregator.aggregate_insights(retrieved_cases)

            # 🔹 Compute Confidence Score
            confidence = confidence_engine.compute_confidence(retrieved_cases)

            insight["confidence_score"] = confidence.get("confidence_score", 0)
            insight["confidence_level"] = confidence.get("confidence_level", "Very Low")

            # 🔹 Generate Clinical Explanation
            explanation = explanation_generator.generate_explanation(
                insight,
                retrieved_cases
            )

            insight["clinical_explanation"] = explanation

            # 🔹 Format Final Output
            result = format_output(
                query_case_id=new_case["case_id"],
                top_matches=retrieved_cases,
                insight=insight
            )

            print(result)

        end_time = time.time()

        log(f"Execution Time: {end_time - start_time:.2f} sec")

    except Exception as e:

        log(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()