# ccms_ai_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from retrieval_engine import retrieve_similar_cases
from insight_aggregator import InsightAggregator
from confidence_engine import ConfidenceEngine
from explanation_generator import ExplanationGenerator

from database import fetch_case_database
from config import TOP_K


# 🔹 Initialize FastAPI App
app = FastAPI(title="CCMS AI Clinical Insight Service")


# 🔹 Load database once
case_database: Dict = fetch_case_database()

insight_aggregator = InsightAggregator()
confidence_engine = ConfidenceEngine()
explanation_generator = ExplanationGenerator()


# 🔹 Input Schema
class CaseInput(BaseModel):
    symptoms: List[str]
    doctor_notes: str


# 🔹 API Endpoint
@app.post("/analyze-case")
def analyze_case(case: CaseInput):

    # 🔹 Prepare patient case (QUERY)
    query_case = {
        "symptoms": case.symptoms,
        "doctor_notes": case.doctor_notes
    }

    # 🔹 Retrieve similar cases (ONLY via retrieval engine)
    top_matches = retrieve_similar_cases(
        query=query_case,
        case_database=case_database,
        top_k=TOP_K
    )

    # 🔹 Safety check
    if not top_matches:
        return {
            "similar_cases": [],
            "diagnosis": "No diagnosis available",
            "treatment_pattern": "No treatment pattern found",
            "confidence": {
                "score": 0,
                "level": "Very Low Confidence"
            },
            "explanation": "No similar clinical cases were found in the database."
        }

    # 🔹 retrieved_cases already contain full data from retrieval engine
    retrieved_cases = top_matches

    # 🔹 Generate insight
    insight = insight_aggregator.aggregate_insights(retrieved_cases)

    # 🔹 Compute confidence
    confidence = confidence_engine.compute_confidence(retrieved_cases)

    insight["confidence_score"] = confidence.get("confidence_score", 0)
    insight["confidence_level"] = confidence.get("confidence_level", "Very Low")

    # 🔹 Generate explanation
    explanation = explanation_generator.generate_explanation(
        insight,
        retrieved_cases
    )

    # 🔹 Format similar cases
    similar_cases = [
        {
            "case_id": case.get("case_id"),
            "similarity": round(float(case.get("similarity", 0)), 4)
        }
        for case in retrieved_cases
    ]

    # 🔹 Final response
    return {
        "similar_cases": similar_cases,
        "diagnosis": insight.get("diagnosis", "Unknown condition"),
        "treatment_pattern": insight.get(
            "treatment",
            "No treatment pattern found"
        ),
        "confidence": {
            "score": insight.get("confidence_score", 0),
            "level": insight.get("confidence_level", "Very Low Confidence")
        },
        "explanation": explanation
    }