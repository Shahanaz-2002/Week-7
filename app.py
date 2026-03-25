# app.py

from fastapi import FastAPI, HTTPException
from typing import Dict
import time

from models import CaseRequest, CaseResponse, SimilarCase, SystemMetrics
from retrieval_engine import retrieve_similar_cases
from insight_generator import InsightGenerator
from database import fetch_case_database
from config import TOP_K


app = FastAPI(title="CCMS AI Similarity Engine")


# 🔹 GLOBALS
case_database: Dict = {}
insight_generator: InsightGenerator = None
response_cache = {}


# 🔹 STARTUP INITIALIZATION
@app.on_event("startup")
def initialize_system():

    global case_database, insight_generator

    case_database = fetch_case_database()

    print("📦 DB SIZE:", len(case_database))

    if not case_database:
        print("⚠ No cases found in database.")
        return

    # Only insight generator needed
    insight_generator = InsightGenerator()

    print("✅ System initialized successfully.")


# 🔹 OUTPUT QUALITY
def determine_output_quality(confidence_score: float) -> str:

    if confidence_score > 0.8:
        return "High"
    elif confidence_score > 0.6:
        return "Moderate"
    else:
        return "Low"


# 🔹 MAIN API ENDPOINT
@app.post("/analyze-case", response_model=CaseResponse)
def analyze_case(request: CaseRequest):

    start_time = time.time()

    request_key = str(request.symptoms) + request.doctor_notes

    # 🔹 CACHE CHECK
    if request_key in response_cache:
        return response_cache[request_key]

    try:

        if not case_database:
            raise HTTPException(
                status_code=500,
                detail="System not initialized properly."
            )

        # 🔹 Prepare query case
        query_case = {
            "symptoms": request.symptoms,
            "doctor_notes": request.doctor_notes
        }

        # 🔹 Retrieve similar cases (ONLY via retrieval engine)
        top_matches = retrieve_similar_cases(
            query=query_case,
            case_database=case_database,
            top_k=TOP_K
        )

        print("🔍 TOP MATCHES:", top_matches)

        # 🔹 Safety check
        if not top_matches:
            raise ValueError("No similar cases retrieved.")

        # 🔹 Format similar cases
        similar_cases = [
            SimilarCase(
                case_id=case.get("case_id"),
                similarity_score=round(float(case.get("similarity", 0.0)), 4)
            )
            for case in top_matches
        ]

        # 🔹 Generate insight
        insight = insight_generator.generate_insight(
            query=query_case,
            case_database=case_database
        )

        print("🧠 INSIGHT:", insight)

        predicted_diagnosis = insight.get("prediction", "N/A")
        confidence_score = insight.get("confidence", 0.0)
        explanation = insight.get("explanation", "N/A")

        # Optional treatment
        suggested_treatment = insight.get("treatment", "N/A")

        # 🔹 Measure Response Time
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        # 🔹 Output Quality
        output_quality = determine_output_quality(confidence_score)

        system_metrics = SystemMetrics(
            response_time_ms=round(response_time_ms, 2),
            output_quality=output_quality
        )

        # 🔹 FINAL RESPONSE
        response = CaseResponse(
            similar_cases=similar_cases,

            predicted_diagnosis=predicted_diagnosis,
            suggested_treatment=suggested_treatment,

            confidence_score=confidence_score,
            confidence_reason=output_quality,

            explanation=explanation,

            system_metrics=system_metrics
        )

        # 🔹 Cache result
        response_cache[request_key] = response

        return response

    except Exception as e:

        print("❌ ERROR:", str(e))

        return CaseResponse(
            similar_cases=[],

            predicted_diagnosis="Error",
            suggested_treatment="Error",

            confidence_score=0.0,
            confidence_reason="Unable to compute similarity.",

            explanation=f"System error: {str(e)}",

            system_metrics=SystemMetrics(
                response_time_ms=0.0,
                output_quality="Error"
            )
        )