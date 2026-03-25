from pydantic import BaseModel, Field
from typing import List


# 🔹 REQUEST MODEL
class CaseRequest(BaseModel):
    symptoms: List[str] = Field(..., min_items=1)
    doctor_notes: str = Field(..., min_length=3)


# 🔹 SIMILAR CASE MODEL
class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


# 🔹 SYSTEM METRICS MODEL
class SystemMetrics(BaseModel):
    response_time_ms: float = Field(..., ge=0)
    output_quality: str = Field(..., min_length=1)


# 🔹 FINAL RESPONSE MODEL
class CaseResponse(BaseModel):

    # Retrieved similar cases
    similar_cases: List[SimilarCase]

    # Prediction outputs
    predicted_diagnosis: str = Field(default="Unknown")
    suggested_treatment: str = Field(default="No treatment available")

    # Confidence outputs
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_reason: str = Field(default="Low Confidence")

    # Explanation
    explanation: str = Field(default="No explanation available")

    # System performance
    system_metrics: SystemMetrics