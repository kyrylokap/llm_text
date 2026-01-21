from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class MedicalReport(BaseModel):
    summary: str = Field(description="A brief summary of the patient's reported condition and history")
    reported_symptoms: List[str] = Field(description="List of symptoms extracted from the patient's description")
    duration: str = Field(
        description="Duration of symptoms as reported by the patient (e.g., '2 days', 'since yesterday')")
    ai_diagnosis_suggestion: str = Field(description="Primary preliminary diagnosis suggestion based on the analysis")
    recommended_specialization: List[str] = Field(
        description="List of recommended medical specialists (e.g., 'Dermatologist', 'General Practitioner')")
    confidence_score: float = Field(
        description="Confidence score of the assessment ranging from 0.0 (uncertain) to 1.0 (highly confident)", ge=0.0,
        le=1.0)

    class Config:
        extra = "ignore"


class AgentResponse(BaseModel):
    action: Literal["question", "final_report"]
    message_to_patient: Optional[str] = None
    report_data: Optional[MedicalReport] = None


class ChatMessage(BaseModel):
    role: str
    content: str
