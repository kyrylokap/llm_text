from typing import Type, List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationError
import pandas as pd
from .app_logging import logger


class MedicalReport(BaseModel):
    reported_summary: str = Field(..., description="A brief summary of the patient's reported condition and history")

    reported_symptoms: str = Field(...,
                                   description="String containing symptoms extracted from the patient's description, "
                                               "separated by commas (e.g. 'Fever, headache, fatigue').")

    sickness_duration: str = Field(...,
                                   description="Duration of symptoms as reported by the patient (e.g., '2 days', 'since yesterday')")

    ai_primary_diagnosis: str = Field(
        ...,
        description="The specific name of the identified condition. "
    )

    ai_diagnosis_reasoning: str = Field(
        ...,
        description="Primary preliminary diagnosis suggestion based on the analysis. "
                    "EXPLANATION: Why do you think this is the case? "
    )

    ai_suggested_management: List[str] = Field(
        ...,
        description="ACTION PLAN: List of recommended steps or treatments. (e.g., 'Rest', 'Drink water')"
                    "NOT a single comma-separated string."
                    "Priority: Use specific advice from RAG context. "
                    "Fallback: If RAG lacks treatment info, provide standard general medical guidelines."
    )

    ai_critical_warning: Optional[str] = Field(
        None,
        description="If the condition requires immediate ER visit (e.g. heart attack signs), "
                    "state it here clearly. If no immediate danger exists, return null."
    )

    ai_recommended_specializations: List[str] = Field(..., description="List of recommended medical specialists (e.g.,"
                                                                       " 'Dermatologist', 'General Practitioner')")

    ai_confidence_score: float = Field(..., description="Confidence score of the assessment ranging from "
                                                        "0.0 (uncertain) to 1.0 (highly confident)",
                                       ge=0.0,
                                       le=1.0)


class ResponseArgs(BaseModel):
    action: Literal["message", "final_report"] = Field(...,
                                                       description="Choose 'message' to ask questions, or 'final_report' to give diagnosis")
    message_to_patient: Optional[str] = Field(None, description="The text of the question (if action is message)")
    report_data: Optional[MedicalReport] = Field(None, description="The full report object (if action is final_report)")


class DiagnoseArgs(BaseModel):
    symptoms: List[str] = Field(
        ...,
        min_length=1,
        max_length=15,
        description="List of symptoms perceived by the patient (e.g. ['fever', 'headache'])"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of probable diseases to return"
    )

    @field_validator('symptoms')
    @classmethod
    def validator_symptoms_item(cls, symptoms: List[str]):
        for i, symptom in enumerate(symptoms):
            if len(symptom) > 100:
                logger.error(f"[ERROR] Symptom at index {i} is too long (max 100 chars)")
                raise ValueError(f"Symptom at index {i} is too long (max 100 chars)")

            clean_symptom = symptom.strip()
            if not clean_symptom:
                logger.error(f"[ERROR] Symptom at index {i} is empty")
                raise ValueError(f"Symptom at index {i} is empty")

            symptoms[i] = clean_symptom
        return symptoms


diseases = pd.read_csv("diseases.csv")
diseases.columns = diseases.columns.str.lower()


def lookup_diseases(symptoms: list[str], top_k: int = 5):
    symptoms = [s.strip().lower() for s in symptoms]
    valid = [s for s in symptoms if s in diseases.columns]

    if not valid:
        return []

    df = diseases.copy()
    df["score"] = df[valid].sum(axis=1)
    top = df.sort_values("score", ascending=False).head(top_k)
    return top["diseases"].tolist()


def provide_response_implementation(**kwargs) -> Dict[str, Any]:
    try:
        response_obj = ResponseArgs(**kwargs)

        if response_obj.action == 'final_report' and not response_obj.report_data:
            logger.error("[ERROR] Action is 'final_report' but 'report_data' is missing.")
            raise ValidationError("Action is 'final_report' but 'report_data' is missing.")

        if response_obj.action == "message" and not response_obj.message_to_patient:
            logger.error("[ERROR] Action is 'message' but 'message_to_patient' is missing.")
            raise ValidationError("Action is 'message' but 'message_to_patient' is missing.")

        return response_obj.model_dump(exclude_none=True)
    except Exception as e:
        logger.error(f"[ERROR] Response Validation Failed (provide_response_implementation): {str(e)}")
        return {"error": f"Response Validation Failed (provide_response_implementation): {str(e)}"}


def generate_openai_tool_definition(tool_class: Type[BaseModel], name: str, description: str):
    schema = tool_class.model_json_schema()

    parameters = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }

    if "$defs" in schema:
        parameters["$defs"] = schema["$defs"]

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


TOOLS = {
    "diagnose": {
        "tool_definition": generate_openai_tool_definition(
            DiagnoseArgs,
            name="diagnose",
            description="Return probable diseases based on a list of symptoms provided by the patient."
        ),
        "args_schema": DiagnoseArgs,
        "implementation": lookup_diseases,
    },
    "provide_response": {
        "tool_definition": generate_openai_tool_definition(
            ResponseArgs,
            name="provide_response",
            description="ALWAYS use this tool to communicate the final answer or ask questions to the user."
        ),
        "args_schema": ResponseArgs,
        "implementation": provide_response_implementation
    }
}
