from typing import Type, List, Literal, Optional
from pydantic import BaseModel, conlist, Field
import pandas as pd


class MedicalReport(BaseModel):
    summary: str = Field(..., description="A brief summary of the patient's reported condition and history")
    reported_symptoms: List[str] = Field(..., description="List of symptoms extracted from the patient's description")
    duration: str = Field(..., description="Duration of symptoms as reported by the patient (e.g., '2 days', 'since yesterday')")
    ai_diagnosis_suggestion: str = Field(..., description="Primary preliminary diagnosis suggestion based on the analysis")
    recommended_specialization: List[str] = Field(..., description="List of recommended medical specialists (e.g.,"
                                                                   " 'Dermatologist', 'General Practitioner')")
    confidence_score: float = Field(...,description="Confidence score of the assessment ranging from "
                                                    "0.0 (uncertain) to 1.0 (highly confident)",
                                    ge=0.0,
                                    le=1.0)


class ResponseArgs(BaseModel):
    action: Literal["message", "final_report"] = Field(...,
                                                       description="Choose 'message' to ask questions, or 'final_report' to give diagnosis")
    message_to_patient: Optional[str] = Field(None, description="The text of the question (if action is message)")
    report_data: Optional[MedicalReport] = Field(None, description="The full report object (if action is final_report)")


class DiagnoseArgs(BaseModel):
    symptoms: conlist(str, min_length=1, max_length=15) = Field(
        ...,
        description="List of symptoms perceived by the patient (e.g. ['fever', 'headache'])"
    )
    top_k: int = Field(
        5,
        description="Number of probable diseases to return"
    )


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

def response_placeholder(**kwargs):
    return kwargs


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
        "implementation": response_placeholder
    }
}
