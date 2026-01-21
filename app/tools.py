from typing import Type
from pydantic import BaseModel, conlist, Field
import pandas as pd


class DiagnoseArgs(BaseModel):
    symptoms: conlist(str, min_length=1, max_length=15) = Field(
        ...,
        description="List of symptoms perceived by the patient (e.g. ['fever', 'headache'])"
    )
    top_k: int = Field(
        3,
        description="Number of probable diseases to return"
    )


diseases = pd.read_csv("diseases.csv")
diseases.columns = diseases.columns.str.lower()


def lookup_diseases(symptoms: list[str], top_k: int = 3):
    symptoms = [s.strip().lower() for s in symptoms]
    valid = [s for s in symptoms if s in diseases.columns]

    if not valid:
        return []

    df = diseases.copy()
    df["score"] = df[valid].sum(axis=1)
    top = df.sort_values("score", ascending=False).head(top_k)
    return top["diseases"].tolist()


def generate_openai_schema(tool_class: Type[BaseModel], name: str, description: str):
    schema = tool_class.model_json_schema()

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        },
    }


TOOLS = {
    "diagnose": {
        "openai_schema": generate_openai_schema(
            DiagnoseArgs,
            name="diagnose",
            description="Return probable diseases based on a list of symptoms provided by the patient."
        ),
        "args_schema": DiagnoseArgs,
        "implementation": lookup_diseases,
    }
}
