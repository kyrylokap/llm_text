import pandas as pd


TOOLS = [
    {
        "name": "diagnose",
        "description": "Return probable diseases based on symptoms",
        "parameters": {
            "type": "object",
            "properties": {
                "diseases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of possible diseases"
                }
            },
            "required": ["diseases"]
        }
    }
]


diseases = pd.read_csv("diseases.csv")


def lookup_diseases(symptoms: list[str], top_k=3):
    symptoms = [s.strip().lower() for s in symptoms]

    valid_columns = [s for s in symptoms if s in diseases.columns.str.lower()]

    if not valid_columns:
        return []

    df = diseases.copy()
    df.columns = df.columns.str.lower()
    df['score'] = df[valid_columns].sum(axis=1)

    top = df.sort_values('score', ascending=False).head(top_k)
    return top['diseases'].tolist()
