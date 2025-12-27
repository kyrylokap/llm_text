import os
import time
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .guardrails import guard_input,scrub_output
from openai import OpenAI
from .logging import logger
from .tools import TOOLS,lookup_diseases
from langdetect import detect
load_dotenv()

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

app = FastAPI(title="Groq Hosted Model API")



class AskRequest(BaseModel):
    symptoms: str
    k: int = 3
    mode: str = "api"  # "api" | "local"
    use_functions: bool = False


from openai import OpenAI

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def chat_once(symptoms: str, use_functions: bool = True):
    system_prompt = """
Jestes asystentem medycznym. Podaj 3 najprawdopodobniejsze choroby na podstawie objawów.
Zawsze używaj nazw symptomów w języku angielskim.
Format odpowiedzi: illness,illness,illness.
"""
    t0 = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": symptoms}
        ],
        max_tokens=512,
        temperature=0.7,
        functions=TOOLS if use_functions else None,
        function_call="auto" if use_functions else "none"
    )

    choice = response.choices[0]

    if getattr(choice.message, "function_call", None):
        symptoms_list = [s.strip().lower() for s in symptoms.split(",")]
        diseases_list = lookup_diseases(symptoms_list, top_k=3)
        text_output = ", ".join(diseases_list)
    else:
        text_output = getattr(choice.message, "content", "")

    dt = time.time() - t0

    return {"text": text_output, "latency_s": round(dt, 3)}


# --- ENDPOINTS ---
@app.get("/")
def root():
    return {"message": "Groq Hosted Model API is running"}


@app.post("/ask")
def ask(request: AskRequest):
    start_time = datetime.now()

    safe_symptoms = guard_input(request.symptoms)

    try:
        lang = detect(safe_symptoms)
        use_functions = True if lang == "en" else False

        result = chat_once(safe_symptoms, use_functions=use_functions)

        logger.info(f"{start_time} | tool=chat_once | status=OK | latency_s={result['latency_s']:.3f}")

        illnesses = scrub_output(result["text"]) if result["text"] else []

        return {
            "illnesses": illnesses,
            "latency_s": result["latency_s"]
        }


    except HTTPException as e:
        logger.error(f"{start_time} | tool=chat_once | status=ERROR | code={e.status_code} | detail={e.detail}")
        raise

    except Exception as e:
        logger.error(f"{start_time} | tool=chat_once | status=ERROR | detail={str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
