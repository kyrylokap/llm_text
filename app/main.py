
from .errors import ToolError, ToolTimeout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .guardrails import guard_input,scrub_output
from .llm_runner import run_with_retry_chat
from langdetect import detect
from .errors import (
    SecurityBlocked,
    ValidationError,
)
app = FastAPI(title="Groq Hosted Model API")



class AskRequest(BaseModel):
    symptoms: str
    k: int = 3
    mode: str = "api"
    use_functions: bool = False

@app.get("/")
def root():
    return ({"message": "This is my llm_text"})

@app.post("/ask")
def ask(request: AskRequest):

    try:
        safe_input = guard_input(request.symptoms)

        result = run_with_retry_chat(
            safe_input,
            use_functions=request.use_functions,
            mode="medical",
            api_mode=request.mode,
        )
        print("Result in main ",result)
        try:
            output = scrub_output(result["text"])
            return {
                "illnesses": output,
                "latency_s": result["latency_s"]
            }

        except ValueError:
            return {
                "message": "Sorry, can't find any illnesses with those symptoms. Please, try again.",
                "latency_s": result["latency_s"]
            }

    except SecurityBlocked as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.detail)
    except ToolTimeout as e:
        raise HTTPException(status_code=504, detail=e.detail)
    except ToolError as e:
        raise HTTPException(status_code=502, detail=e.detail)
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
