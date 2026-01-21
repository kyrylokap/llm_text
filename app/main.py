import base64
import json
from json import JSONDecodeError
from typing import Optional
from .errors import ToolError, ToolTimeout
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from .guardrails import guard_input, scrub_output
from .llm_runner import run_with_retry_chat
from .errors import (
    SecurityBlocked,
    ValidationError,
)
from .app_logging import logger

app = FastAPI(title="Groq Hosted Model API")


@app.get("/")
def root():
    return ({"message": "This is my llm_text"})


@app.post("/chat")
async def ask(
        message: str = Form(),
        history: str = Form(),
        image: Optional[UploadFile] = File(None),
        k: int = 3,
        mode: str = "api",
        use_functions: bool = False
):
    logger.info("Endpoint ask called")
    try:
        image_b64 = None
        if image:
            logger.info(f"Processing image: {image.filename}")
            content = await image.read()
            image_b64 = base64.b64encode(content).decode('utf-8')

        try:
            history_list = json.loads(history)
        except JSONDecodeError:
            logger.error("ValidationError")
            raise ValidationError("Model returned invalid JSON")

        guard_input(message)

        result = run_with_retry_chat(
            current_message=message,
            use_functions=use_functions,
            history=history_list,
            api_mode=mode,
            image_data=image_b64,
        )
        try:
            if result.get("type") == "chat":
                return {
                    "status": "chat",
                    "message": result["message"],
                    "latency_s": result["latency_s"]
                }

            if result.get("type") == "report":
                output = scrub_output(result["data"])
                return {
                    "status": "complete",
                    "report": output,
                    "latency_s": result["latency_s"]
                }
        except ValueError:
            logger.error("ValueError")

            return {
                "message": "Sorry, can't find any illnesses with those symptoms. Please, try again.",
                "latency_s": result["latency_s"]
            }

    except SecurityBlocked as e:
        logger.error("HTTPException")

        raise HTTPException(status_code=400, detail=e.detail)
    except ValidationError as e:
        logger.error("ValidationError")

        raise HTTPException(status_code=422, detail=e.detail)
    except ToolTimeout as e:
        logger.error("ToolTimeout")

        raise HTTPException(status_code=504, detail=e.detail)
    except ToolError as e:
        logger.error("ToolError")

        raise HTTPException(status_code=502, detail=e.detail)
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(e)

        raise HTTPException(status_code=500, detail="Internal server error")
