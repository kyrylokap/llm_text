import base64
import json

from json import JSONDecodeError
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from .errors import ToolError, ToolTimeout, InvalidHistoryFormatError, ImageProcessingError
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from .guardrails import guard_input, scrub_output
from .llm_runner import run_with_retry_chat, ChatMessage
from .errors import (
    SecurityBlocked,
    ValidationError,
)
from .app_logging import logger


app = FastAPI(title="Groq Hosted Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return ({"message": "This is my llm_text"})


@app.post("/ask")
async def ask(
        message: str = Form(""),
        history: str = Form("[]"),
        images: Optional[List[UploadFile]] = File(None),
        k: int = Form(5),
        mode: str = Form("api"),
        use_functions: bool = Form(True)
):
    logger.info("Endpoint ask called")
    try:
        processed_images = await _process_uploaded_images(images)

        chat_history = _parse_chat_history(history)

        guard_input(message)

        result = run_with_retry_chat(
            current_message=message,
            use_functions=use_functions,
            history=chat_history,
            api_mode=mode,
            images_list=processed_images,
            k=k
        )

        return _format_llm_response(result)
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

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal server error")


async def _process_uploaded_images(files: Optional[List[UploadFile]]) -> List[Dict[str, str]]:
    if not files:
        return []

    processed = []
    for image in files:
        logger.info(f"Processing image: {image.filename}")
        try:
            content = await image.read()
            b64 = base64.b64encode(content).decode('utf-8')
            mime = image.content_type or "image/jpeg"
            processed.append({
                "data": b64,
                "mime": mime
            })
        except Exception as e:
            logger.error(f"Failed to process image {image.filename}: {e}")
            raise ImageProcessingError(f"Failed to process image {image.filename}")

    return processed

def _parse_chat_history(history_json: str) -> List[ChatMessage]:
    if not history_json or not history_json.strip():
        return []

    try:
        raw_data = json.loads(history_json)
        return [ChatMessage(**item) for item in raw_data]
    except JSONDecodeError:
        logger.error("ValidationError: Invalid JSON in history")
        raise InvalidHistoryFormatError("Model returned invalid JSON")
    except Exception as e:
        logger.error(f"History item validation error: {e}")
        raise InvalidHistoryFormatError(f"Invalid history item: {e}")


def _format_llm_response(result: Dict[str, Any]) -> Dict[str, Any]:
    result_type = result.get("type")

    try:
        if result_type == "chat":
            return {
                "status": "chat",
                "message": result["message"],
            }

        if result_type == "report":
            clean_report = scrub_output(result["data"])
            return {
                "status": "complete",
                "report": clean_report,
            }
        raise ValueError(f"Unknown result type: {result_type}")
    except (KeyError, ValueError) as e:
        logger.error(f"Response formatting error: {e}")
        return {
            "status": "chat",
            "message": "I'm sorry, I'm having trouble understanding these symptoms. Could you describe them in more detail or send a photo?",
        }
