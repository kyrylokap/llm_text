import time
import os
import json

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from json import JSONDecodeError
from .errors import ToolError, ValidationError, EmptyModelOutput
from .prompts import MEDICAL_PROMPT, MEDICAL_PROMPT_VISION
from .tools import TOOLS
from .dispatcher import execute_tool
from .rag import MiniRAG
from .app_logging import logger

class ChatMessage(BaseModel):
    role: str
    content: str


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
LOCAL_MODEL_NAME = "EleutherAI/gpt-neo-125M"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger.info("INITIALIZED CLIENT")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
local_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

rag = MiniRAG()
logger.info("INITIALIZED RAG")
rag.load_csv(
    path="medical_rag_pubmed.csv",
    text_columns=["text"],
)
rag.load_enriched_csv(
    path="medical_rag_enriched.csv"
)
logger.info("LOADED RAG CSV")


def run_with_retry_chat(current_message: str, **kwargs):
    last_exception = None
    for i in range(2):
        try:
            logger.warning(f"CALLED run_with_retry_chat {i + 1} time")
            return chat_once(current_message, **kwargs)
        except EmptyModelOutput as e:
            logger.error("EmptyModelOutput detected")
            last_exception = e
            time.sleep(0.2)

    if last_exception:
        raise last_exception


def chat_once(
        current_message,
        history: List[ChatMessage],
        image_data=None,
        image_mime="image/jpeg",
        use_functions=True,
        api_mode="api",
        k: int = 5
):
    rag_text = _get_rag_context(current_message, k)

    if api_mode == "local":
        return _run_local_mode(current_message, rag_text)

    logger.info("CALLED API MODE")
    messages = _build_api_messages(history, current_message, rag_text, image_data, image_mime)

    MAX_TURNS = 3
    current_turn = 0

    while current_turn < MAX_TURNS:
        current_turn += 1

        tools_payload = [TOOLS[t]["tool_definition"] for t in TOOLS] if use_functions else None
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_payload,
                tool_choice="auto",
                timeout=30,
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"API Error: {e}")
            raise ToolError(f"Provider Error: {e}")

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            logger.warning("Model didn't use tool, falling back to text content")
            return {
                "type": "chat",
                "message": response_message.content or "I couldn't generate a structured response.",
            }

        logger.info(f"MODEL REQUESTED {len(tool_calls)} TOOL(S)")

        messages.append(response_message)

        for tool_call in tool_calls:
            execution_result = _execute_tool_call(tool_call, messages)

            if execution_result.get("is_final"):
                return _handle_special_tool_response(
                    "provide_response",
                    execution_result["args"],
                )


def _get_rag_context(message: str, k: int) -> str:
    if not message:
        return ""

    context_docs = rag.query(message, k=k)
    rag_text_parts = [
        f"[Source: {doc['source']}] {doc['text']}"
        for doc in context_docs
    ]
    return "\n\n".join(rag_text_parts)


def _run_local_mode(current_message: str, rag_text: str) -> Dict[str, Any]:
    logger.info("CALLED LOCAL MODE")

    full_prompt = (
        f"{MEDICAL_PROMPT}\n"
        f"RAG Context:\n{rag_text}\n"
        f"Patient Description:\n{current_message}"
    )
    try:
        out = local_gen(
            full_prompt,
            max_new_tokens=120,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )

        text = out[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"Local Model Error: {e}")
        text = "I apologize, I am unable to process this request locally."

    if not text:
        logger.error("EmptyModelOutput")
        raise EmptyModelOutput()

    return {
        "type": "chat",
        "message": text,
    }


def _build_api_messages(
        history: List[ChatMessage],
        current_message: str,
        rag_text: str,
        image_data: Optional[str],
        image_mime: str
) -> List[Dict[str, Any]]:
    messages = [{"role": "system", "content": MEDICAL_PROMPT_VISION}]

    for msg in history:
        messages.append({"role": msg.role, "content": str(msg.content)})

    text_payload = f"RAG Context:\n{rag_text}\n\nPatient Description:\n{current_message}"
    user_content = [{"type": "text", "text": text_payload}]

    if image_data:
        logger.info("ATTACHING IMAGE TO LLM REQUEST")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_mime};base64,{image_data}",
                "detail": "auto"
            }
        })

    messages.append({"role": "user", "content": user_content})
    return messages


def _execute_tool_call(tool_call, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    fn_name = tool_call.function.name
    fn_args_json = tool_call.function.arguments
    call_id = tool_call.id

    logger.info(f"EXECUTING TOOL: {fn_name}")

    try:
        args = json.loads(fn_args_json)
    except JSONDecodeError:
        logger.error(f"ValidationError: Invalid JSON args for {fn_name}")
        raise ValidationError("Function call arguments must be valid JSON")

    if fn_name == "provide_response":
        return {"is_final": True, "args": args}

    tool_result = execute_tool(fn_name, args)

    if "error" in tool_result:
        logger.error(f"ToolError in {fn_name}")
        raise ToolError(tool_result["error"])

    messages.append({
        "role": "tool",
        "tool_call_id": call_id,
        "name": fn_name,
        "content": json.dumps(tool_result)
    })

    logger.info("TOOL EXECUTED. FEEDING RESULT BACK TO LLM...")
    return {"is_final": False}


def _handle_special_tool_response(fn_name: str, args: dict) -> Optional[Dict[str, Any]]:
    if fn_name != "provide_response":
        return None

    if args.get("action") == "message":
        return {
            "type": "chat",
            "message": args.get("message_to_patient"),
        }
    elif args.get("action") == "final_report":
        return {
            "type": "report",
            "data": args.get("report_data"),
        }
    return None
