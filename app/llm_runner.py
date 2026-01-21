import time
import os
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from json import JSONDecodeError
from .errors import ToolError, ValidationError
from .errors import EmptyModelOutput
from .prompts import MEDICAL_PROMPT, MEDICAL_PROMPT_VISION
from .schemas import AgentResponse
from .tools import TOOLS
from .dispatcher import execute_tool
from .rag import MiniRAG
from .app_logging import logger

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger.info("INITIALIZED CLIENT")

MODEL_NAME = os.getenv("MODEL_NAME")

LOCAL_MODEL_NAME = "EleutherAI/gpt-neo-125M"
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

logger.info("LOADED RAG CSV")


def run_with_retry_chat(current_message: str, **kwargs):
    last = None
    for i in range(2):
        try:
            logger.warning(f"CALLED run_with_retry_chat {i} time")
            return chat_once(current_message, **kwargs)
        except EmptyModelOutput as e:
            logger.error("EmptyModelOutput")

            last = e
            time.sleep(0.2)
    raise last


def chat_once(current_message, history: list, image_data=None, use_functions=True, api_mode="api"):
    context = []
    if current_message:
        context.extend(rag.query(current_message))

    rag_text = "\n".join(context[:5])

    t0 = time.time()

    if api_mode == "local":
        logger.info("CALLED LOCAL MODE")

        full_prompt = MEDICAL_PROMPT
        full_prompt += (
                "\nRAG Context:\n"
                + "\n".join(rag_text)
                + "\nPatient Description:\n"
                + "\n".join(current_message)
        )

        out = local_gen(
            full_prompt,
            max_new_tokens=120,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        text = out[0]["generated_text"].strip()

        if not text:
            logger.error("EmptyModelOutput")

            raise EmptyModelOutput()

        return {"text": text, "latency_s": round(time.time() - t0, 3)}


    logger.info("CALLED API MODE")

    messages = [{"role": "system", "content": MEDICAL_PROMPT_VISION}]
    for msg in history:
        messages.append({"role": msg["role"], "content": str(msg["content"])})

    user_content = []
    text_payload = f"RAG Context:\n{rag_text}\n\nPatient Description:\n{current_message}"
    user_content.append({"type": "text", "text": text_payload})

    if image_data:
        logger.info("ATTACHING IMAGE TO LLM REQUEST")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
                "detail": "auto"
            }
        })

    messages.append({"role": "user", "content": user_content})



    MAX_TURNS = 3
    current_turn = 0

    while current_turn < MAX_TURNS:
        current_turn += 1

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            functions=[TOOLS[t]["openai_schema"] for t in TOOLS] if use_functions else None,
            function_call="auto" if use_functions else "none",
            timeout=30,
            temperature=0.3,
        )

        msg = response.choices[0].message

        if hasattr(msg, "function_call") and msg.function_call:
            logger.info("USED FUNCTION CALL")

            fn_name = msg.function_call.name
            fn_args_json = msg.function_call.arguments

            try:
                args = json.loads(fn_args_json)

            except JSONDecodeError:
                logger.error("ValidationError")
                raise ValidationError("Function call arguments must be valid JSON")

            tool_result = execute_tool(fn_name, args)

            if "error" in tool_result:
                logger.error("ToolError")
                raise ToolError(tool_result["error"])

            messages.append({
                "role": "assistant",
                "function_call": {
                    "name": fn_name,
                    "arguments": fn_args_json
                },
                "content": None
            })

            messages.append({
                "role": "function",
                "name": fn_name,
                "content": json.dumps(tool_result)
            })

            logger.info("TOOL EXECUTED. FEEDING RESULT BACK TO LLM...")
            continue

        content = msg.content
        if not content:
            raise EmptyModelOutput()

        try:
            data = json.loads(content)
        except JSONDecodeError:
            logger.error("ValidationError")
            return {
                "type": "chat",
                "message": content,
                "latency_s": round(time.time() - t0, 3),
            }

        try:
            agent_response = AgentResponse(**data)
        except Exception as e:
            raise ValidationError(str(e))

        if agent_response.action == "question":
            return {
                "type": "chat",
                "message": agent_response.message_to_patient,
                "latency_s": round(time.time() - t0, 3),
            }

        elif agent_response.action == "final_report":
            return {
                "type": "report",
                "data": agent_response.report_data.model_dump(),
                "latency_s": round(time.time() - t0, 3),
            }
