import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .errors import ToolError, EmptyModelOutput
from .prompts import MEDICAL_PROMPT
from .tools import TOOLS
from .dispatcher import execute_tool
from .rag import MiniRAG

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL"),
)

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

rag.load_csv(
    path="medical_rag_pubmed.csv",
    text_columns=["text"],
)

print(f"[RAG] Loaded {rag.index.ntotal} documents")


def run_with_retry_chat(symptoms, **kwargs):
    last = None
    for _ in range(2):
        try:
            return chat_once(symptoms, **kwargs)
        except EmptyModelOutput as e:
            last = e
            time.sleep(0.2)
    raise last


def chat_once(symptoms, use_functions=True, mode="medical", api_mode="api"):

    context = []
    if mode == "medical":
        context = rag.query(symptoms)

    full_prompt = MEDICAL_PROMPT
    if context:
        full_prompt += "\nContext:\n" + "\n".join(context)
    full_prompt += "\n" + symptoms

    t0 = time.time()

    if api_mode == "local":
        out = local_gen(
            full_prompt,
            max_new_tokens=120,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = out[0]["generated_text"]
        text = generated[len(full_prompt):].strip()

        if not text:
            raise EmptyModelOutput()

        return {"text": text, "latency_s": round(time.time() - t0, 3)}

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": MEDICAL_PROMPT},
            {"role": "user", "content": symptoms},
        ],
        functions=[TOOLS[t]["openai_schema"] for t in TOOLS] if use_functions else None,
        function_call="auto" if use_functions else "none",
        timeout=15,
    )

    msg = response.choices[0].message

    if msg.function_call:
        name = msg.function_call.name
        args = eval(msg.function_call.arguments)
        result = execute_tool(name, args)
        return {
            "text": ",".join(result.get("result", [])),
            "latency_s": round(time.time() - t0, 3),
        }

    if not msg.content:
        raise EmptyModelOutput()
    print(msg.content)
    return {"text": msg.content.strip(), "latency_s": round(time.time() - t0, 3)}
