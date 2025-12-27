import re
from fastapi import HTTPException

INJECTION_PATTERNS = [
    r"ignore\s+(all|any|previous)\s+instructions",
    r"break\s+(the\s+)?system",
    r"reveal\s+(the\s+)?system",
    r"show\s+(the\s+)?system\s+prompt",
    r"disregard\s+previous",
    r"jailbreak",
    r"developer\s+message",
    r"<\s*system\s*>",
]

PATH_TRAVERSAL_PATTERN = r"(\.\./)+"

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def injection_score(text: str) -> int:
    score = 0
    norm = normalize(text)

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, norm):
            score += 2

    if "ignore" in norm:
        score += 1
    if "instead" in norm:
        score += 1

    return score

def scrub_input(text: str) -> str:
    norm = normalize(text)
    norm = re.sub(
        r"(ignore|break|reveal|show|disregard)\s+.*",
        "",
        norm
    )
    return norm[:500]

def guard_input(text: str) -> str:
    if re.search(PATH_TRAVERSAL_PATTERN, text):
        raise HTTPException(status_code=400, detail="Path traversal detected")

    score = injection_score(text)
    if score >= 5:
        raise HTTPException(status_code=400, detail="Prompt injection detected")
    if score >= 3:
        return scrub_input(text)

    return text

def scrub_output(text: str, max_items: int = 3) -> list[str]:
    if not text or not text.strip():
        raise ValueError("Empty output from model")

    text = text.lower().strip()
    items = re.split(r"[,\n;]", text)
    items = [i.strip() for i in items if 2 < len(i) < 50]

    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)

    if not seen:
        raise ValueError("No valid items found in output")

    return seen[:max_items]
