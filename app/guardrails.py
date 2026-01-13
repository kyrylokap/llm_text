import re
from .errors import SecurityBlocked

INJECTION_PATTERNS = [
    r"ignore\s+(all|any|previous)\s+instructions",
    r"break\s+(the\s+)?system",
    r"reveal\s+(the\s+)?system",
    r"show\s+(the\s+)?system\s+prompt",
    r"disregard\s+previous",
    r"jailbreak",
    r"developer\s+message",
]

GREETING_PATTERNS = [
    r"\b(hi|hello|hey|cze[śs]ć|hej|witaj|dzień dobry|dobry wieczór)\b",
]

PATH_TRAVERSAL_PATTERN = r"(\.\./)|(\.\.\\)|(\./)+"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_greeting(text: str) -> bool:
    norm = normalize(text)
    return any(re.search(p, norm) for p in GREETING_PATTERNS)


def guard_input(text: str) -> str:
    if re.search(PATH_TRAVERSAL_PATTERN, text):
        raise SecurityBlocked("Path traversal detected")

    norm = normalize(text)
    score = sum(bool(re.search(p, norm)) for p in INJECTION_PATTERNS)

    if score >= 2:
        raise SecurityBlocked("Prompt injection detected")

    return text


def scrub_output(text: str, max_items: int = 3) -> list[str] | str:
    if not text or not text.strip():
        raise ValueError("Empty output")

    if text.count(",") < 2:
        print("text in guardrails fast return", text)
        return text
    text = text.lower()

    items = re.split(r"[,\n;]", text)
    items = [i.strip() for i in items if len(i.strip()) >= 2]

    unique = []
    for i in items:
        if i not in unique:
            unique.append(i)

    if not unique:
        raise ValueError("No valid items")

    return unique[:max_items]
