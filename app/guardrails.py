import json
import re
from .errors import SecurityBlocked
from .app_logging import logger

INJECTION_PATTERNS = [
    r"ignore\s+(all|any|previous)\s+instructions",
    r"break\s+(the\s+)?system",
    r"reveal\s+(the\s+)?system",
    r"show\s+(the\s+)?system\s+prompt",
    r"disregard\s+previous",
    r"jailbreak",
    r"developer\s+message",
]

PATH_TRAVERSAL_PATTERN = r"(\.\./)|(\.\.\\)|(\./)+"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def guard_input(text: str) :
    for item in text:
        if re.search(PATH_TRAVERSAL_PATTERN, item):
            logger.error("PATH_TRAVERSAL_PATTERN DETECTED")
            raise SecurityBlocked("Path traversal detected")

        norm_item = normalize(item)
        score = sum(bool(re.search(p, norm_item)) for p in INJECTION_PATTERNS)
        if score >= 2:
            raise SecurityBlocked("Prompt injection detected")


def scrub_output(data, max_items: int = 3):
    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return data[:max_items]

    if not data or not data.strip():
        logger.error("scrub_output received empty string")
        raise ValueError("Empty output")

    if data.startswith("{") and data.endswith("}"):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            pass

    return str(data)
