import asyncio
import concurrent.futures
from .tools import TOOLS

async def execute_tool(name: str, arguments: dict, timeout: int = 3):
    if name not in TOOLS:
        return {"error": "tool_not_allowed"}

    tool = TOOLS[name]

    try:
        validated = tool["args_schema"](**arguments)
    except Exception as e:
        return {"error": "validation_error", "details": str(e)}

    impl = tool["implementation"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: impl(**validated.dict()))
        try:
            result = await asyncio.to_thread(impl, **validated.dict())
            return {"ok": True, "result": result}
        except concurrent.futures.TimeoutError:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": "tool_failed", "details": str(e)}