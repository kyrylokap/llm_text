from fastapi import HTTPException

class SecurityBlocked(HTTPException):
    def __init__(self, detail="Input blocked by security policy"):
        super().__init__(status_code=400, detail=detail)

class ValidationError(HTTPException):
    def __init__(self, detail="Invalid request or response format"):
        super().__init__(status_code=422, detail=detail)

class ToolError(HTTPException):
    def __init__(self, detail="Upstream model error"):
        super().__init__(status_code=502, detail=detail)

class ToolTimeout(HTTPException):
    def __init__(self, detail="Upstream model timeout"):
        super().__init__(status_code=504, detail=detail)

class EmptyModelOutput(ToolError):
    def __init__(self, detail="Empty model output"):
        super().__init__(detail)

class InvalidHistoryFormatError(HTTPException):
    def __init__(self, detail="Invalid JSON format in history"):
        super().__init__(status_code=422, detail=detail)

class ImageProcessingError(HTTPException):
    def __init__(self, detail="Failed to process uploaded image"):
        super().__init__(status_code=422, detail=detail)