from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, description="Imagen del dígito codificada en base64")


class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    detail: str
