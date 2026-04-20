from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from .model_loader import get_model
from .preprocessing import ImageDecodeError, decode_base64_image, preprocess_for_model
from .schemas import ErrorResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: carga el modelo al iniciar para que la primera request no pague ese costo.
    get_model()
    yield


app = FastAPI(
    title="MNIST Digit Classifier API",
    description="Clasifica dígitos manuscritos (0-9) a partir de una imagen en base64.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida"},
        500: {"model": ErrorResponse, "description": "Error interno"},
    },
)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        img = decode_base64_image(payload.image_base64)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    tensor = preprocess_for_model(img)

    model = get_model()
    probs = model.predict(tensor, verbose=0)[0]
    prediction = int(probs.argmax())
    confidence = float(probs[prediction])

    return PredictResponse(prediction=prediction, confidence=confidence)
