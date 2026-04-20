import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_cnn.keras"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model file not found — run the notebook first to export the model.",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def _png_base64_from_array(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mnist_sample():
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    return x_test[0], int(y_test[0])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_image(client):
    arr, real_label = _mnist_sample()
    b64 = _png_base64_from_array(arr)
    r = client.post("/predict", json={"image_base64": b64})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == real_label
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_invalid_base64(client):
    r = client.post("/predict", json={"image_base64": "this-is-not-base64!!!"})
    assert r.status_code == 400


def test_predict_empty_string(client):
    r = client.post("/predict", json={"image_base64": ""})
    # Pydantic rechaza con 422 por min_length=1
    assert r.status_code == 422


def test_predict_valid_base64_but_not_an_image(client):
    not_image = base64.b64encode(b"just some random bytes, not an image").decode("ascii")
    r = client.post("/predict", json={"image_base64": not_image})
    assert r.status_code == 400


def test_predict_missing_field(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422
