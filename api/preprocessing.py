import base64
import binascii
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """Raised when the incoming base64 string cannot be turned into a valid image."""


def decode_base64_image(image_base64: str) -> Image.Image:
    if not image_base64 or not image_base64.strip():
        raise ImageDecodeError("image_base64 is empty")

    # Acepta data URIs: "data:image/png;base64,...."
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("invalid base64 string") from exc

    if not raw:
        raise ImageDecodeError("decoded image is empty")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("unsupported or corrupt image format") from exc

    return img


def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL al mismo formato que el modelo vio en el notebook:
    grayscale 28x28, normalizada a [0, 1], shape (1, 28, 28, 1).

    Detecta fondo claro (típico de fotos/escaneos) e invierte para que
    el trazo quede en blanco sobre fondo negro, como MNIST.
    """
    img = img.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32)

    # MNIST: fondo negro (~0), trazo blanco (~255).
    # Si la media de píxeles es alta, asumimos fondo claro e invertimos.
    if arr.mean() > 127.0:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr
