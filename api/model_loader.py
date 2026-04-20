import os
from functools import lru_cache
from pathlib import Path

from tensorflow import keras

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_cnn.keras"


@lru_cache(maxsize=1)
def get_model():
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run the notebook first to train and export the model."
        )
    return keras.models.load_model(model_path)
