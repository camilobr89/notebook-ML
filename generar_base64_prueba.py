"""
Genera un base64 de una imagen MNIST para probar la API.
Uso:
    python generar_base64_prueba.py
    python generar_base64_prueba.py 5   # elige una imagen cuyo dígito real es 5
"""
import base64
import io
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist


def main():
    (_, _), (x_test, y_test) = mnist.load_data()

    if len(sys.argv) > 1:
        target = int(sys.argv[1])
        idx = int(np.where(y_test == target)[0][0])
    else:
        idx = 0

    img_array = x_test[idx]
    real_label = int(y_test[idx])

    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    print(f"\nDígito real en esta imagen: {real_label}")
    print(f"\n--- base64 (copia y pega en Postman / curl) ---\n")
    print(b64)
    print(f"\n--- JSON listo para Postman ---\n")
    print(f'{{"image_base64": "{b64}"}}')


if __name__ == "__main__":
    main()
