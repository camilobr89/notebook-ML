"""
Convierte CUALQUIER imagen (PNG, JPG, etc.) a base64 para probar la API.

Uso:
    python imagen_a_base64.py ruta/a/mi_digito.png
"""
import base64
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Uso: python imagen_a_base64.py <ruta_a_imagen>")
        sys.exit(1)

    ruta = Path(sys.argv[1])
    if not ruta.exists():
        print(f"Error: no existe el archivo {ruta}")
        sys.exit(1)

    with open(ruta, "rb") as f:
        raw_bytes = f.read()

    b64 = base64.b64encode(raw_bytes).decode()

    print(f"\nImagen: {ruta.name} ({len(raw_bytes)} bytes)")
    print(f"\n--- JSON listo para Postman ---\n")
    print(f'{{"image_base64": "{b64}"}}')


if __name__ == "__main__":
    main()
