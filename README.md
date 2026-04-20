# MNIST Digit Classifier — Kata Senior ML

Pipeline completo de clasificación de dígitos manuscritos (0–9) sobre **MNIST**, desde entrenamiento hasta consumo vía API.

## Stack elegido y justificación

| Componente | Elección | Razón |
|---|---|---|
| Framework ML | **TensorFlow / Keras** | `model.fit()` entrega `history` con curvas loss/accuracy de forma nativa (requisito). |
| Arquitectura | **CNN simple** (2×Conv + Dense) | ~99% accuracy en MNIST, superior a MLP / sklearn para imágenes. |
| Export | **`.keras`** | Formato oficial Keras 3 — un único archivo (arquitectura + pesos + optimizer). |
| API | **FastAPI + Uvicorn** | Validación Pydantic, docs OpenAPI automáticas, performance async. |
| Preprocesamiento API | **Pillow + NumPy** | Reproduce exactamente el pipeline del notebook: grayscale → 28×28 → `[0,1]`. |
| Tests | **pytest + `TestClient`** | Cubre: imagen válida, base64 inválido, payload vacío, formato no-imagen, campo faltante. |

## Estructura

```
notebook-ML/
├── notebook/mnist_training.ipynb   # pipeline ML completo
├── api/
│   ├── main.py                     # FastAPI app + /predict
│   ├── model_loader.py             # carga cacheada del modelo
│   ├── preprocessing.py            # base64 → tensor
│   └── schemas.py                  # Pydantic models
├── models/mnist_cnn.keras          # modelo exportado (tras correr el notebook)
├── tests/test_api.py
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Entrenar y exportar el modelo

```bash
jupyter notebook notebook/mnist_training.ipynb
```

Ejecuta todas las celdas. El notebook:

1. Carga MNIST desde `keras.datasets`.
2. Normaliza a `[0,1]` y añade canal.
3. Entrena la CNN (10 epochs, ~1–3 min en CPU).
4. Muestra curvas de accuracy/loss, matriz de confusión y 32 predicciones.
5. **Exporta el modelo a `models/mnist_cnn.keras`**.

## 3. Levantar la API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs interactivas: <http://localhost:8000/docs>

## 4. Consumir el endpoint

### Endpoint

```
POST /predict
Content-Type: application/json

{ "image_base64": "<base64 de la imagen>" }
```

Respuesta:

```json
{ "prediction": 7, "confidence": 0.998 }
```

### Ejemplo con `curl`

```bash
# codifica una imagen local
B64=$(base64 -i mi_digito.png)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$B64\"}"
```

### Ejemplo con Python

```python
import base64, requests

with open("mi_digito.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post("http://localhost:8000/predict", json={"image_base64": b64})
print(r.json())  # {'prediction': 7, 'confidence': 0.998}
```

### Errores manejados

| Caso | HTTP | Ejemplo |
|---|---|---|
| Campo ausente | 422 | `{}` |
| `image_base64` vacío | 422 | `{"image_base64": ""}` |
| base64 inválido | 400 | `{"image_base64": "not-b64!!!"}` |
| Bytes válidos pero no son imagen | 400 | base64 de texto plano |

## 5. Tests

```bash
pytest -v
```

> Los tests requieren que el modelo exista en `models/mnist_cnn.keras`. Corre primero el notebook.

## Flujo

```
notebook/mnist_training.ipynb
        │
        ├── carga MNIST
        ├── preprocesa (norm + reshape)
        ├── entrena CNN
        ├── evalúa (curvas + matriz + 32 ejemplos)
        └── exporta → models/mnist_cnn.keras
                           │
                           ▼
            api/main.py  (FastAPI)
                 │
     POST /predict { image_base64 }
                 │
        decode → preprocess → predict
                 │
           { prediction, confidence }
```
