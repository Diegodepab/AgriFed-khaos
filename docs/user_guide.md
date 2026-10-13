# Guía de Usuario y Manual Operativo: AgriFed-KHAOS

Manual práctico y directo para operar **AgriFed-KHAOS** con **cualquier dataset de visión artificial** (cultivos agrícolas, frutos, piezas industriales, patología vegetal, etc.).

---

## 1. Estructura de Datos Requerida (Agnóstica de Dominio)

El framework consume datasets estructurados en el formato estándar `ImageFolder`. No requiere configuraciones complejas ni bases de datos:

```text
mi_dataset/
├── clase_A/
│   ├── imagen_001.jpg
│   └── imagen_002.jpg
├── clase_B/
│   ├── imagen_003.jpg
│   └── imagen_004.jpg
└── clase_C/
    └── imagen_005.jpg
```

> [!TIP]
> Los nombres de las subcarpetas definen automáticamente las clases del modelo. El framework infiere la cantidad de clases dinámicamente sin necesidad de modificar el código fuente.

---

## 2. Puesta en Marcha del Entorno (Docker)

Para garantizar reproducibilidad y aislar las dependencias de PyTorch, Ultralytics y FastAPI:

```bash
# 1. Configurar variables de entorno iniciales
cp .env.example .env

# 2. Levantar el contenedor de desarrollo
make up
# O directamente: docker compose up -d

# 3. Entrar a la terminal interactiva dentro del contenedor
docker compose exec agrifed bash
```

---

## 3. Guía Paso a Paso del Ciclo Federado

### Paso 1: Entrenar el Modelo Base Oficial (Servidor Maestro)

El servidor central entrena una primera versión de referencia utilizando un dataset inicial.

```bash
python scripts/train_base.py \
  --data-dir /ruta/a/mi_dataset/train \
  --output models/base_model.pt \
  --metadata models/metadata.json \
  --epochs 3 \
  --batch-size 32 \
  --lr 0.001
```

**Artefactos Generados:**
* `models/base_model.pt`: Checkpoint de pesos iniciales de PyTorch.
* `models/metadata.json`: Manifiesto con la arquitectura, lista de clases, huella criptográfica SHA-256 e hiperparámetros.

---

### Paso 2: Servir el Modelo como API REST (Inferencia para Cliente 1)

El servidor maestro expone el modelo mediante una API REST HTTP para que clientes externos (agrónomos, aplicaciones móviles o sistemas de clasificación) puedan evaluar imágenes **sin instalar PyTorch ni clonar el repositorio**.

```bash
python scripts/serve.py \
  --model models/global_model.pt \
  --metadata models/metadata.json \
  --host 0.0.0.0 \
  --port 8000
```

#### ¿Cómo consume el Cliente 1 el servicio?

* **Comprobar estado del servicio:**
  ```bash
  curl http://localhost:8000/health
  ```
  *Respuesta:*
  ```json
  {
    "status": "online",
    "model_path": "models/global_model.pt",
    "model_sha256": "81f1e944747c385c3e0db758810e7b7...",
    "num_classes": 3,
    "classes": ["clase_A", "clase_B", "clase_C"]
  }
  ```

* **Diagnosticar una imagen individual:**
  ```bash
  curl -X POST http://localhost:8000/predict \
    -F "file=@/ruta/a/foto_campo.jpg"
  ```
  *Respuesta:*
  ```json
  {
    "predicted_class": "clase_A",
    "confidence": 0.9421,
    "probabilities": {
      "clase_A": 0.9421,
      "clase_B": 0.0412,
      "clase_C": 0.0167
    },
    "filename": "foto_campo.jpg"
  }
  ```

* **Diagnosticar un lote de imágenes (Batch Inference):**
  ```bash
  curl -X POST http://localhost:8000/predict/batch \
    -F "files=@foto_1.jpg" \
    -F "files=@foto_2.jpg"
  ```

---

### Paso 3: Entrenamiento Local On-Premise (Cliente 2 - Cero Fuga de Datos)

Un cliente colaborador posee imágenes privadas en sus instalaciones y desea contribuir a mejorar el modelo global **sin compartir sus imágenes**.

1. El Cliente 2 recibe únicamente el archivo `models/base_model.pt` y `models/metadata.json`.
2. Ejecuta el entrenamiento local en su máquina utilizando `client_trainer.py`:

```bash
python scripts/client_trainer.py \
  --data-dir /ruta/a/datos_privados_cliente \
  --weights-in models/base_model.pt \
  --weights-out models/client_parameters.pt \
  --metadata models/metadata.json \
  --epochs 2 \
  --mu 0.01 \
  --dp-noise 0.0
```

#### Parámetros Clave:
* `--data-dir` *(Requerido)*: Carpeta local con las imágenes privadas del cliente.
* `--mu` *(Opcional, defecto 0.01)*: Parámetro proximal de FedProx que previene que los pesos locales diverjan excesivamente del modelo base.
* `--dp-noise` *(Opcional, defecto 0.0)*: Desviación estándar de ruido gaussiano para **Privacidad Diferencial** (evita ataques de reconstrucción de imágenes por inversión de gradiente).

**Garantía de Privacidad:**
* **0 bytes de imágenes abandonan el host del cliente.**
* El cliente genera exclusivamente `models/client_parameters.pt` (tensores numéricos) y `models/client_parameters.meta.json` (firma SHA-256).

---

### Paso 4: Auditoría e Incorporación de Parámetros (Servidor Maestro)

El servidor central recibe el archivo `client_parameters.pt` y ejecuta la puerta de seguridad antes de agregarlo:

```bash
python scripts/aggregate.py \
  --base-model models/base_model.pt \
  --client-weights models/client_parameters.pt \
  --output models/global_model.pt \
  --metadata models/metadata.json \
  --eta 0.6 \
  --max-norm 150.0
```

#### ¿Qué ocurre en este paso?
1. [`ParameterValidator`](../src/agrifed_khaos/wrappers.py#L48) verifica:
   - Que coincidan exactamente la cantidad de tensores y sus dimensiones.
   - Que no existan valores `NaN` o `Inf`.
   - Que la distancia euclidiana $\|\Delta W\|_2 \le 150.0$.
2. Si la validación pasa, aplica la agregación ponderada:
   $$W_{\text{global}} = W_{\text{base}} + \eta \cdot (W_{\text{cliente}} - W_{\text{base}})$$
3. Genera el checkpoint actualizado `models/global_model.pt` y actualiza el historial en `metadata.json`.

---

### Paso 5: Evaluación y Diagnóstico Visual (Servidor o Cliente 1)

Permite evaluar cuantitativamente el modelo sobre un conjunto de prueba y generar mapas de atención Grad-CAM:

* **Evaluación completa de un split de prueba (Matriz de Confusión):**
  ```bash
  python scripts/evaluate.py \
    --model models/global_model.pt \
    --metadata models/metadata.json \
    --data-dir /ruta/a/mi_dataset/test \
    --out-dir docs/images
  ```
  Genera el reporte de exactitud, precisión, recall y el gráfico `docs/images/confusion_matrix.png`.

* **Inferencia explicable sobre una imagen individual (Grad-CAM):**
  ```bash
  python scripts/evaluate.py \
    --model models/global_model.pt \
    --metadata models/metadata.json \
    --image /ruta/a/muestra.jpg \
    --out-dir docs/images
  ```
  Genera `docs/images/inference_sample.png` con la predicción, nivel de confianza y el mapa de calor que resalta las regiones determinantes para la red neuronal.

---

## 4. Simulación Rápida de Todo el Ciclo en un Solo Comando

Para validar el flujo completo en un entorno de desarrollo o integración continua:

```bash
python scripts/simulate_federated_nodes.py \
  --dataset-root /ruta/a/mi_dataset \
  --model resnet18 \
  --train-samples-client2 64
```

Este script simula en memoria los 3 nodos (Master, Cliente 1 Tester y Cliente 2 On-Premise Trainer), ejecutando la evaluación $T_0$, el entrenamiento local con FedProx, la auditoría del Parameter Gate y la re-evaluación en $T_1$.

---

## 5. Control de Calidad y Pruebas Automatizadas

Antes de subir cambios al repositorio o desplegar en producción:

```bash
# 1. Comprobación estricta de estilo (Ruff)
docker compose exec agrifed ruff check .

# 2. Comprobación estricta de tipado estático (MyPy)
docker compose exec agrifed mypy src/

# 3. Ejecución de la suite completa de pruebas (Pytest)
docker compose exec agrifed pytest
```

