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

### Paso 1: Entrenar el Modelo Base Oficial con Auto-FL (Servidor Maestro)

El servidor central entrena una primera versión de referencia seleccionando automáticamente la arquitectura adecuada según el perfil de los clientes periféricos y las dimensiones del dataset:

```bash
python scripts/train_base.py \
  --data-dir /ruta/a/mi_dataset/train \
  --arch auto \
  --target-client-tier cpu-high \
  --output models/base_model.pt \
  --metadata models/metadata.json \
  --epochs 3 \
  --batch-size 32 \
  --lr 0.001
```

#### Opciones Clave de Auto-FL:
* `--arch auto` *(Defecto)*: Utiliza [`ModelSelector`](../src/agrifed_khaos/hardware.py) para auditar el hardware objetivo de los clientes periféricos y el tamaño del dataset, seleccionando entre `mobilenetv4`, `resnet18` o `convnext`.
* `--target-client-tier {cpu-low, cpu-high, gpu}`:
  - `cpu-low`: Cooperativas con CPUs modestas $\rightarrow$ `mobilenetv4_conv_small`.
  - `cpu-high`: Servidores locales en cooperativas $\rightarrow$ `resnet18`.
  - `gpu`: Estaciones de trabajo dedicadas $\rightarrow$ `convnext_tiny`.

**Artefactos Generados:**
* `models/base_model.pt`: Checkpoint de pesos iniciales de PyTorch.
* `models/metadata.json`: Manifiesto con la arquitectura seleccionada, clases descubiertas, huella criptográfica SHA-256 e hiperparámetros.

---

### Paso 2: Servir el Modelo como API REST (Inferencia para Cliente 1)

El servidor maestro expone el modelo mediante una API REST HTTP para que clientes externos puedan evaluar imágenes **sin instalar PyTorch ni clonar el repositorio**:

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

---

### Paso 3: Entrenamiento Local On-Premise (Clientes Colaboradores - Cero Fuga de Datos)

Un cliente colaborador posee imágenes privadas y entrena localmente:

```bash
# Modo Transfer Learning Federado (Recomendado: ultra-rápido, ~18 KB de transferencia)
python scripts/client_trainer.py \
  --data-dir /ruta/a/datos_privados_cliente \
  --weights-in models/base_model.pt \
  --weights-out models/client_parameters.pt \
  --metadata models/metadata.json \
  --head-only \
  --noise-std 0.01 \
  --client-id coop_sevilla
```

#### Parámetros Clave:
* `--data-dir` *(Requerido)*: Carpeta local con las imágenes privadas del cliente.
* `--head-only` *(Opcional, altamente recomendado)*: Congela el extractor de características y entrena únicamente la capa clasificadora final (~18 KB de transferencia frente a 42 MB).
* `--noise-std` *(Opcional, alias `--param-noise`)*: Aplica **ruido gaussiano adaptativo escalado por RMS por capa**, garantizando privacidad sin destruir capas de baja magnitud (`bias`).
* `--client-id`: Identificador de la cooperativa en el manifiesto y firma criptográfica.

**Garantía de Privacidad y Cumplimiento Normativo:**
* **0 bytes de imágenes abandonan el host perimetral.**
* Se genera `models/client_parameters.pt` (tensores numéricos) y `models/client_parameters.meta.json` (linaje, recuento `num_samples` y firma HMAC).

---

### Paso 4: Auditoría e Incorporación Multi-Cliente (Servidor Maestro)

El servidor central recibe las actualizaciones de uno o varios clientes y ejecuta la agregación ponderada simultánea:

```bash
python scripts/aggregate.py \
  --base-model models/base_model.pt \
  --client-weights models/client_coop1.pt models/client_coop2.pt \
  --output models/global_model.pt \
  --metadata models/metadata.json \
  --max-norm 150.0 \
  --max-relative-norm 1.0 \
  --skip-invalid-clients
```

#### ¿Qué ocurre automáticamente en este paso?
1. **Compuerta de Linaje e Integridad Criptográfica:**
   - Comprueba que los hashes de modelo base coincidan exactamente (`base_model_sha256`).
   - Verifica la firma HMAC autenticando que `num_samples` no haya sido manipulado.
2. **Compuerta Estructural y Esquema Cerrado:**
   - Valida compatibilidad con `head_only` o `full` sin tolerar claves espurias.
3. **Compuerta de Divergencia Absoluta y Relativa ([`ParameterValidator`](../src/agrifed_khaos/wrappers.py)):**
   - Comprueba finitud numérica (sin `NaN` ni `Inf`).
   - Audita cota absoluta $\|\Delta W\|_2 \le 150.0$.
   - Audita cota relativa $\|\Delta W\|_2 / \|W_{\text{base}}\|_2 \le 1.0$ (protección ideal para `--head-only`).
   - `--skip-invalid-clients`: Descarta nodos corruptos o anómalos sin detener la agregación de los válidos.
4. **Agregación Multi-Cliente Ponderada por Muestras (`FedOpt`):**
   $$\Delta W_{\text{fed}} = \sum_{k=1}^K \frac{n_k}{N} (W_k - W_{\text{base}})$$
   $$W_{\text{global}} = W_{\text{base}} + \eta \cdot \Delta W_{\text{fed}}$$
5. **Frozen BatchNorm:** Preserva 100% invariantes las medias y varianzas de normalización del modelo base.
6. Actualiza el historial `aggregation_history` en `metadata.json` y guarda `models/global_model.pt`.

---

### Paso 5: Evaluación y Diagnóstico Visual (Servidor o Cliente 1)

* **Evaluación completa de un split de prueba con Canario de Observabilidad:**
  ```bash
  python scripts/evaluate.py \
    --model models/global_model.pt \
    --metadata models/metadata.json \
    --data-dir /ruta/a/mi_dataset/test \
    --canary-history docs/canary_history.json \
    --warn-drop 0.08 \
    --critical-drop 0.20 \
    --out-dir docs/images
  ```

* **Inferencia explicable sobre una imagen individual (Grad-CAM):**
  ```bash
  python scripts/evaluate.py \
    --model models/global_model.pt \
    --metadata models/metadata.json \
    --image /ruta/a/muestra.jpg \
    --out-dir docs/images
  ```
