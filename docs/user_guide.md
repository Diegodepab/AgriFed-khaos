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
* `models/base_model.pt`: Checkpoint bundle autocontenido con `state_dict` y diccionario `metadata` (`classes`, `num_classes`, `img_size`, `architecture`).
* `models/metadata.json`: Espejo auditable en JSON con hiperparámetros y huella SHA-256.

---

### Paso 2: Servir el Modelo como API REST (Inferencia y Diagnóstico)

El servidor maestro expone el modelo mediante una API REST HTTP para que clientes externos o aplicaciones de campo puedan clasificar imágenes **sin instalar PyTorch ni clonar el repositorio**:

```bash
python scripts/serve.py \
  --model models/global_model.pt \
  --host 0.0.0.0 \
  --port 8000
```

> [!NOTE]
> `serve.py` extrae automáticamente la resolución (`img_size`), el número de clases y los umbrales calibrados directamente del bundle `.pt`. Si el checkpoint contiene `calibrated_thresholds` calculados por `evaluate.py`, `/predict` aplica escalado calibrado One-vs-Rest (`probs[c] / tau_c`) para decisiones óptimas en desbalance de clases.

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
    "calibrated": true,
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
  --head-only \
  --allow-new-classes \
  --dp-clip-norm 1.0 \
  --dp-noise-std 0.01 \
  --client-id coop_sevilla
```

#### Parámetros Clave:
* `--data-dir` *(Requerido)*: Carpeta local con las imágenes privadas del cliente. El cargador remapea canónicamente los índices locales contra las clases del modelo base y aplica **hard projection** en clases ausentes para no distorsionar sus pesos.
* `--head-only` *(Opcional, altamente recomendado)*: Congela el extractor de características y entrena únicamente la capa clasificadora final (~18 KB de transferencia frente a 42 MB).
* `--allow-new-classes` *(Opcional)*: Permite al cliente incorporar clases emergentes no vistas en el modelo base mediante inicialización determinista CRC32.
* `--dp-clip-norm` y `--dp-noise-std` *(Opcional)*: Aplica **Privacidad Diferencial formal** (recorte global de norma $L_2$ y mecanismo gaussiano $\mathcal{N}(0, \sigma^2 C^2)$), eximiendo buffers de normalización.
* `--client-id`: Identificador de la cooperativa en el manifiesto y firma criptográfica HMAC.

**Garantía de Privacidad y Cumplimiento Normativo:**
* **0 bytes de imágenes abandonan el host perimetral.**
* Se genera `models/client_parameters.pt` (bundle con `state_dict` y `metadata`) y `models/client_parameters.meta.json` (firma HMAC para auditar `num_samples`).

---

### Paso 4: Auditoría e Incorporación Multi-Cliente (Servidor Maestro)

El servidor central recibe las actualizaciones de uno o varios clientes y ejecuta la agregación ponderada simultánea con momento FedAvgM:

```bash
python scripts/aggregate.py \
  --base-model models/base_model.pt \
  --client-weights models/client_coop1.pt models/client_coop2.pt \
  --output models/global_model.pt \
  --allow-new-classes \
  --server-momentum 0.9 \
  --max-norm 150.0 \
  --max-relative-norm 1.0 \
  --skip-invalid-clients
```

#### ¿Qué ocurre automáticamente en este paso?
1. **Compuerta de Linaje e Integridad Criptográfica:**
   - Comprueba que los hashes de modelo base coincidan exactamente (`base_model_sha256`).
   - Verifica la firma HMAC autenticando que `num_samples` no haya sido manipulado por nodos bizantinos.
2. **Alineación Nominal y Expansión Dinámica de Clases:**
   - Con `--allow-new-classes`, el servidor realiza la unión ordenada de clases y alinea nominalmente cada fila de pesos por nombre de clase, expandiendo el modelo global de forma determinista.
3. **Compuerta de Divergencia Absoluta y Relativa ([`ParameterValidator`](../src/agrifed_khaos/wrappers.py)):**
   - Comprueba finitud numérica (sin `NaN` ni `Inf`).
   - Audita cota absoluta $\|\Delta W\|_2 \le 150.0$ y cota relativa $\|\Delta W\|_2 / \|W_{\text{base}}\|_2 \le 1.0$.
   - `--skip-invalid-clients`: Descarta nodos corruptos o anómalos sin detener la agregación de los válidos.
4. **Agregación Multi-Cliente con Servidor FedAvgM:**
   $$\Delta W_{\text{fed}} = \sum_{k=1}^K \frac{n_k}{N} (W_k - W_{\text{base}})$$
   $$V_{t} = \beta V_{t-1} + \Delta W_{\text{fed}}$$
   $$W_{\text{global}} = W_{\text{base}} + \eta \cdot V_t$$
5. **Frozen BatchNorm:** Preserva 100% invariantes las medias y varianzas de normalización del modelo base.
6. Genera el bundle autocontenido `models/global_model.pt` y actualiza `models/metadata.json`.

---

### Paso 5: Evaluación, Calibración One-vs-Rest y Diagnóstico Visual

* **Evaluación completa con Calibración de Umbrales OvR e Inyección en el Checkpoint:**
  ```bash
  python scripts/evaluate.py \
    --model models/global_model.pt \
    --data-dir /ruta/a/mi_dataset/test \
    --calibrate-thresholds \
    --canary-history docs/canary_history.json \
    --out-dir docs/images
  ```
  Esto calcula los umbrales óptimos One-vs-Rest $\tau_c^*$ para cada clase $c \in \{0, \dots, C-1\}$ que maximizan el F1-Score, inyectándolos dentro del bloque `metadata["calibrated_thresholds"]` del propio archivo `.pt` para uso directo en `serve.py`.

* **Inferencia explicable sobre una imagen individual (Grad-CAM):**
  ```bash
  python scripts/evaluate.py \
    --model models/global_model.pt \
    --image /ruta/a/muestra.jpg \
    --out-dir docs/images
  ```
