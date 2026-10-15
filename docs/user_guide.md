# Guía de Usuario: AgriFed-KHAOS

Manual operativo de **AgriFed-KHAOS** para aplicar Aprendizaje Federado sobre **cualquier dataset de visión artificial** estructurado en carpetas estándar.

Esta guía está organizada en torno a los **tres roles del sistema**, de modo que cada usuario consulte únicamente las instrucciones que le corresponden sin sobrecarga técnica innecesaria.

---

## 1. Estructura de Datos Universal (ImageFolder)

El framework es 100% agnóstico del cultivo o tipo de imagen. Solo requiere que las imágenes estén organizadas en subcarpetas según su clase:

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

*Los nombres de las subcarpetas definen automáticamente las clases. El sistema infiere el número de clases y adapta las resoluciones dinámicamente.*

---

## 2. Rol 1: Administrador del Sistema (Dueño del Aprendizaje Federado)

El Administrador gestiona el servidor central, entrena el modelo base inicial, expone la API de diagnóstico y fusiona las actualizaciones de los colaboradores.

### Paso 1: Puesta en Marcha del Entorno
```bash
# Iniciar contenedor Docker reproducible
cp .env.example .env
make up
docker compose exec agrifed bash
```

### Paso 2: Entrenar el Modelo Base Oficial
Siguiendo la filosofía de mínima fricción, solo debe indicar dónde están las imágenes iniciales:

```bash
python scripts/train_base.py --data-dir mi_dataset/train
```

*El comando selecciona automáticamente la arquitectura óptima (`auto`), infiere las clases, entrena el extractor de características y guarda el bundle autocontenido `models/base_model.pt`.*

<details>
<summary>Opciones de personalización avanzada (opcional)</summary>

* `--arch {auto, resnet18, resnet50, mobilenetv4_conv_small}`: Fuerza una arquitectura concreta (defecto: `auto`).
* `--target-client-tier {cpu-low, cpu-high, gpu}`: Perfil de hardware de los clientes perimetrales (defecto: `cpu-low`).
* `--epochs <int>`: Épocas de entrenamiento (defecto: 3).
* `--batch-size <int>`: Tamaño del lote (defecto: 32).
* `--lr <float>`: Tasa de aprendizaje (defecto: 0.001).
* `--img-size <int>`: Resolución de entrada en píxeles (defecto: 224).
* `--output <path>`: Ruta del checkpoint resultante (defecto: `models/base_model.pt`).
</details>

### Paso 3: Levantar el Servicio de Inferencia REST
Expone el modelo para que cualquier técnico o aplicación de campo pueda clasificar fotos:

```bash
python scripts/serve.py
```

*Inicia la API REST en `http://0.0.0.0:8000`. Carga automáticamente `models/global_model.pt` (o `base_model.pt`) con inferencia calibrada One-vs-Rest.*

### Paso 4: Fusionar Actualizaciones de Colaboradores (Agregación Federada)
Cuando recibe los archivos de parámetros de los colaboradores (ej. `coop1.pt`, `coop2.pt`), los fusiona en el modelo global:

```bash
python scripts/aggregate.py --client-weights models/coop1.pt models/coop2.pt
```

*Aplica compuertas de seguridad contra anomalías (`ParameterValidator`), calcula el promedio ponderado por muestras con momento FedAvgM y actualiza `models/global_model.pt`.*

<details>
<summary>Opciones avanzadas de agregación</summary>

* `--server-momentum <float>`: Factor de momento del servidor FedAvgM (defecto: 0.9).
* `--allow-new-classes`: Permite expandir dinámicamente el clasificador si los colaboradores descubrieron patologías nuevas.
* `--max-norm <float>`: Límite de norma L2 para rechazar actualizaciones anómalas (defecto: 150.0).
* `--skip-invalid-clients`: Si un cliente no pasa las compuertas de seguridad, lo descarta y continúa con los válidos.
</details>

### Paso 5: Evaluar y Calibrar el Modelo Global
Evalúa el rendimiento sobre un split de prueba independiente, optimiza los umbrales de decisión One-vs-Rest y genera mapas explicables Grad-CAM:

```bash
python scripts/evaluate.py --data-dir mi_dataset/test
```

---

## 3. Rol 2: Usuario Consumidor (Diagnóstico de Campo)

El Consumidor es un técnico o productor en el campo que únicamente desea clasificar fotos y obtener un diagnóstico fiable en tiempo real.

### Requisitos:
* **No necesita clonar este repositorio.**
* **No necesita instalar Python ni PyTorch.**
* Solo necesita conectividad de red con el Servidor Maestro.

### ¿Cómo clasificar una imagen?
Con una simple petición HTTP (desde terminal `curl`, aplicación móvil o navegador):

```bash
curl -X POST http://<IP_O_DOMINIO_SERVIDOR>:8000/predict \
  -F "file=@foto_campo.jpg"
```

### Respuesta recibida:
```json
{
  "predicted_class": "repilo",
  "confidence": 0.9421,
  "probabilities": {
    "sano": 0.0412,
    "repilo": 0.9421,
    "cochinilla": 0.0167
  },
  "calibrated": true,
  "filename": "foto_campo.jpg"
}
```

---

## 4. Rol 3: Usuario Colaborador (Cooperativa con Datos Privados)

El Colaborador posee imágenes valiosas tomadas en sus instalaciones, pero por normativas de privacidad (RGPD) o secreto comercial **no puede ni quiere subir sus fotos a ningún servidor central**.

Acepta colaborar en el entrenamiento federado con una condición innegociable: **0 bytes de fotos saldrán jamás de su red local**.

### ¿Qué necesita el Colaborador?
1. Recibir del Administrador el archivo del modelo base (`base_model.pt`).
2. Tener en su disco local una carpeta con sus fotos privadas organizadas por clase (ej. `/mis_fotos/`).

### ¿Cómo ejecuta el entrenamiento local?

#### Opción A: Mediante Docker (Recomendado, sin instalar dependencias)
Si dispone de Docker, no necesita instalar librerías de Python. Ejecuta un único comando montando su carpeta de fotos:

```bash
docker run --rm \
  -v /ruta/a/mis_fotos:/data \
  -v /ruta/a/modelos:/models \
  agrifed \
  python scripts/client_trainer.py --data-dir /data --weights-in /models/base_model.pt --weights-out /models/coop_pesos.pt
```

#### Opción B: Con entorno Python local
Si prefiere ejecutar directamente en su máquina con Python:

```bash
python scripts/client_trainer.py --data-dir /ruta/a/mis_fotos/
```

### ¿Qué artefacto se genera y qué devuelve al Administrador?
* El script genera un único archivo ligero de parámetros numéricos: `models/client_parameters.pt` (de apenas ~18 KB en modo clasificador).
* El colaborador envía **únicamente** ese archivo `.pt` al Administrador (por correo, SFTP o API segura).
* **Garantía Técnica Verificable:** Ninguna imagen abandona su disco. El archivo `.pt` solo contiene pesos matemáticos de la red neuronal, auditados y protegidos con ruido de Privacidad Diferencial.

<details>
<summary>Opciones de privacidad y entrenamiento avanzado para el colaborador</summary>

* `--head-only`: *(Recomendado por defecto)* Entrena solo la capa clasificadora final (~18 KB frente a 42 MB de transferencia).
* `--dp-clip-norm <float>` y `--dp-noise-std <float>`: Aplica Privacidad Diferencial formal mediante recorte de norma L2 y ruido gaussiano sobre los pesos.
* `--allow-new-classes`: Permite al cliente entrenar clases nuevas no vistas en el modelo base mediante inicialización determinista CRC32.
* `--client-id <nombre>`: Identificador del colaborador en el manifiesto criptográfico HMAC.
</details>
