# Guía de Usuario: AgriFed-KHAOS

> Versión en español. English version available in [docs/user_guide.en.md](user_guide.en.md).

Manual operativo de **AgriFed-KHAOS** para entrenar, servir y federar modelos de visión artificial sobre **cualquier dataset estructurado en carpetas estándar**.

---

## 1. Estructura de Datos Universal (ImageFolder)

El framework es agnóstico del cultivo o tipo de imagen. Requiere que las imágenes estén organizadas en subcarpetas según su clase:

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

<details>
<summary><strong>Cómo ubicar tus carpetas para trabajar con Docker o Python local</strong></summary>

En despliegues basados en Docker, el contenedor monta el directorio actual del repositorio en `/workspace`:
* **Recomendado:** Coloca la carpeta de datos directamente dentro del proyecto, por ejemplo en `dataset/` o `mi_dataset/`. Así, dentro del contenedor la ruta será inmediatamente `dataset/train` o `mi_dataset/train`.
* **Si tus datos están fuera del proyecto (ej. <code>/home/usuario/mis_imagenes/</code>):**
  - *Opción A (Copiar dentro del proyecto):* `cp -r /home/usuario/mis_imagenes ./dataset`
  - *Opción B (Montar como volumen externo):* En `compose.yaml` puedes mapear la ruta externa al contenedor:
    ```yaml
    volumes:
      - .:/workspace
      - /home/usuario/mis_imagenes:/data:ro
    ```
    Y dentro del contenedor indicarás `--data-dir /data/train`.
</details>

---

## 2. Rol 1: Administrador del Sistema (Servidor Maestro)

El Administrador gestiona el servidor central, entrena el modelo base inicial, expone los servicios de inferencia y ejecuta la agregación federada de la red.

### Paso 1: Puesta en Marcha del Entorno
```bash
cp .env.example .env
make up
docker compose exec agrifed bash
```

### Paso 2: Entrenar el Modelo Base Oficial
Indique la carpeta que contiene las imágenes del entrenamiento inicial:

```bash
python scripts/train_base.py --data-dir mi_dataset/train
```

*Selecciona automáticamente la mejor arquitectura (`auto`), infiere las clases, entrena el modelo y genera el bundle autocontenido `models/base_model.pt`.*

<details>
<summary>Opciones de personalización avanzada</summary>

* `--arch {auto, resnet18, resnet50, mobilenetv4_conv_small}`: Fuerza una arquitectura concreta (defecto: `auto`).
* `--target-client-tier {cpu-low, cpu-high, gpu}`: Perfil de hardware de los clientes perimetrales (defecto: `cpu-low`).
* `--epochs <int>`: Épocas de entrenamiento (defecto: 3).
* `--batch-size <int>`: Tamaño del lote (defecto: 32).
* `--lr <float>`: Tasa de aprendizaje (defecto: 0.001).
* `--img-size <int>`: Resolución de entrada en píxeles (defecto: 224).
* `--output <path>`: Ruta del checkpoint resultante (defecto: `models/base_model.pt`).
</details>

### Paso 3: Levantar el Servicio de Inferencia y Recepción REST
Inicia el microservicio HTTP tanto para atender diagnósticos de campo como para recibir actualizaciones automáticas de los colaboradores:

```bash
python scripts/serve.py
```

*Escucha en `http://0.0.0.0:8000`. Carga automáticamente `models/global_model.pt` (o `base_model.pt`) con inferencia calibrada One-vs-Rest y habilita el endpoint de recepción `/submit-update`.*

### Paso 4: Fusión de Parámetros con Servidor FedAvgM
El Administrador fusiona las actualizaciones recibidas de los colaboradores utilizando el algoritmo **FedAvgM (Federated Averaging con Momento del Servidor)**:

```bash
python scripts/aggregate.py --client-weights models/client_updates/*.pt
```

* **Momento del Servidor ($\beta = 0.9$):** A diferencia de un promedio aritmético pasivo, FedAvgM mantiene un buffer de velocidad $V_t = \beta V_{t-1} + \Delta W_{\text{fed}}$, estabilizando y acelerando la trayectoria de convergencia global frente a la heterogeneidad extrema de los datos de cada agricultor.
* **Compuertas de Seguridad ([`ParameterValidator`](../src/agrifed_khaos/wrappers.py)):** Audita finitud numérica (sin `NaN` ni `Inf`), cotas de divergencia L2 absolutas y relativas, y autenticidad criptográfica HMAC.

<details>
<summary>Opciones avanzadas de agregación</summary>

* `--server-momentum <float>`: Factor de momento del servidor $\beta$ en $[0, 1)$ (defecto: 0.9).
* `--eta <float>`: Tasa de aprendizaje del servidor (defecto: 0.6).
* `--allow-new-classes`: Permite expandir dinámicamente el clasificador si los colaboradores descubrieron patologías nuevas registradas en la taxonomía global.
* `--max-norm <float>`: Límite de norma L2 para rechazar actualizaciones anómalas (defecto: 150.0).
* `--skip-invalid-clients`: Si un cliente no supera las compuertas de seguridad, lo descarta y continúa con los válidos.
</details>

### Paso 5: Evaluación y Calibración del Modelo Global
Evalúa el rendimiento sobre un split de prueba independiente, optimiza los umbrales de decisión One-vs-Rest ($\tau_c^*$) y genera mapas explicables Grad-CAM:

```bash
python scripts/evaluate.py --data-dir mi_dataset/test
```

---

## 3. Rol 2: Usuario Consumidor (Diagnóstico de Campo)

El Consumidor es un técnico o agricultor que necesita clasificar imágenes de campo y obtener un diagnóstico fiable con certidumbre calibrada.

### Modalidad A: Inferencia Remota Centralizada (API REST)
Diseñada para técnicos con conectividad de red que desean diagnosticar fotos sin instalar librerías de IA:
* **Requisitos:** No necesita clonar el repositorio ni instalar PyTorch. Solo necesita conectividad con el Servidor Maestro.

```bash
curl -X POST http://<IP_O_DOMINIO_SERVIDOR>:8000/predict \
  -F "file=@foto_campo.jpg"
```

**Respuesta recibida:**
```json
{
  "predicted_class": "repilo",
  "confidence": 0.9421,
  "probabilities": {
    "sano": 0.0412,
    "repilo": 0.9421,
    "cochinilla": 0.0167
  },
  "threshold_applied": "calibrated",
  "filename": "foto_campo.jpg"
}
```

### Modalidad B: Inferencia Perimetral Desconectada (Offline-First)
Diseñada para técnicos en zonas rurales sin cobertura móvil:
* El técnico descarga el bundle autocontenido `models/global_model.pt`.
* La inferencia se ejecuta 100% en local en el dispositivo perimetral (laptop, tablet o app embebida con ONNX Runtime / TFLite) sin requerir conexión a internet.

---

## 4. Rol 3: Usuario Colaborador (Cooperativa con Datos Privados)

El Colaborador posee imágenes privadas tomadas en sus explotaciones agrícolas. Por normativas de privacidad (RGPD) o secreto comercial, **no permite la salida de ninguna imagen de su red local**.

Colabora en el entrenamiento federado mediante un agente local automatizado que intercambia únicamente parámetros matemáticos.

### ¿Cómo opera el Colaborador en la práctica?

<details open>
<summary><strong>Guía Paso a Paso para la Cooperativa desde Cero</strong></summary>

Si tu cooperativa se une a la red de colaboración federada desde una máquina limpia:

1. **Clonar el proyecto y preparar el entorno:**
   ```bash
   git clone https://github.com/Diegodepab/AgriFed-khaos.git
   cd AgriFed-khaos
   cp .env.example .env
   make up
   docker compose exec agrifed bash
   ```

2. **Descargar los pesos del modelo base oficial:**
   El Administrador del Servidor Maestro publica el modelo base vía HTTP. Descárgalo directamente a la carpeta `models/`:
   ```bash
   mkdir -p models
   curl -o models/base_model.pt http://<IP_SERVIDOR_MAESTRO>:8000/download-model
   ```

3. **Ejecutar el entrenamiento privado local y enviar parámetros:**
   Indica la ruta de tus fotos locales (ej. `mis_fotos_privadas/` o `dataset/val/`):
   ```bash
   python scripts/client_trainer.py \
     --data-dir mis_fotos_privadas/ \
     --weights-in models/base_model.pt \
     --head-only \
     --client-id coop_mi_zona \
     --server-url http://<IP_SERVIDOR_MAESTRO>:8000
   ```

*El proceso tarda menos de 1 minuto, actualiza únicamente la capa clasificadora (~18 KB), y envía el paquete con firma HMAC al servidor central sin que ninguna foto salga de tus instalaciones.*
</details>

#### Ejecución Automatizada con Docker (Recomendado):
El contenedor corre de forma desatendida, entrena sobre los datos privados locales y transmite los parámetros por streaming HTTP al servidor central:

```bash
docker run --rm \
  -v /ruta/a/mis_fotos:/data \
  agrifed \
  python scripts/client_trainer.py \
    --data-dir /data \
    --server-url http://<IP_SERVIDOR_MAESTRO>:8000
```

#### Ejecución con Python local:
```bash
python scripts/client_trainer.py \
  --data-dir /ruta/a/mis_fotos/ \
  --server-url http://<IP_SERVIDOR_MAESTRO>:8000
```

### Garantías Técnicas de Privacidad:
* **0 Bytes de Imágenes Transferidas:** El script extrae gradientes localmente y genera únicamente el archivo numérico `models/client_parameters.pt` (~18 KB en modo clasificador).
* **Transmisión Automatizada:** Al concluir las épocas locales, el agente transmite los tensores directamente al endpoint seguro `/submit-update` (o por streaming gRPC de Flower en memoria viva). El operador de la cooperativa no manipula archivos manualmente.
* **Privacidad Diferencial Formal:** Opcionalmente recorta la norma L2 y añade ruido gaussiano ($\mathcal{N}(0, \sigma^2 C^2)$), impidiendo cualquier ataque de reconstrucción inversa de imágenes.

> [!WARNING]
> **Advertencia Crítica de Gobernanza y Alineación de Clases (`--allow-new-classes`):**
> En visión artificial federada bajo distribuciones No-IID, la incorporación de nuevas clases dinámicas requiere que todas las cooperativas respeten una **taxonomía nominal canónica compartida** (ej. nombres estandarizados como `repilo`, `antracnosis`, `cochinilla`). Si dos entidades utilizan nombres distintos para la misma patología o asignan el mismo nombre a clases biológicamente distintas, se producirán colisiones de etiquetas y olvido catastrófico (*catastrophic forgetting*). El servidor alinea nominalmente por nombre de clase mediante inicialización determinista CRC32, por lo que la coherencia en la nomenclatura de carpetas es un requisito indispensable de gobernanza.

<details>
<summary>Opciones avanzadas para el colaborador</summary>

* `--head-only`: *(Activado por defecto)* Entrena solo el clasificador final (~18 KB frente a 42 MB de transferencia).
* `--server-url <url>`: URL del servidor central para transmisión automatizada de la actualización.
* `--dp-clip-norm <float>` y `--dp-noise-std <float>`: Aplica Privacidad Diferencial formal sobre los pesos.
* `--allow-new-classes`: Permite registrar clases emergentes respetando la taxonomía global de AgriFed.
* `--client-id <nombre>`: Identificador del colaborador en el manifiesto y firma criptográfica HMAC.
</details>
