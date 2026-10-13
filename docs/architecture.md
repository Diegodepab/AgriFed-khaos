# Arquitectura del Sistema: AgriFed-KHAOS

Documento de especificación técnica y decisiones arquitectónicas de **AgriFed-KHAOS**, framework modular de Aprendizaje Federado (*Federated Learning*, FL) orientado a Visión Artificial.

---

## 1. Principios de Diseño y Estándares SOLID

El framework está concebido bajo estándares de ingeniería de software y MLOps modernos:

1. **Agnosticismo de Dominio (Domain-Agnostic Core):** El núcleo del framework no conoce ni depende de ningún cultivo, planta o patología específica. Opera sobre tensores arbitrarios de imágenes agrupadas en estructuras de clases estándar (`ImageFolder`) o manifiestos anotados.
2. **Desacoplamiento Data-Centric (Principio de Inversión de Dependencias - DIP):** La curación, filtrado de calidad y normalización de datos corresponden exclusivamente al repositorio hermano [`agrivision-khaos`](../../agrivision-khaos). `AgriFed-KHAOS` asume que los datos locales del cliente son un estándar de oro y se concentra únicamente en la optimización distribuida y la agregación de pesos.
3. **Patrón Adaptador (Adapter Pattern - Principio Abierto/Cerrado - OCP):** La abstracción [`BaseFLWrapper`](../src/agrifed_khaos/wrappers.py#L119) desacopla por completo el runtime de comunicación federada (Flower / tensores NumPy) de los frameworks subyacentes de deep learning (PyTorch, Timm, Ultralytics YOLO).
4. **Cero Fuga de Datos (Zero Data Leakage Contract):** Los datos físicos en bruto (imágenes RGB y etiquetas locales) jamás abandonan el host perimetral del cliente. La red solo transporta parámetros matemáticos y firmas criptográficas.

---

## 2. Topología del Sistema y Modelo de Tres Actores

```mermaid
flowchart TD
    subgraph MasterHost ["SERVIDOR MAESTRO (Orquestador & Proveedor de Servicio)"]
        W_Base["Modelo Base Oficial (models/base_model.pt)<br/>Firma Criptográfica SHA-256"]
        Metadata["Registro de Linaje (models/metadata.json)<br/>Historial Inmutable de Rondas"]
        Gate["Quality & Security Gate (ParameterValidator)<br/>• Finitud (sin NaNs/Infs)<br/>• Coincidencia Estructural<br/>• Cota L2: ||ΔW||₂ ≤ τ"]
        Aggregator["Agregador SanitizedFedProx<br/>W_global = W_base + η · (W_client - W_base)"]
        W_Global["Modelo Global Actualizado (models/global_model.pt)"]
        REST_API["API REST de Inferencia (scripts/serve.py)<br/>• GET /health<br/>• POST /predict<br/>• POST /predict/batch"]
    end

    subgraph Client1Host ["ACTOR 1: CLIENTE EVALUADOR / CONSUMIDOR (REST)"]
        C1_Data["Imágenes de Campo Propias"]
        C1_Curl["Cliente HTTP Ligero (curl / Web / App)<br/>NO clona el repositorio ni instala PyTorch"]
    end

    subgraph Client2Host ["ACTOR 2: CLIENTE ENTRENADOR ON-PREMISE (FL COLABORADOR)"]
        C2_PrivateData["Dataset Privado Local<br/>(100% On-Premise)"]
        C2_Trainer["scripts/client_trainer.py<br/>• Optimización FedProx (μ)<br/>• Privacidad Diferencial (DP-SGD σ)"]
        C2_Params["Artefacto de Parámetros Exportado:<br/>client_parameters.pt (42.7 MB Tensores, 0B Imágenes)<br/>client_parameters.meta.json (SHA-256)"]
    end

    C1_Data --> C1_Curl
    C1_Curl -->|1. Inferencia remota POST /predict| REST_API
    REST_API -->|Diagnóstico + Confianza| C1_Curl

    W_Base -->|2. Descarga de pesos base| C2_Trainer
    C2_PrivateData --> C2_Trainer
    C2_Trainer --> C2_Params
    C2_Params -->|3. Entrega de tensores numéricos| Gate

    Gate -->|4. Aprobado por el filtro de seguridad| Aggregator
    Aggregator --> W_Global
    Aggregator --> Metadata
    W_Global --> REST_API
```

---

## 3. Componentes Arquitectónicos Clave

### A. Patrón Adaptador de Modelos (`BaseFLWrapper`)

Flower y los orquestadores distribuidos requieren listas planas de matrices NumPy (`list[np.ndarray]`). Los modelos de visión artificial gestionan parámetros de capas y buffers de estado (`torch.nn.Parameter`, `state_dict`).

* [`BaseFLWrapper`](../src/agrifed_khaos/wrappers.py#L119): Interfaz abstracta que exige:
  - `get_parameters() -> list[np.ndarray]`
  - `set_parameters(parameters: Sequence[np.ndarray]) -> None`
  - `train_local_epoch(train_data, epochs, prox_mu, reference_parameters) -> TrainingStats`
  - `evaluate(eval_data) -> EvaluationStats`
* [`TorchModuleWrapper`](../src/agrifed_khaos/wrappers.py#L167): Implementación genérica para arquitecturas PyTorch. **Aísla exclusivamente los parámetros con `requires_grad=True`**, omitiendo buffers volátiles de normalización (como `running_mean` o `num_batches_tracked` de BatchNorm) que causan inconsistencias dimensionales en la optimización proximal.
* [`TimmClassificationWrapper`](../src/agrifed_khaos/wrappers.py#L305): Factoría para inicializar cualquier backbone moderno de Vision Transformer (ViT) o CNN mediante `timm`.
* [`YOLOv8Wrapper`](../src/agrifed_khaos/wrappers.py#L338): Adaptador para redes de detección y segmentación de Ultralytics.

---

### B. Regularización FedProx y Función de Pérdida Proximal

En escenarios federados agrícolas o industriales, las distribuciones locales son altamente **No-IID** (desbalance de clases por clima, cámara, variedad). Para prevenir la divergencia del cliente (*client drift*), el entrenamiento local aplica la función de pérdida proximal de **FedProx**:

$$\mathcal{L}_{\text{prox}}(W; W_{\text{ref}}) = \mathcal{L}_{\text{CE}}(W; X, Y) + \frac{\mu}{2} \sum_{l=1}^{L} \| W_l - W_{l,\text{ref}} \|_2^2$$

Donde:
* $\mathcal{L}_{\text{CE}}$ es la entropía cruzada del lote local.
* $\mu \ge 0$ es el hiperparámetro proximal (por defecto $0.01$).
* $W_{\text{ref}}$ son los pesos del modelo global ancla transmitidos por el servidor al inicio de la ronda.

---

### C. Puerta de Calidad y Seguridad (`ParameterValidator`)

Todo vector de parámetros recibido por el servidor central es auditado de forma determinista antes de su incorporación:

1. **Integridad Estructural:** Coincidencia estricta en el número de tensores, formas dimensionales (`shape`) y tipo de dato (`dtype`).
2. **Finitud Numérica:** Verificación estricta de ausencia total de valores no finitos (`NaN` o `Inf`).
3. **Cota de Divergencia Euclidiana ($L_2$ Gate):**
   $$\|\Delta W\|_2 = \sqrt{\sum_{i=1}^N \sum_{j} (W_{i,j}^{\text{cliente}} - W_{i,j}^{\text{base}})^2} \le \tau$$
   Si $\|\Delta W\|_2 > \tau$ (por defecto $\tau = 150.0$), la actualización es rechazada automáticamente, mitigando ataques de envenenamiento de modelo (*Sybil attacks* / *Model Replacement*).

---

### D. Privacidad Diferencial (DP-SGD) en el Cliente

Para evitar que un servidor central no confiable reconstruya imágenes mediante ataques de inversión de gradiente (*Deep Leakage from Gradients*, DLG), el cliente puede activar ruido de privacidad diferencial antes de exportar sus pesos:

$$W_{\text{export}} = W_{\text{local}} + \mathcal{N}(0, \sigma^2 \mathbf{I})$$

Donde $\sigma$ es la desviación estándar del ruido gaussiano calibrado.

---

### E. Gobernanza de Modelos y Linaje Criptográfico (SHA-256)

Cada checkpoint generado incorpora una huella digital inmutable:
* **Modelo Base:** Se almacena con su digest SHA-256, hiperparámetros de entrenamiento y lista de clases en [`models/metadata.json`](../models/metadata.json).
* **Actualización del Cliente:** Genera un manifiesto [`client_parameters.meta.json`](../models/client_parameters.meta.json) vinculando el SHA-256 de los pesos locales y el SHA-256 del modelo base del cual desciende.
* **Agregación Global:** [`scripts/aggregate.py`](../scripts/aggregate.py) registra un historial inmutable de rondas (`aggregation_history`) conteniendo los hashes de entrada, la norma $\|\Delta W\|_2$ verificada y el hash del nuevo checkpoint.

---

## 4. Diagrama de Secuencia: Ciclo de Vida Federado Completo

```mermaid
sequenceDiagram
    autonumber
    actor C1 as Cliente 1 (Evaluador Externo)
    participant Server as Servidor Maestro (API & Gate)
    actor C2 as Cliente 2 (Colaborador On-Premise)

    Note over Server: Estado Inicial: models/base_model.pt (Hash: H_base)
    
    rect rgb(245, 248, 255)
    Note over C1,Server: FASE A: Inferencia Remota Pre-Actualización (T0)
    C1->>Server: POST /predict con foto_campo.jpg
    Server->>Server: Inferencia con modelo base W_base
    Server-->>C1: Diagnóstico clase + Confianza T0
    end

    rect rgb(255, 250, 245)
    Note over Server,C2: FASE B: Entrenamiento Privado On-Premise (Zero-Leakage)
    Server->>C2: Descarga de base_model.pt (Verifica SHA-256 == H_base)
    Note over C2: C2 entrena en su servidor local con client_trainer.py.<br/>¡Cero fotos salen del cliente!<br/>L_prox = L_ce + 0.5 * mu * ||W - W_base||²
    C2->>C2: Exporta client_parameters.pt y calcula SHA-256 H_c2
    C2->>Server: Transfiere ÚNICAMENTE client_parameters.pt (42.7 MB Tensores, 0B Imágenes)
    end

    rect rgb(245, 255, 245)
    Note over Server: FASE C: Auditoría de Seguridad & Agregación
    Server->>Server: ParameterValidator comprueba: finitud, shapes y ||ΔW||₂ ≤ 150.0
    Server->>Server: Agregación FedProx: W_global = W_base + η · (W_c2 - W_base)
    Server->>Server: Guarda models/global_model.pt y actualiza metadata.json
    end

    rect rgb(245, 248, 255)
    Note over C1,Server: FASE D: Inferencia Remota Post-Actualización (T1)
    C1->>Server: POST /predict con foto_campo.jpg
    Server->>Server: Inferencia con modelo global actualizado W_global
    Server-->>C1: Diagnóstico mejorado + Mayor confianza T1
    end
```

