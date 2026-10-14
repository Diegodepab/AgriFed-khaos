# Flujo Operacional de Extremo a Extremo: Del Cliente 1 al Cliente 2 y Agregación Federada

Este documento describe el caso de uso real de negocio de **AgriFed-KHAOS**: cómo una red de organizaciones o cooperativas colabora para entrenar un modelo global de diagnóstico sin transferir imágenes privadas entre ellas.

---

## 1. Escenario de Negocio: La Problemática

* **Cliente 1 (Evaluador Externo / Técnico de Campo):** Dispone de imágenes de campo dudosas (ej. síntomas incipientes o variantes raras) y consulta el servicio central mediante la API REST. El modelo base actual responde con baja certidumbre o confusión diagnóstica.
* **Cliente 2, 3... K (Cooperativas / Productores con Datasets Privados):** Disponen de miles de fotos de campo de esas mismas patologías. Por normativas de privacidad (RGPD / secretos industriales), **no pueden ni quieren subir sus fotos a ningún servidor externo**.
* **Solución AgriFed-KHAOS:** Las cooperativas descargan los pesos base, entrenan localmente en sus propios servidores (con `freeze_bn_stats` y ruido adaptativo RMS), y envían exclusivamente sus archivos de parámetros `.pt` acompañados de un manifiesto criptográfico `.meta.json`. El servidor maestro fusiona todas las actualizaciones ponderándolas por el número de muestras ($n_k$) mediante `FedOpt`. El Cliente 1 vuelve a consultar la API REST y comprueba la mejora sustancial del diagnóstico.

---

## 2. Mapa Visual del Flujo de Datos

```mermaid
flowchart TD
    subgraph Cliente1 ["Cliente 1 (Evaluador - Sin instalación IA)"]
        C1_Req["Foto de Campo local (mi_foto.jpg)"]
        C1_Eval["Diagnóstico vía curl / HTTP"]
    end

    subgraph ServidorMaestro ["Servidor Maestro (Hub de Inferencia y Agregación)"]
        REST_API["API REST de Inferencia (scripts/serve.py)"]
        MasterModel["Modelo Base Oficial (train_base.py --arch auto)"]
        Gate["ParameterValidator Gate<br/>• Finitud & Esquema Cerrado<br/>• Cota L2 Absoluta & Relativa<br/>• Linaje & HMAC con num_samples"]
        Aggregator["Agregador Multi-Cliente FedOpt<br/>ΔW_fed = Σ (n_k / N) · (W_k - W_base)<br/>W_global = W_base + η · ΔW_fed"]
        GlobalModel["Modelo Global Oficial (models/global_model.pt)"]
    end

    subgraph ClientesColaboradores ["Clientes 2 ... K (Cooperativas On-Premise)"]
        CK_Data["Datasets Privados Locales (100% On-Premise)"]
        CK_Train["Entrenamiento Local (client_trainer.py)<br/>• Frozen BatchNorm Statistics<br/>• FedProx Proximal Loss<br/>• Ruido Adaptativo RMS"]
        CK_Params["client_k.pt (Tensores Numéricos, 0B Fotos)<br/>client_k.meta.json (SHA-256 + n_k + HMAC)"]
    end

    C1_Req -->|1. POST /predict (Prueba T0)| REST_API
    REST_API -->|Diagnóstico + Confianza T0| C1_Eval

    MasterModel -->|2. Descarga de Pesos Base (base_model.pt)| CK_Train
    CK_Data --> CK_Train
    CK_Train --> CK_Params
    CK_Params -->|3. Entrega de Tensores Numéricos| Gate

    Gate -->|4. Aprobado por la compuerta de seguridad| Aggregator
    Aggregator --> GlobalModel
    GlobalModel --> REST_API

    C1_Req -->|5. POST /predict (Re-evaluación T1)| REST_API
    REST_API -->|Diagnóstico Mejorado T1| C1_Eval
```

---

## 3. Guía Operativa Paso a Paso: Comandos Reales y Reproducibles

### Paso 1: En el Servidor Maestro (Selección Auto-FL, Entrenamiento Base y Servicio)

1. **Entrenar el Modelo Base con Auto-FL:**
   El selector inteligente [`ModelSelector`](../src/agrifed_khaos/hardware.py) determina la arquitectura óptima según el perfil de cliente perimetral (`--target-client-tier`) y las dimensiones del dataset:
   ```bash
   python scripts/train_base.py \
       --data-dir /ruta/al/dataset/train \
       --arch auto \
       --target-client-tier cpu-high \
       --output models/base_model.pt \
       --metadata models/metadata.json
   ```
   *Salida generada:*
   - `models/base_model.pt`: Checkpoint de pesos base (ResNet-18 para `cpu-high`, ~42.7 MB).
   - `models/metadata.json`: Lista de clases descubiertas, hash SHA-256 de linaje y configuración.

2. **Levantar el Servicio de Inferencia REST (para clientes externos):**
   ```bash
   python scripts/serve.py --model models/base_model.pt --port 8000
   ```
   *El servidor queda escuchando en `http://0.0.0.0:8000` listo para atender peticiones.*

---

### Paso 2: Cliente 1 (Evaluador externo sin clonar el repositorio)

El Cliente 1 no requiere tener Python, PyTorch ni GPU instalados. Puede consumir el servicio con una herramienta estándar como `curl`:

- **Comando de Consulta de Inferencia:**
  ```bash
  curl -X POST -F "file=@mi_foto_campo.jpg" http://<DOMINIO_O_IP_SERVIDOR>:8000/predict
  ```

- **Respuesta JSON recibida:**
  ```json
  {
    "predicted_class": "sano",
    "confidence": 0.7542,
    "probabilities": {
      "sano": 0.7542,
      "repilo": 0.1815,
      "cochinilla_tizne": 0.0643
    },
    "filename": "mi_foto_campo.jpg"
  }
  ```

---

### Paso 3: Clientes Colaboradores (Entrenamiento On-Premise con Privacidad Total)

Cada cooperativa colaboradora entrena sobre sus imágenes privadas:

1. **Descargar los Pesos Base:**
   Descarga únicamente el archivo `models/base_model.pt` del Maestro.

2. **Ejecutar el Entrenamiento Local (con Frozen BatchNorm y Ruido Adaptativo RMS):**
   ```bash
   python scripts/client_trainer.py \
       --data-dir /datos/privados/cooperativa/ \
       --weights-in models/base_model.pt \
       --weights-out models/client_parameters.pt \
       --epochs 2 \
       --batch-size 32 \
       --mu 0.01 \
       --noise-std 0.01 \
       --client-id coop_sevilla
   ```

3. **¿Qué artefactos se generan exactamente?**
   - **`client_parameters.pt`**: Diccionario serializado con tensores de pesos numéricos ($W$).
   - **`client_parameters.meta.json`**: Manifiesto criptográfico que incluye el SHA-256 de los pesos, el linaje del modelo base, el recuento auditado de muestras (`num_samples`) y la firma HMAC.
   - **Garantía:** **0 bytes de imágenes abandonan el host perimetral.**

---

### Paso 4: En el Servidor Maestro (Auditoría Multi-Cliente y Agregación Ponderada)

Cuando el Maestro recibe las actualizaciones de una o múltiples cooperativas, ejecuta la compuerta de seguridad y la agregación simultánea:

```bash
python scripts/aggregate.py \
    --base-model models/base_model.pt \
    --client-weights models/client_coop1.pt models/client_coop2.pt \
    --output models/global_model.pt \
    --metadata models/metadata.json \
    --eta 0.6 \
    --max-norm 150.0 \
    --max-relative-norm 1.0 \
    --skip-invalid-clients
```

**Qué hace este comando internamente:**
1. **Compuertas de Seguridad Individuales por Cliente:**
   - Verifica linaje SHA-256 (`base_model_sha256`) y autenticidad HMAC de los tensores y de `num_samples`.
   - Verifica finitud de tensores y ausencia de NaNs/Infs.
   - Audita que la divergencia absoluta $\|\Delta W\|_2 \le 150.0$ y relativa $\le 1.0$.
   - Con `--skip-invalid-clients`, si un cliente está corrupto, lo descarta con un warning sin cancelar la ronda.
2. **Fusión Ponderada Multi-Cliente (`FedOpt`):**
   $$\Delta W_{\text{fed}} = \sum_{k=1}^K \frac{n_k}{N} (W_k - W_{\text{base}})$$
   $$W_{\text{global}} = W_{\text{base}} + \eta \cdot \Delta W_{\text{fed}}$$
3. **Invarianza de Normalización:** Los buffers de BatchNorm (`running_mean`, `running_var`) se mantienen intactos e invariantes del modelo base.
4. **Registro de Rondas:** Actualiza `aggregation_history` en `models/metadata.json` con la cuota relativa y norma de cada cooperativa.
5. Guarda el nuevo modelo oficial en `models/global_model.pt`. El servicio REST recarga el modelo actualizado automáticamente.

---

### Paso 5: Verificación de Mejora en el Cliente 1

El Cliente 1 vuelve a enviar la misma imagen de test:
```bash
curl -X POST -F "file=@mi_foto_campo.jpg" http://<DOMINIO_O_IP_SERVIDOR>:8000/predict
```
El cliente comprueba que la certidumbre aumentó sustancialmente (ej. de 75.4% a 96.8%) gracias al conocimiento acumulado de las cooperativas, **sin que ninguna imagen fuera compartida**.

---

## 4. Resultados Empíricos (Caso de Estudio de Referencia: Olivo)

Evaluación certificada sobre el dataset de referencia de 9 clases foliares (1.170 imágenes de test):

| Métrica de Rendimiento | Modelo Base ($T_0$) | Modelo Global Federado ($T_1$) | Impacto Neto ($\Delta$) | Estado |
|---|:---:|:---:|:---:|:---:|
| **Exactitud Global (Test Accuracy)** | `74.44%` | **`82.56%`** | **`+8.12%`** | :white_check_mark: Mejora Robusta y Real |
| **Pérdida en Test (Cross-Entropy Loss)** | `0.8611` | **`0.4333`** | **`-0.4278`** | :white_check_mark: Pérdida Reducida a la Mitad |
| **Macro F1-Score** | `64.15%` | **`70.60%`** | **`+6.45%`** | :white_check_mark: Separabilidad Incrementada |
| **Confianza en Muestra de Campo (Repilo)** | `79.62%` | **`96.87%`** | **`+17.25%`** | :white_check_mark: Certidumbre Diagnóstica Alta |
| **Datos de Imágenes Transferidos por Cooperativas** | `0 bytes` | `0 bytes` | **`0 B`** | :lock: **Privacidad Total (Zero Leakage)** |
| **Norma de Perturbación ($\|\Delta W\|_2$)** | — | `12.0091` | (Límite $\le 150.0$) | :white_check_mark: Aprobado por el Gate |
