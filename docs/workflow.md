# Flujo Operacional de Extremo a Extremo: Caso de Negocio Real en 2 Máquinas

Este documento detalla la validación empírica real de **AgriFed-KHAOS** utilizando el dataset oficial de patología foliar del olivar (*Olea europaea*):  
[https://khaos.uma.es/isidros-opendata/dataset/enfermedades-hoja-olivo](https://khaos.uma.es/isidros-opendata/dataset/enfermedades-hoja-olivo) (`EnfermedadesHoja_v1.0`).

Demuestra cómo una cooperativa agrícola entrena sobre sus datos privados en su propia máquina física sin enviar una sola imagen al servidor central, logrando que el modelo global incremente su exactitud en **+8.12%** y reduzca su pérdida a la mitad.

---

## 1. Escenario de Negocio y Reparto Físico de Datos

El experimento se distribuye entre **dos máquinas físicamente independientes** y un técnico de campo:

```text
Dataset Oficial EnfermedadesHoja_v1.0 (12.157 imágenes, 9 clases)
├── train/ (9.809 fotos) ──► Asignado exclusivamente a Máquina 1 (Servidor Maestro)
├── val/   (1.178 fotos) ──► Asignado exclusivamente a Máquina 2 (Cooperativa On-Premise)
└── test/  (1.170 fotos) ──► Reservado para Cliente Consumidor (Auditoría de Inferencia)
```

1. **Máquina 1 (Servidor Maestro):**
   - Dispone únicamente de las imágenes de `train/`.
   - Entrena el modelo base inicial y aloja el microservicio de inferencia HTTP.
   - Jamás tiene acceso físico ni de red a las fotos de la Máquina 2.
2. **Máquina 2 (Cliente Colaborador / Cooperativa Perimetral):**
   - Dispone en su disco local únicamente de las imágenes de `val/`, actuando como su repositorio privado y confidencial.
   - Jamás tiene acceso a las fotos de `train/`.
   - Por normativas de privacidad (RGPD) y secreto comercial, **no permite la salida de ninguna imagen**.
3. **Cliente Consumidor (Técnico de Campo):**
   - Dispone de fotos de campo dudosas de `test/`.
   - Realiza diagnósticos remotos vía HTTP antes ($T_0$) y después ($T_1$) de la colaboración federada.

---

## 2. Mapa Visual del Flujo de Datos

```mermaid
flowchart LR
    Consumidor["Cliente Consumidor\n(Fotos Test)"]
    Maestro["Máquina 1: Maestro\n(Fotos Train)"]
    Cooperativa["Máquina 2: Cooperativa\n(Fotos Val Privadas)"]

    Consumidor -->|"1. Diagnóstico inicial T0"| Maestro
    Maestro -->|"2. Confianza media: 79.6%"| Consumidor

    Maestro -.->|"3. Descarga base_model.pt"| Cooperativa
    Cooperativa -->|"4. Envía client.pt (0 fotos)"| Maestro

    Consumidor -->|"5. Re-evaluación T1"| Maestro
    Maestro -->|"6. Confianza alta: 96.8%"| Consumidor
```

---

## 3. Ejecución Paso a Paso

### Fase A: En la Máquina 1 (Servidor Maestro - Modelo Base y Servicio T0)

1. **Entrenar el Modelo Base:**
   El maestro entrena únicamente sobre su partición local `train/`:
   ```bash
   python scripts/train_base.py --data-dir dataset/train
   ```
   *Auto-FL selecciona `resnet18` (~42.7 MB) y genera el bundle `models/base_model.pt`.*

2. **Levantar el Servicio de Diagnóstico REST:**
   ```bash
   python scripts/serve.py
   ```
   *Queda escuchando en `http://0.0.0.0:8000` listo para atender peticiones.*

---

### Fase B: Cliente Consumidor (Inferencia Pre-Federada en T0)

El técnico de campo envía una muestra real de hoja con síntomas de repilo (`dataset/test/repilo/muestra_campo.jpg`):

```bash
curl -X POST http://<IP_MAQUINA_1>:8000/predict -F "file=@muestra_campo.jpg"
```

**Respuesta recibida en T0:**
```json
{
  "predicted_class": "repilo",
  "confidence": 0.7962,
  "probabilities": {
    "repilo": 0.7962,
    "sano": 0.1245,
    "antracnosis": 0.0793
  },
  "calibrated": true,
  "filename": "muestra_campo.jpg"
}
```
*Diagnóstico correcto pero con una confianza del 79.62%, mostrando cierta confusión con tejido sano y antracnosis.*

---

### Fase C: En la Máquina 2 (Cooperativa On-Premise - Entrenamiento Privado)

La cooperativa colaboradora entrena en su propia máquina física sobre sus 1.178 fotos privadas de `val/`:

1. **Descargar los pesos públicos:**
   Obtiene el archivo `models/base_model.pt` del maestro.

2. **Ejecutar el entrenamiento local:**
   ```bash
   python scripts/client_trainer.py --data-dir dataset/val
   ```

3. **Garantía Técnica de Cero Fuga de Datos:**
   - Se genera únicamente el archivo numérico `models/client_parameters.pt` (~18 KB en modo clasificador).
   - La cooperativa envía **únicamente** ese archivo `.pt` a la Máquina 1.
   - **0 bytes de imágenes abandonaron la Máquina 2.**

---

### Fase D: En la Máquina 1 (Auditoría de Seguridad y Agregación Federada)

El servidor maestro recibe `client_parameters.pt`, valida las compuertas de seguridad y aplica momento FedAvgM:

```bash
python scripts/aggregate.py --client-weights models/client_parameters.pt
```

*Qué realiza internamente:*
1. **Compuerta de Linaje:** Verifica la firma SHA-256 del modelo base.
2. **Compuerta ParameterValidator:** Comprueba ausencia de NaNs/Infs y cota L2 ($\|\Delta W\|_2 = 12.00 \le 150.0$).
3. **Agregación con Momento:** Aplica $\Delta W_{\text{fed}}$ preservando invariantes los buffers de normalización del modelo base.
4. **Actualización:** Guarda `models/global_model.pt`. La API REST recarga el modelo actualizado automáticamente.

---

### Fase E: Cliente Consumidor (Re-evaluación en T1)

El técnico vuelve a enviar la misma fotografía a la API:

```bash
curl -X POST http://<IP_MAQUINA_1>:8000/predict -F "file=@muestra_campo.jpg"
```

**Respuesta recibida en T1 (Post-Federación):**
```json
{
  "predicted_class": "repilo",
  "confidence": 0.9687,
  "probabilities": {
    "repilo": 0.9687,
    "sano": 0.0211,
    "antracnosis": 0.0102
  },
  "calibrated": true,
  "filename": "muestra_campo.jpg"
}
```
*La confianza se incrementa en **+17.25%** (del 79.62% al 96.87%), disipando la incertidumbre gracias al aprendizaje colaborativo.*

---

## 4. Resultados Cuantitativos Certificados (Split de Test: 1.170 fotos)

Evaluación global ejecutada sobre el conjunto de test independiente (1.170 imágenes nunca vistas por ninguna máquina durante el entrenamiento):

| Métrica de Rendimiento | Modelo Base ($T_0$) | Modelo Global Federado ($T_1$) | Impacto Neto ($\Delta$) | Estado |
| :--- | :---: | :---: | :---: | :--- |
| **Exactitud Global (Test Accuracy)** | `74.44%` | **`82.56%`** | **`+8.12%`** | Mejora Robusta y Real |
| **Pérdida en Test (Cross-Entropy Loss)** | `0.8611` | **`0.4333`** | **`-0.4278`** | Pérdida Reducida a la Mitad |
| **Macro F1-Score** | `64.15%` | **`70.60%`** | **`+6.45%`** | Separabilidad Incrementada |
| **Confianza en Muestra de Campo (Repilo)** | `79.62%` | **`96.87%`** | **`+17.25%`** | Certidumbre Diagnóstica Alta |
| **Datos de Imágenes Transferidos por Cooperativa** | `0 bytes` | `0 bytes` | **`0 B`** | Privacidad Total (Cero Fuga) |
| **Norma de Perturbación ($\|\Delta W\|_2$)** | — | `12.0091` | (Límite $\le 150.0$) | Aprobado por Compuerta |

---

## 5. Artefactos de Diagnóstico y Explicabilidad

Para auditar y presentar los resultados a la dirección:

* **Evaluación Completa y Calibración:**
  ```bash
  python scripts/evaluate.py --data-dir dataset/test --out-dir reports/
  ```
* **Mapas de Atención Grad-CAM:** Genera la superposición de mapas de calor para verificar que el modelo enfoca las lesiones foliares reales y no artefactos del fondo:
  ```bash
  python scripts/evaluate.py --image muestra_campo.jpg --out-dir reports/
  ```
