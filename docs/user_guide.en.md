# User Guide: AgriFed-KHAOS

> English documentation. Versión en español disponible en [docs/user_guide.md](user_guide.md).

Operational manual for **AgriFed-KHAOS** to train, serve, and federate computer vision models on **any vision dataset organized in standard directory structures**.

---

## 1. Universal Data Structure (ImageFolder)

The framework is agnostic to crop species, pathology, or image format. It requires images to be partitioned into subdirectories corresponding to their target class:

```text
my_dataset/
├── class_A/
│   ├── image_001.jpg
│   └── image_002.jpg
├── class_B/
│   ├── image_003.jpg
│   └── image_004.jpg
└── class_C/
    └── image_005.jpg
```

*Subdirectory names automatically define the ground-truth classes. The system resolves the number of classes and adapts input resolutions dynamically.*

<details>
<summary><strong>How to organize your directories for Docker vs Local Python</strong></summary>

In Docker-based environments, the container mounts the project repository into `/workspace`:
* **Recommended:** Place your dataset directory directly inside the cloned repository, e.g., in `dataset/` or `my_dataset/`. Inside the container, the path is immediately `dataset/train` or `my_dataset/train`.
* **If your data resides in an external host folder (e.g., <code>/home/user/my_images/</code>):**
  - *Option A (Copy inside project):* `cp -r /home/user/my_images ./dataset`
  - *Option B (Mount as external volume):* In `compose.yaml`, map the external directory into the container:
    ```yaml
    volumes:
      - .:/workspace
      - /home/user/my_images:/data:ro
    ```
    Then inside the container, specify `--data-dir /data/train`.
</details>

---

## 2. Role 1: System Administrator (Master Server)

The Administrator manages the central orchestrator, trains the initial baseline model, exposes inference microservices, and executes federated aggregation across network participants.

### Step 1: Environment Setup
```bash
cp .env.example .env
make up
docker compose exec agrifed bash
```

### Step 2: Train the Official Baseline Model
Specify the directory containing initial baseline training images:

```bash
python scripts/train_base.py --data-dir my_dataset/train
```

*Auto-FL automatically resolves the optimal architecture (`auto`), discovers classes, executes balanced training, and serializes the self-contained bundle `models/base_model.pt`.*

<details>
<summary>Advanced configuration options</summary>

* `--arch {auto, resnet18, resnet50, mobilenetv4_conv_small}`: Forces a specific backbone architecture (default: `auto`).
* `--target-client-tier {cpu-low, cpu-high, gpu}`: Edge client compute tier profile (default: `cpu-low`).
* `--epochs <int>`: Training epochs (default: 3).
* `--batch-size <int>`: Batch size (default: 32).
* `--lr <float>`: Learning rate (default: 0.001).
* `--img-size <int>`: Input image spatial resolution in pixels (default: 224).
* `--output <path>`: Destination checkpoint path (default: `models/base_model.pt`).
</details>

### Step 3: Launch REST Inference and Model Reception Microservice
Starts the zero-dependency HTTP microservice to serve real-time field diagnostics and accept automated collaborative parameter uploads:

```bash
# Standard execution (development / private network)
python scripts/serve.py

# Or with native TLS termination (production)
python scripts/serve.py --ssl-keyfile /path/to/privkey.pem --ssl-certfile /path/to/fullchain.pem
```

*Listens on `0.0.0.0:8000`. Automatically hot-reloads `models/global_model.pt` (or `base_model.pt`), applies calibrated One-vs-Rest thresholding, exposes `/download-model` for client bootstrap, and activates secure atomic ingestion at `/submit-update`.*

### Step 4: Parameter Aggregation with FedAvgM Server Momentum
The Administrator aggregates parameter bundles received from network clients using **FedAvgM (Federated Averaging with Server Momentum)**:

```bash
python scripts/aggregate.py --client-weights models/client_updates/*.pt
```

* **Server Momentum ($\beta = 0.9$):** In contrast to passive arithmetic averaging, FedAvgM maintains a momentum velocity buffer $V_t = \beta V_{t-1} + \Delta W_{\text{fed}}$, dampening oscillations and accelerating global convergence under heavy Non-IID client distributions (Hsu et al., 2019).
* **Buffer Lineage & Provenance:** `server_momentum.pt` is stored as a versioned bundle (`format_version: 2`) cryptographically bound to `parent_base_sha256` and `produced_global_sha256`. If aggregation begins from an unrelated base model, the buffer discards obsolete momentum and resets velocity to zero. The buffer is persisted atomically strictly *after* the global model checkpoint is committed to disk.
* **Security & Quality Gates ([`ParameterValidator`](../src/agrifed_khaos/wrappers.py)):** Enforces numeric finiteness (rejecting `NaN` / `Inf`), strict structural parameter schemas, absolute and relative L2 divergence bounds, and HMAC cryptographic authentication.

<details>
<summary>Advanced aggregation options</summary>

* `--server-momentum <float>`: Server momentum factor $\beta \in [0, 1)$ (default: 0.9; tunable to 0.0 for standard FedAvg).
* `--eta <float>`: Server learning rate step size (default: 0.6).
* `--allow-new-classes`: Enables dynamic classifier expansion when clients discover novel pathologies conforming to global taxonomy.
* `--max-norm <float>`: Maximum absolute L2 perturbation norm permitted before rejection (default: 150.0).
* `--skip-invalid-clients`: Bypasses Byzantine or corrupted client updates without aborting aggregation for valid nodes.
</details>

### Step 5: Global Model Evaluation and Threshold Calibration
Evaluates generalization on an independent test split, tunes One-vs-Rest decision thresholds ($\tau_c^*$), and generates Grad-CAM explainability heatmaps:

```bash
python scripts/evaluate.py --data-dir my_dataset/test
```

---

## 3. Role 2: Consumer User (Field Agronomist & Diagnostic Apps)

The Consumer is an agronomist, technician, or field app that queries the model to obtain reliable disease diagnostics with calibrated confidence metrics.

### Mode A: Centralized Remote Inference (REST API)
Designed for field operators with mobile connectivity who require diagnostics without installing machine learning frameworks:
* **Requirements:** Zero AI dependencies. No need to clone the repository or install PyTorch. Only requires HTTP access to the Master Server.

```bash
curl -X POST http://<SERVER_HOST_OR_IP>:8000/predict \
  -F "file=@field_sample.jpg"
```

**Diagnostic Response:**
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
  "filename": "field_sample.jpg"
}
```

### Mode B: Edge Offline-First Inference
Designed for field operators working in remote rural zones without cellular reception:
* The user downloads the self-contained bundle `models/global_model.pt`.
* Inference executes 100% locally on the edge device (laptop, field terminal, or mobile app via ONNX Runtime / TFLite) with zero internet access required.

---

## 4. Role 3: Collaborator User (On-Premise Farm / Cooperative)

The Collaborator possesses private field images captured across its agricultural holdings. Under GDPR regulations, commercial confidentiality, or bandwidth constraints, **zero raw images are permitted to leave the local network**.

The collaborator participates in federated training through an automated local agent that exchanges mathematical parameter tensors only.

### How Does the Collaborator Operate in Practice?

<details open>
<summary><strong>Step-by-Step Guide for Collaborators from Scratch</strong></summary>

If your cooperative joins the federated collaboration network on a clean machine:

1. **Clone the repository and prepare the environment:**
   ```bash
   git clone https://github.com/Diegodepab/AgriFed-khaos.git
   cd AgriFed-khaos
   cp .env.example .env
   make up
   docker compose exec agrifed bash
   ```

2. **Download official baseline model weights:**
   The master server publishes the baseline model via HTTP. Download it directly into `models/`:
   ```bash
   mkdir -p models
   curl -o models/base_model.pt http://<MASTER_SERVER_IP>:8000/download-model
   ```

3. **Execute local private training and transmit parameters:**
   Specify your local photos directory (e.g., `my_private_photos/` or `dataset/val/`):
   ```bash
   python scripts/client_trainer.py \
     --data-dir my_private_photos/ \
     --weights-in models/base_model.pt \
     --head-only \
     --client-id coop_my_region \
     --server-url http://<MASTER_SERVER_IP>:8000
   ```

*The process runs in under 1 minute, trains only the classifier head (~18 KB), and transmits the parameter bundle alongside its companion `.meta.json` manifest to the central server's `/submit-update` endpoint without a single photo leaving your facility.*
</details>

#### Automated Execution with Docker (Recommended):
The container runs unattended, trains over local private data, and streams parameters via HTTP to the orchestrator:

```bash
docker run --rm \
  -v /path/to/my_photos:/data \
  agrifed \
  python scripts/client_trainer.py \
    --data-dir /data \
    --server-url http://<MASTER_SERVER_IP>:8000
```

#### Native Local Python Execution:
```bash
python scripts/client_trainer.py \
  --data-dir /path/to/my_photos/ \
  --server-url http://<MASTER_SERVER_IP>:8000
```

### Technical Privacy & Ingestion Guarantees:
* **0 Bytes of Images Transferred:** The trainer extracts gradients locally and exports only the numerical tensor bundle `models/client_parameters.pt` (~18 KB in classifier head mode) and its companion manifest `client_parameters.meta.json`.
* **Atomic Ingestion Transactions:** `/submit-update` serializes client uploads via asynchronous locks (`asyncio.Lock`) and publishes each update atomically to an immutable directory `models/client_updates/updates/{client_id}/{stamp}/` via `os.replace`, guaranteeing weights and manifests appear simultaneously.
* **Transport Layer Security (TLS/HTTPS Required in Production):** Ingestion credentials `AGRIFED_AUTH_TOKEN` (Bearer) and `AGRIFED_HMAC_KEY` (integrity signature) travel in plaintext without TLS. Outside isolated local subnets, terminate TLS natively using `--ssl-*` in `serve.py` or behind an HTTPS reverse proxy (Nginx/Caddy). Clients can specify `--ca-bundle` and `--insecure-skip-tls-verify` for custom or testing PKI.
* **Parameter Perturbation (Gaussian Mechanism on ΔW):** Optionally clips global L2 sensitivity (`--dp-clip-norm`) and injects calibrated Gaussian noise (`--dp-noise-std`, $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$), mitigating gradient inversion and reconstruction attacks. *(Note: Implements Gaussian perturbation on parameter deltas; does not evaluate an analytical $(\epsilon, \delta)$ differential privacy accountant).*
* **Threat Model Boundaries:** The framework assumes an honest, authorized orchestrator; cryptographic multi-party computation (*SecAgg*) masking weights from the server is not included. See [Threat Model & Privacy Boundaries](architecture.en.md#5-threat-model--privacy-boundaries) for the full specification.

> [!WARNING]
> **Critical Governance Warning on Taxonomy Alignment (`--allow-new-classes`):**
> When performing federated visual incremental learning under Non-IID distributions, incorporating new classes dynamically requires all participating cooperatives to adhere to a **shared canonical naming taxonomy** (e.g., standardized pathology names such as `repilo`, `antracnosis`, `cochinilla`). If two distinct entities use divergent names for the same condition, or assign the same label to biologically distinct phenomena, label collisions and catastrophic forgetting will occur. The server performs nominal alignment via deterministic CRC32 initialization; disciplined folder nomenclature is therefore an indispensable governance requirement.

<details>
<summary>Advanced collaborator options</summary>

* `--head-only`: *(Enabled by default)* Trains only the linear classifier head (~18 KB vs 42 MB transfer).
* `--server-url <url>`: Central server endpoint for automated update transmission (weights and companion manifest).
* `--ca-bundle <path>`: Path to custom CA certificate bundle for corporate HTTPS deployments.
* `--insecure-skip-tls-verify`: Skips TLS certificate verification (development/testing environments only).
* `--dp-clip-norm <float>` & `--dp-noise-std <float>`: Applies L2 norm clipping and Gaussian parameter perturbation over $\Delta W$ (must be specified together).
* `--allow-new-classes`: Enables registration of emerging pathologies adhering to global AgriFed taxonomy.
* `--client-id <name>`: Collaborator identifier included in manifest and cryptographic HMAC signature.
</details>

