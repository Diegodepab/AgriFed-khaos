#!/usr/bin/env python3
"""Lightweight REST Inference Server for AgriFed-KHAOS.

Allows external clients (e.g., Client 1 / Field Agronomists) to query model predictions
via HTTP REST API without cloning the repository or installing PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms

# Defend against Decompression Bomb Denial-of-Service attacks (CWE-400)
Image.MAX_IMAGE_PIXELS = 25_000_000  # Max 25 megapixels (~5000x5000)
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit per image




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgriFed REST Inference Server")
    parser.add_argument(
        "--model", type=str, default="models/global_model.pt", help="Path to model weights"
    )
    parser.add_argument(
        "--metadata", type=str, default="models/metadata.json", help="Path to metadata json"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Listen host (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Model architecture (default: resolved from metadata.json or 'resnet18')",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference hardware device: auto, cpu, cuda (default: auto)",
    )
    return parser.parse_args()


def create_app(
    model_path: str,
    metadata_path: str,
    architecture: str | None = None,
    device_name: str = "auto",
) -> FastAPI:
    app = FastAPI(
        title="AgriFed-KHAOS Inference API",
        description="Zero-dependency REST service for vision model inference & diagnosis",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_p = Path(model_path)
    meta_p = Path(metadata_path)

    from agrifed_khaos.models import build_model, load_checkpoint_bundle

    active_model_p: Path | None = None
    last_mtime: float = -1.0
    bundle_meta: dict[str, Any] = {}
    classes: list[str] = []
    arch: str = architecture or "resnet18"
    num_classes: int = 9
    calibrated_thresholds: Any = None
    model_sha256: str = ""
    img_size_tuple: tuple[int, int] = (224, 224)
    transform = transforms.Compose(
        [
            transforms.Resize(img_size_tuple),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    net: torch.nn.Module = build_model(architecture=arch, num_classes=num_classes).to(device)

    def _resolve_and_reload_model(force: bool = False) -> None:
        nonlocal active_model_p, last_mtime, bundle_meta, classes, arch, num_classes
        nonlocal calibrated_thresholds, model_sha256, img_size_tuple, transform, net

        candidate_p = Path(model_path)
        if not candidate_p.is_file():
            base_cand = Path("models/base_model.pt")
            if base_cand.is_file():
                candidate_p = base_cand
            else:
                return
        else:
            if (
                candidate_p == Path("models/base_model.pt")
                and Path("models/global_model.pt").is_file()
            ):
                candidate_p = Path("models/global_model.pt")

        current_mtime = candidate_p.stat().st_mtime
        if not force and active_model_p == candidate_p and current_mtime == last_mtime:
            return

        state, b_meta = load_checkpoint_bundle(candidate_p, device=device)
        sha = hashlib.sha256()
        with candidate_p.open("rb") as handle:
            while chunk := handle.read(65536):
                sha.update(chunk)
        model_sha256 = sha.hexdigest()

        meta: dict[str, Any] = {}
        if meta_p.is_file():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        classes = b_meta.get("classes") or meta.get("classes") or []
        arch = (
            architecture
            or b_meta.get("architecture")
            or meta.get("architecture")
            or "resnet18"
        )
        num_classes = (
            b_meta.get("num_classes")
            or meta.get("num_classes")
            or (len(classes) if classes else 9)
        )
        calibrated_thresholds = b_meta.get("calibrated_thresholds") or meta.get(
            "calibrated_thresholds"
        )

        net = build_model(architecture=arch, num_classes=num_classes).to(device)
        if state:
            net.load_state_dict(state)
            net.eval()

        raw_img_size = b_meta.get("img_size") or meta.get("img_size") or 224
        if isinstance(raw_img_size, int):
            img_size_tuple = (raw_img_size, raw_img_size)
        elif isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) == 2:
            img_size_tuple = (int(raw_img_size[0]), int(raw_img_size[1]))
        else:
            img_size_tuple = (224, 224)

        transform = transforms.Compose(
            [
                transforms.Resize(img_size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        active_model_p = candidate_p
        last_mtime = current_mtime

    _resolve_and_reload_model()

    def _infer_single(img_bytes: bytes, filename: str | None) -> dict[str, Any]:
        _resolve_and_reload_model()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Image.DecompressionBombError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Decompression bomb detected: image exceeds pixel limit ({e})",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format or corrupted payload: {e}",
            ) from e

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = net(tensor)
            probs = F.softmax(logits, dim=1)[0]

            # Apply calibrated decision thresholds if present
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())

            if calibrated_thresholds:
                # 1. Exact binary thresholding if explicit single positive-class threshold
                if num_classes == 2 and any(
                    k in calibrated_thresholds
                    for k in ("calibrated_youden", "calibrated_f1", "optimal_threshold")
                ):
                    tau_pos: float | None = None
                    if "calibrated_youden" in calibrated_thresholds:
                        opt = calibrated_thresholds["calibrated_youden"].get("optimal_threshold")
                        if opt is not None:
                            tau_pos = float(opt)
                    elif "calibrated_f1" in calibrated_thresholds:
                        opt = calibrated_thresholds["calibrated_f1"].get("optimal_threshold")
                        if opt is not None:
                            tau_pos = float(opt)
                    elif "optimal_threshold" in calibrated_thresholds:
                        opt = calibrated_thresholds.get("optimal_threshold")
                        if opt is not None:
                            tau_pos = float(opt)

                    if tau_pos is not None:
                        pred_idx = 1 if float(probs[1].item()) >= tau_pos else 0
                        confidence = float(probs[pred_idx].item())
                else:
                    # 2. General C >= 2 calibrated score scaling:
                    # calibrated_scores[c] = probs[c] / max(tau_c, 1e-6)
                    calibrated_scores: list[float] = []
                    for c in range(num_classes):
                        c_name = classes[c] if c < len(classes) else f"class_{c}"
                        tau_c = 0.50
                        if isinstance(calibrated_thresholds, dict):
                            val = calibrated_thresholds.get(c_name)
                            if val is None:
                                val = calibrated_thresholds.get(str(c))
                            if val is None:
                                val = calibrated_thresholds.get(c)
                            if isinstance(val, dict) and "optimal_threshold" in val:
                                val = val["optimal_threshold"]
                            if val is not None and isinstance(val, (int, float)):
                                tau_c = float(val)
                        elif isinstance(calibrated_thresholds, (list, tuple)) and c < len(
                            calibrated_thresholds
                        ):
                            tau_c = float(calibrated_thresholds[c])

                        p_c = float(probs[c].item())
                        calibrated_scores.append(p_c / max(tau_c, 1e-6))

                    pred_idx = int(torch.tensor(calibrated_scores).argmax().item())
                    confidence = float(probs[pred_idx].item())

        pred_class = (
            classes[pred_idx] if classes and pred_idx < len(classes) else f"class_{pred_idx}"
        )
        prob_dict = {
            classes[i] if i < len(classes) else f"class_{i}": round(float(probs[i].item()), 4)
            for i in range(num_classes)
        }

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "filename": filename,
            "threshold_applied": "calibrated" if calibrated_thresholds else "nominal_argmax",
        }

    @app.get("/")
    def root() -> dict[str, Any]:
        _resolve_and_reload_model()
        return {
            "name": "AgriFed-KHAOS REST Inference Server",
            "status": "online",
            "active_model": str(active_model_p) if active_model_p else str(model_p),
            "architecture": arch,
            "classes": classes,
            "endpoints": {
                "health": "GET /health",
                "interactive_docs": "GET /docs",
                "predict": "POST /predict",
                "predict_batch": "POST /predict/batch",
                "download_model": "GET /download-model",
                "submit_update": "POST /submit-update",
            },
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        _resolve_and_reload_model()
        return {
            "status": "online",
            "model_path": str(active_model_p) if active_model_p else str(model_p),
            "model_sha256": model_sha256,
            "architecture": arch,
            "num_classes": num_classes,
            "classes": classes,
            "img_size": list(img_size_tuple),
            "calibrated_thresholds": calibrated_thresholds,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Uploaded image exceeds maximum allowed limit of "
                    f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
                ),
            )

        return _infer_single(content, file.filename)

    @app.post("/predict/batch")
    async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:  # noqa: B008
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded for batch inference")

        predictions: list[dict[str, Any]] = []
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                predictions.append({"filename": file.filename, "error": "Not an image"})
                continue
            content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
            if len(content) > MAX_UPLOAD_SIZE_BYTES:
                predictions.append({
                    "filename": file.filename,
                    "error": (
                        f"Image exceeds limit of "
                        f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
                    ),
                })
                continue
            try:
                predictions.append(_infer_single(content, file.filename))
            except HTTPException as exc:
                predictions.append({"filename": file.filename, "error": exc.detail})
            except Exception as exc:
                predictions.append({"filename": file.filename, "error": str(exc)})

        return {
            "batch_size": len(files),
            "predictions": predictions,
        }

    @app.get("/download-model")
    def download_model() -> FileResponse:
        """Endpoint to allow collaborative edge clients to download the model checkpoint bundle."""
        _resolve_and_reload_model()
        target_file = active_model_p or (model_p if model_p.is_file() else None)
        if not target_file or not target_file.is_file():
            raise HTTPException(status_code=404, detail="Model file not found on server")
        return FileResponse(
            path=str(target_file),
            filename=target_file.name,
            media_type="application/octet-stream",
        )

    @app.post("/reload-model")
    def reload_model() -> dict[str, Any]:
        """Manually trigger checkpoint reloading from disk."""
        _resolve_and_reload_model(force=True)
        return {
            "status": "reloaded",
            "model_path": str(active_model_p) if active_model_p else None,
            "model_sha256": model_sha256,
            "calibrated": calibrated_thresholds is not None,
        }

    @app.post("/submit-update")
    async def submit_update(
        file: UploadFile = File(...),  # noqa: B008
        manifest: UploadFile | None = File(None),  # noqa: B008
        client_id: str = "client",
        authorization: str | None = Header(None),  # noqa: B008
    ) -> dict[str, Any]:
        """Automated ingestion endpoint for collaborative edge clients to upload
        trained parameter bundles and companion cryptographic manifests.
        """
        # 1. Optional client authentication via pre-shared Bearer token
        server_token = os.getenv("AGRIFED_AUTH_TOKEN")
        if server_token:
            expected_auth = f"Bearer {server_token}"
            if not authorization or authorization.strip() != expected_auth:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: invalid or missing Bearer token in Authorization header",
                )

        # 2. Client ID validation and normalization
        trimmed_id = client_id.strip()
        if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", trimmed_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid client_id: must match regex ^[a-zA-Z0-9_-]{1,64}$",
            )
        safe_client_id = trimmed_id

        # 3. Read uploaded parameter file
        content = await file.read(50 * 1024 * 1024)  # 50 MB max limit for parameter tensors
        if not content:
            raise HTTPException(status_code=400, detail="Empty parameter file uploaded")
        file_sha256 = hashlib.sha256(content).hexdigest()

        # 4. Manifest processing & integrity validation
        manifest_bytes: bytes | None = None
        manifest_verified = False

        if manifest is not None:
            manifest_bytes = await manifest.read(1 * 1024 * 1024)  # 1 MB max for JSON manifest
            if manifest_bytes:
                try:
                    manifest_data: dict[str, Any] = json.loads(manifest_bytes.decode("utf-8"))
                except Exception as exc:
                    raise HTTPException(
                        status_code=400, detail=f"Malformed manifest JSON: {exc}"
                    ) from exc

                declared_sha = manifest_data.get("sha256") or manifest_data.get(
                    "client_parameters_sha256"
                )
                if declared_sha and declared_sha.lower() != file_sha256.lower():
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Integrity check failed: uploaded payload SHA-256 ({file_sha256}) "
                            f"does not match manifest declared sha256 ({declared_sha})"
                        ),
                    )

                server_hmac_key = os.getenv("AGRIFED_HMAC_KEY")
                if server_hmac_key:
                    from agrifed_khaos.security import verify_manifest_hmac

                    sig = manifest_data.get("hmac_signature")
                    if not sig:
                        raise HTTPException(
                            status_code=403,
                            detail=(
                                "Server requires HMAC signature in manifest, but none was provided."
                            ),
                        )
                    if not verify_manifest_hmac(
                        key=server_hmac_key,
                        manifest=manifest_data,
                        actual_client_sha256=file_sha256,
                    ):
                        raise HTTPException(
                            status_code=403,
                            detail=(
                                "HMAC signature verification failed: invalid key or tampered "
                                "metadata."
                            ),
                        )

                manifest_verified = True

        # 5. Persist weights and companion manifest atomically
        target_dir = Path("models/client_updates")
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = target_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        dest_path = target_dir / f"{safe_client_id}_parameters.pt"
        dest_meta = target_dir / f"{safe_client_id}_parameters.meta.json"

        dest_path.write_bytes(content)

        if manifest_bytes:
            dest_meta.write_bytes(manifest_bytes)
        else:
            synth_meta = {
                "client_id": safe_client_id,
                "sha256": file_sha256,
                "timestamp": time.time(),
                "note": "Synthesized by server: client did not upload separate manifest",
            }
            dest_meta.write_text(json.dumps(synth_meta, indent=2), encoding="utf-8")

        # Archive versioned copy to prevent silent loss or race condition overwrites
        ts = int(time.time())
        (archive_dir / f"{safe_client_id}_{ts}_parameters.pt").write_bytes(content)
        if manifest_bytes:
            (archive_dir / f"{safe_client_id}_{ts}_parameters.meta.json").write_bytes(
                manifest_bytes
            )

        return {
            "status": "ready_for_aggregation",
            "client_id": safe_client_id,
            "bytes_received": len(content),
            "sha256": file_sha256,
            "has_manifest": manifest_bytes is not None,
            "manifest_verified": manifest_verified,
            "saved_to": str(dest_path),
            "meta_saved_to": str(dest_meta),
        }

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.model, args.metadata, architecture=args.arch, device_name=args.device)
    print(f"[*] Starting AgriFed REST Inference Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
