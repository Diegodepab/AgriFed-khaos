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
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    return parser.parse_args()


def create_app(
    model_path: str,
    metadata_path: str,
    architecture: str | None = None,
) -> FastAPI:
    from agrifed_khaos.models import build_model

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

    device = torch.device("cpu")
    meta_p = Path(metadata_path)
    classes: list[str] = []
    arch = architecture
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        classes = meta.get("classes", [])
        if not arch:
            arch = meta.get("architecture")

    num_classes = len(classes) if classes else 9
    arch = arch or "resnet18"
    net = build_model(architecture=arch, num_classes=num_classes).to(device)
    model_p = Path(model_path)
    model_sha256 = ""
    if model_p.is_file():
        state = torch.load(str(model_p), map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.eval()
        sha = hashlib.sha256()
        with model_p.open("rb") as handle:
            while chunk := handle.read(65536):
                sha.update(chunk)
        model_sha256 = sha.hexdigest()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _infer_single(img_bytes: bytes, filename: str | None) -> dict[str, Any]:
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
            pred_idx = int(probs.argmax().item())
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
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "online",
            "model_path": str(model_p),
            "model_sha256": model_sha256,
            "num_classes": num_classes,
            "classes": classes,
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

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.model, args.metadata, architecture=args.arch)
    print(f"[*] Starting AgriFed REST Inference Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
