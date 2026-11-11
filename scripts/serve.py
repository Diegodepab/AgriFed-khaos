#!/usr/bin/env python3
"""Lightweight REST Inference Server and MLOps Coordinator for AgriFed-KHAOS.

Supports 3 dedicated web interfaces and endpoints on port 8000:
- GET /: Agnostic Client (model selection, disease prediction, confidence & probabilities)
- GET /expert: Expert Agronomist (HITL queue review, uncertainty sampling, on-premise FL tutorial)
- GET /master: Administrator / MLOps (model manager, async training/aggregation/calibration runner)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from agrifed_khaos.api.auth import router as auth_router
from agrifed_khaos.api.catalog import router as catalog_router
from agrifed_khaos.api.datasets import router as datasets_router
from agrifed_khaos.api.hitl import router as hitl_router
from agrifed_khaos.api.inference import (
    MAX_UPLOAD_SIZE_BYTES,
)
from agrifed_khaos.api.inference import (
    router as inference_router,
)
from agrifed_khaos.api.jobs import router as jobs_router
from agrifed_khaos.api.mlops import router as mlops_router
from agrifed_khaos.api.views import router as views_router
from agrifed_khaos.catalog import ModelCatalogManager
from agrifed_khaos.federated_ops import FederatedOperationsManager
from agrifed_khaos.hitl import HITLQueueManager
from agrifed_khaos.inference_service import InferenceEngine, compute_ood_guardrail
from agrifed_khaos.jobs import JobManager, QueuedJob
from agrifed_khaos.schemas import (
    AutoFLRequest,
    DatasetConvertCropsRequest,
    DatasetDeduplicateRequest,
    DatasetFetchURLRequest,
    DatasetInspectRequest,
)
from agrifed_khaos.users_db import UserDatabaseManager

logger = logging.getLogger("agrifed.serve")

# Re-export public API symbols for backwards compatibility with tests and callers
__all__ = [
    "AutoFLRequest",
    "DatasetConvertCropsRequest",
    "DatasetDeduplicateRequest",
    "DatasetFetchURLRequest",
    "DatasetInspectRequest",
    "JobManager",
    "MAX_UPLOAD_SIZE_BYTES",
    "QueuedJob",
    "compute_ood_guardrail",
    "create_app",
    "parse_args",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the AgriFed inference server."""
    parser = argparse.ArgumentParser(description="AgriFed REST Inference Server & Web Portal")
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
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=os.getenv("AGRIFED_SSL_KEYFILE"),
        help="Path to SSL private key file for HTTPS (default: $AGRIFED_SSL_KEYFILE)",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=os.getenv("AGRIFED_SSL_CERTFILE"),
        help="Path to SSL certificate file for HTTPS (default: $AGRIFED_SSL_CERTFILE)",
    )
    return parser.parse_args()


def create_app(
    model_path: str,
    metadata_path: str = "models/metadata.json",
    architecture: str | None = None,
    device_name: str = "auto",
) -> FastAPI:
    """Instantiate and configure the FastAPI server, mounting all modular routers."""
    app = FastAPI(
        title="AgriFed-KHAOS Inference API & MLOps Portal",
        description="Zero-dependency REST service and UI for vision model inference & diagnosis",
        version="1.0.0",
    )

    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolved_models_dir = (
        Path(model_path).parent if Path(model_path).parent.is_dir() else Path("models")
    )

    catalog_manager = ModelCatalogManager(
        catalog_path=resolved_models_dir / "catalog.json",
        models_dir=resolved_models_dir,
    )
    app.state.catalog = catalog_manager

    fl_ops_manager = FederatedOperationsManager(
        storage_path=resolved_models_dir / "federated_operations.json"
    )
    app.state.fl_ops = fl_ops_manager

    job_manager = JobManager(queue_file=resolved_models_dir / "job_queue.json")
    app.state.job_manager = job_manager

    hitl_manager = HITLQueueManager()
    app.state.hitl = hitl_manager

    users_db = UserDatabaseManager(db_path=resolved_models_dir / "users.db")
    app.state.users_db = users_db

    inference_engine = InferenceEngine(
        app=app,
        model_path=model_path,
        metadata_path=metadata_path,
        architecture=architecture,
        device_name=device_name,
    )
    app.state.inference_engine = inference_engine
    app.state.net = inference_engine.net
    app.state.classes = inference_engine.classes

    @app.on_event("startup")
    async def _on_startup() -> None:
        await job_manager.recover_queue()
        inference_engine.resolve_and_reload_model()

    app.include_router(auth_router)
    app.include_router(views_router)
    app.include_router(catalog_router)
    app.include_router(datasets_router)
    app.include_router(hitl_router)
    app.include_router(jobs_router)
    app.include_router(mlops_router)
    app.include_router(inference_router)

    return app


def main() -> None:
    """Server entrypoint with automatic TLS detection and security notices."""
    args = parse_args()
    app = create_app(args.model, args.metadata, architecture=args.arch, device_name=args.device)

    if args.ssl_keyfile and args.ssl_certfile:
        print(
            f"[*] Starting AgriFed REST Inference Server on https://{args.host}:{args.port} "
            "(TLS enabled)"
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            log_level="info",
        )
    else:
        auth_configured = bool(os.getenv("AGRIFED_AUTH_TOKEN") or os.getenv("AGRIFED_HMAC_KEY"))
        if args.host not in ("127.0.0.1", "localhost") and auth_configured:
            print(
                "[!] SECURITY NOTICE: Server is listening on unencrypted HTTP with "
                "authentication/HMAC enabled.\n"
                "    Bearer tokens and HMAC signatures travel in plaintext unless encrypted.\n"
                "    Outside a trusted network, provide --ssl-certfile and --ssl-keyfile or run\n"
                "    behind an HTTPS reverse proxy (e.g., Nginx, Caddy) terminating TLS."
            )
        print(f"[*] Starting AgriFed REST Inference Server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

