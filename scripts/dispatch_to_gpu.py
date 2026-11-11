#!/usr/bin/env python3
"""Remote GPU Job Dispatcher & Live Log Bridge for AgriFed-KHAOS.

Offloads heavy training workloads (``classification`` or ``detection``) from the
control-plane node to the dedicated ``agrifed-gpu`` worker.

Features:
- Probes ``GET /health`` before dispatching; transparent fallback to local CPU
  when the worker is offline (unless ``--strict-gpu``).
- Content-addressed dataset cache: the dataset fingerprint is used as the
  remote dataset id, so re-training on the same data skips the upload.
- GPU-aware defaults: AutoBatch for detection and an input resolution derived
  from the dataset's native image size (small objects such as wheat heads gain a
  lot from 1024 px instead of 640 px).
- Follows the job by ``job_id`` (robust to other jobs in the shared FIFO queue)
  and streams logs with a monotonic cursor.
- Downloads ``.pt`` / ``.meta.json`` back into ``models/`` and then deletes
  them from the worker so the GPU host's disk never fills up.
- Never prints the worker's network address (it is internal infrastructure).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_CONSECUTIVE_POLL_ERRORS = 120  # ~4 minutes at 2s polling


def _log(msg: str) -> None:
    print(msg, flush=True)


def _redact(text: str, gpu_url: str) -> str:
    return text.replace(gpu_url, "[acelerador-remoto]") if gpu_url else text


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("AGRIFED_GPU_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def infer_detection_img_size(data_dir: Path, sample: int = 24) -> int:
    """Pick a YOLO input size from the dataset's native resolution (640..1280, /32)."""
    try:
        from PIL import Image
    except ImportError:
        return 640
    dims: list[int] = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(p) as im:
                dims.append(max(im.size))
        except OSError:
            continue
        if len(dims) >= sample:
            break
    if not dims:
        return 640
    native = statistics.median(dims)
    return int(min(1280, max(640, round(native / 32) * 32)))


def dataset_fingerprint(data_dir: Path) -> str:
    """Stable content fingerprint from relative paths, sizes and mtimes."""
    h = hashlib.sha256()
    for p in sorted(data_dir.rglob("*")):
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        h.update(str(p.relative_to(data_dir)).encode("utf-8", "replace"))
        h.update(f":{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:20]


def _build_local_fallback_cmd(args: argparse.Namespace) -> list[str]:
    if args.task == "detection":
        return [
            sys.executable, "scripts/train_detection.py",
            "--data-dir", args.data_dir,
            "--output", args.output,
            "--metadata", args.metadata,
            "--epochs", str(args.epochs),
            "--batch-size", str(max(1, args.batch_size)),
            "--img-size", str(args.img_size or 640),
            "--lr", str(args.lr),
            "--arch", args.arch if args.arch.lower().startswith("yolo") else "yolov8n",
        ]  # fmt: skip

    cmd = [
        sys.executable, "scripts/train_base.py",
        "--data-dir", args.data_dir,
        "--output", args.output,
        "--metadata", args.metadata,
        "--epochs", str(args.epochs),
        "--batch-size", str(max(1, args.batch_size)),
        "--lr", str(args.lr),
        "--arch", args.arch,
        "--target-client-tier", args.target_client_tier,
        "--tl-strategy", args.tl_strategy,
        "--loss", args.loss,
        "--aug", args.aug,
        "--max-samples", "0",
    ]  # fmt: skip
    if args.train_split is not None:
        cmd += [
            "--train-split", str(args.train_split),
            "--val-split", str(args.val_split if args.val_split is not None else 0.1),
            "--test-split", str(args.test_split if args.test_split is not None else 0.1),
        ]  # fmt: skip
    return cmd


def _run_local_fallback(args: argparse.Namespace) -> int:
    cmd = _build_local_fallback_cmd(args)
    _log(f"[*] [GPU-DISPATCH] Ejecutando entrenamiento local: {' '.join(cmd)}")
    return int(subprocess.run(cmd, check=False).returncode)


def _create_dataset_tar(data_dir: Path, tar_path: Path) -> None:
    """Uncompressed tar (images are already compressed) dereferencing symlinks."""
    with tarfile.open(tar_path, "w", dereference=True) as tf:
        for item in sorted(data_dir.iterdir()):
            tf.add(item, arcname=item.name)


def _ensure_remote_dataset(client: httpx.Client, gpu_url: str, data_dir: Path, slug: str) -> str:
    """Return the remote dataset id, uploading only when it is not cached yet."""
    dataset_id = f"ds_{slug[:40]}_{dataset_fingerprint(data_dir)}"
    probe = client.get(f"{gpu_url}/datasets/{dataset_id}")
    if probe.status_code == 200:
        _log(
            "[*] [GPU-DISPATCH] Dataset ya presente en la caché del acelerador; se omite la subida."
        )
        return dataset_id

    _log(f"[*] [GPU-DISPATCH] Empaquetando dataset '{data_dir.name}' para el acelerador...")
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_f:
        tmp_tar = Path(tmp_f.name)
    try:
        _create_dataset_tar(data_dir, tmp_tar)
        size_mb = tmp_tar.stat().st_size / (1024 * 1024)
        _log(f"[*] [GPU-DISPATCH] Transfiriendo {size_mb:.1f} MB al nodo de cómputo...")
        t0 = time.monotonic()
        with tmp_tar.open("rb") as f_in:
            resp = client.post(
                f"{gpu_url}/jobs/upload-dataset",
                data={"dataset_id": dataset_id},
                files={"archive": (f"{dataset_id}.tar", f_in, "application/x-tar")},
                timeout=httpx.Timeout(1800.0, connect=10.0),
            )
        resp.raise_for_status()
        elapsed = max(0.1, time.monotonic() - t0)
        _log(
            f"[*] [GPU-DISPATCH] Dataset transferido en {elapsed:.0f}s "
            f"({size_mb / elapsed:.1f} MB/s)."
        )
        return dataset_id
    finally:
        tmp_tar.unlink(missing_ok=True)


def _follow_job(client: httpx.Client, gpu_url: str, job_id: str) -> int:
    """Stream logs for ``job_id`` until it finishes. Returns its exit code."""
    log_cursor: int | None = None
    last_state = ""
    last_pos: int | None = None
    errors = 0
    while True:
        time.sleep(2.0)
        try:
            st = client.get(f"{gpu_url}/jobs/by-id/{job_id}").raise_for_status().json()
            state = st.get("state", "unknown")
            if state == "queued":
                pos = st.get("position")
                if pos != last_pos:
                    _log(f"[*] [GPU-DISPATCH] En cola del acelerador (posición #{pos})...")
                    last_pos = pos
            elif state == "running":
                if last_state != "running":
                    _log("[*] [GPU-DISPATCH] Job en ejecución en el acelerador.")
                params = {"since": log_cursor} if log_cursor is not None else {"since": 0}
                logs = client.get(f"{gpu_url}/jobs/status", params=params).raise_for_status().json()
                if logs.get("active_job_id") == job_id:
                    for line in logs.get("logs", []):
                        _log(f"[GPU] {line}")
                log_cursor = int(logs.get("total_log_lines", log_cursor or 0))
            elif state in ("succeeded", "failed"):
                # Flush the tail of the logs produced after the last poll
                if log_cursor is not None:
                    tail = (
                        client.get(f"{gpu_url}/jobs/status", params={"since": log_cursor})
                        .raise_for_status()
                        .json()
                    )
                    for line in tail.get("logs", []):
                        _log(f"[GPU] {line}")
                code = st.get("exit_code")
                return 0 if state == "succeeded" else int(code if code not in (None, 0) else 1)
            else:
                _log(
                    "[!] [GPU-DISPATCH] El acelerador ya no conoce el job (¿reinicio del worker?)."
                )
                return 1
            last_state = state
            errors = 0
        except (httpx.HTTPError, ValueError) as exc:
            errors += 1
            if errors in (1, 10) or errors % 30 == 0:
                _log(
                    f"[!] [GPU-DISPATCH] Aviso consultando el acelerador: "
                    f"{_redact(str(exc), gpu_url)}"
                )
            if errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                _log("[!] [GPU-DISPATCH] Acelerador inalcanzable de forma prolongada; abortando.")
                return 1


def _download_artifacts(client: httpx.Client, gpu_url: str, args: argparse.Namespace) -> int:
    out_weights = Path(args.output).resolve()
    out_meta = Path(args.metadata).resolve()
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    _log("[*] [GPU-DISPATCH] Descargando pesos (.pt) y metadatos desde el acelerador...")
    wanted = [
        (f"{args.slug}_base.pt", out_weights, True),
        (f"{args.slug}.meta.json", out_meta, True),
        (
            f"{args.slug}_evaluation_report.json",
            out_weights.parent / f"{args.slug}_evaluation_report.json",
            False,
        ),
        (
            f"{args.slug}_eval_preview.jpg",
            out_weights.parent / f"{args.slug}_eval_preview.jpg",
            False,
        ),
    ]
    for remote_name, local_path, required in wanted:
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        with client.stream(
            "GET",
            f"{gpu_url}/jobs/{args.slug}/artifacts/{remote_name}",
            timeout=httpx.Timeout(600.0, connect=10.0),
        ) as resp:
            if resp.status_code != 200:
                if required:
                    _log(
                        f"[!] [GPU-DISPATCH] Falta el artefacto '{remote_name}' "
                        f"(HTTP {resp.status_code})."
                    )
                    return 1
                continue
            with tmp_path.open("wb") as f_out:
                for chunk in resp.iter_bytes(1024 * 1024):
                    f_out.write(chunk)
        tmp_path.replace(local_path)  # atomic: never a truncated checkpoint in models/

    try:
        client.delete(f"{gpu_url}/jobs/{args.slug}/artifacts")
    except httpx.HTTPError:
        pass  # non-fatal: the worker keeps them until the next cleanup
    _log(
        f"[+] [GPU-DISPATCH] Artefactos recibidos en '{out_weights.name}' "
        f"({out_weights.stat().st_size / (1024 * 1024):.2f} MB)."
    )
    return 0


def dispatch_to_gpu(args: argparse.Namespace) -> int:
    gpu_url = (
        (
            args.gpu_url
            or os.environ.get("AGRIFED_GPU_WORKER_URL")
            or os.environ.get("AGRIFED_GPU_ENDPOINT")
            or ""
        )
        .strip()
        .rstrip("/")
    )
    if not gpu_url:
        return _run_local_fallback(args)

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        _log(f"[!] [GPU-DISPATCH] Directorio de datos no existe: {data_dir}")
        return 1

    client = httpx.Client(headers=_auth_headers(), timeout=httpx.Timeout(30.0, connect=5.0))
    try:
        try:
            health: dict[str, Any] = client.get(f"{gpu_url}/health").raise_for_status().json()
        except (httpx.HTTPError, ValueError) as exc:
            _log(
                f"[!] [GPU-DISPATCH] Acelerador remoto no disponible "
                f"({_redact(str(exc), gpu_url)})."
            )
            if args.strict_gpu:
                return 1
            _log("[*] [GPU-DISPATCH] Fallback transparente a CPU local...")
            return _run_local_fallback(args)

        vram_free = health.get("vram_system_free_mb")
        _log(
            f"[*] [GPU-DISPATCH] Acelerador remoto conectado | {health.get('device_name', 'GPU')} "
            f"(VRAM {health.get('vram_total_gb', 0)} GB, libre {vram_free} MiB, "
            f"ocupado={health.get('is_busy', False)}, cola={health.get('queue_length', 0)})"
        )

        is_det = args.task == "detection"
        img_size = args.img_size or (infer_detection_img_size(data_dir) if is_det else 224)
        batch_size = 0 if (is_det and not args.fixed_batch) else max(1, args.batch_size)
        if is_det:
            _log(
                f"[*] [GPU-DISPATCH] Perfil GPU detección: imgsz={img_size}px, "
                f"batch={'AutoBatch' if batch_size == 0 else batch_size}."
            )

        try:
            dataset_id = (
                args.data_dir
                if args.shared_fs
                else _ensure_remote_dataset(client, gpu_url, data_dir, args.slug)
            )
            payload: dict[str, Any] = {
                "slug": args.slug,
                "task": args.task,
                "dataset_path": dataset_id,
                "architecture": args.arch,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "lr": args.lr,
                "img_size": img_size,
                "target_client_tier": args.target_client_tier,
                "transfer_learning": args.tl_strategy,
                "loss_function": args.loss,
                "augmentation": args.aug,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "test_split": args.test_split,
            }
            job_info = client.post(f"{gpu_url}/jobs/train", json=payload).raise_for_status().json()
        except (httpx.HTTPError, ValueError) as exc:
            _log(f"[!] [GPU-DISPATCH] Error preparando el job remoto: {_redact(str(exc), gpu_url)}")
            if args.strict_gpu:
                return 1
            return _run_local_fallback(args)

        job_id = job_info.get("job_id", "")
        _log(
            f"[*] [GPU-DISPATCH] Job remoto registrado ({job_info.get('status')}, "
            f"arquitectura={job_info.get('architecture')}, id={job_id})."
        )

        exit_code = _follow_job(client, gpu_url, job_id)
        if exit_code != 0:
            _log(f"[!] [GPU-DISPATCH] El entrenamiento en GPU finalizó con código {exit_code}.")
            return exit_code
        return _download_artifacts(client, gpu_url, args)
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch training job to remote agrifed-gpu worker"
    )
    default_url = (
        os.environ.get("AGRIFED_GPU_WORKER_URL") or os.environ.get("AGRIFED_GPU_ENDPOINT") or ""
    )
    parser.add_argument("--gpu-url", type=str, default=default_url)
    parser.add_argument("--slug", type=str, required=True)
    parser.add_argument(
        "--task", type=str, choices=["classification", "detection"], default="classification"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--fixed-batch", action="store_true", help="Disable AutoBatch for detection"
    )
    parser.add_argument(
        "--img-size", type=int, default=0, help="0 = infer from dataset (detection)"
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--arch", type=str, default="auto")
    parser.add_argument("--target-client-tier", type=str, default="gpu")
    parser.add_argument("--tl-strategy", type=str, default="partial")
    parser.add_argument("--loss", type=str, default="focal")
    parser.add_argument("--aug", type=str, default="in_the_wild")
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--test-split", type=float, default=None)
    parser.add_argument(
        "--shared-fs", action="store_true", help="Dataset path is already on the worker"
    )
    parser.add_argument(
        "--strict-gpu", action="store_true", help="Fail instead of falling back to CPU"
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(dispatch_to_gpu(parse_args()))
