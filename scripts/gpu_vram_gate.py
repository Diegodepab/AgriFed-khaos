#!/usr/bin/env python3
"""VRAM admission gate for shared GPU hosts.

Runs as the first step of every GPU job in ``agrifed-gpu``. It blocks until the
physical GPU has at least ``--min-free-mb`` of free memory (as reported by
``nvidia-smi``, i.e. across *all* containers on the host) for two consecutive
probes, so a training job never starts while a neighbouring service (for example
another team's inference server) is holding the VRAM it needs.

It never touches CUDA itself, so it adds zero VRAM footprint.

Exit codes: 0 = admitted, 3 = timed out waiting for VRAM.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time


def query_free_mb() -> tuple[float, float] | None:
    """Return (free_mb, total_mb) for GPU 0, or None if nvidia-smi is unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    try:
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for free VRAM before a GPU job")
    parser.add_argument("--min-free-mb", type=float, default=6000.0)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-s", type=float, default=10.0)
    args = parser.parse_args()

    first = query_free_mb()
    if first is None:
        print("[*] [VRAM-GATE] nvidia-smi no disponible; se omite el control de VRAM.", flush=True)
        return 0

    deadline = time.monotonic() + max(0.0, args.timeout_s)
    consecutive_ok = 0
    last_report = 0.0
    while True:
        probe = query_free_mb()
        free_mb, total_mb = probe if probe is not None else first
        if free_mb >= args.min_free_mb:
            consecutive_ok += 1
            if consecutive_ok >= 2:
                print(
                    f"[*] [VRAM-GATE] VRAM libre {free_mb:.0f}/{total_mb:.0f} MiB "
                    f"(mínimo {args.min_free_mb:.0f} MiB). Job admitido.",
                    flush=True,
                )
                return 0
        else:
            consecutive_ok = 0
            now = time.monotonic()
            if now - last_report >= 60.0:
                print(
                    f"[*] [VRAM-GATE] GPU ocupada por otro servicio: libre {free_mb:.0f} MiB "
                    f"< {args.min_free_mb:.0f} MiB requeridos. Esperando...",
                    flush=True,
                )
                last_report = now

        if time.monotonic() >= deadline:
            print(
                f"[!] [VRAM-GATE] Tiempo de espera agotado ({args.timeout_s:.0f}s) sin VRAM libre.",
                flush=True,
            )
            return 3
        time.sleep(max(1.0, args.poll_s))


if __name__ == "__main__":
    sys.exit(main())
