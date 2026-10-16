"""Cryptographic provenance, hashing, and HMAC authentication for AgriFed-KHAOS.

Provides tamper-evident signatures over model checkpoints and manifest metadata,
guaranteeing lineage, scope isolation, and sample integrity across distributed nodes.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for cryptographic provenance verification."""
    sha = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_manifest_hmac(
    key: str,
    client_sha256: str,
    base_model_sha256: str,
    training_scope: str,
    client_id: str = "client",
    num_samples: int | None = None,
) -> str:
    """Compute HMAC-SHA256 signature over a canonical payload of critical manifest metadata.

    Guarantees tamper-evidence across weights, base lineage, scope, client identity,
    and sample weighting.

    Validates against delimiter injection (RFC 2104 compliant canonical payload).
    """
    if "|" in client_id:
        raise ValueError(f"client_id cannot contain delimiter '|': {client_id!r}")
    if "|" in training_scope:
        raise ValueError(f"training_scope cannot contain delimiter '|': {training_scope!r}")

    payload_parts = [
        f"sha256={client_sha256}",
        f"base_sha256={base_model_sha256}",
        f"scope={training_scope}",
        f"client_id={client_id}",
    ]
    if num_samples is not None:
        payload_parts.append(f"num_samples={num_samples}")
    canonical_payload = "|".join(payload_parts)

    return hmac.new(
        key.encode("utf-8"),
        canonical_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_manifest_hmac(
    key: str,
    manifest: dict[str, object],
    actual_client_sha256: str,
) -> bool:
    """Verify a manifest signature against the received checkpoint digest."""
    signature = manifest.get("hmac_signature")
    base_sha = manifest.get("base_model_sha256")
    scope = manifest.get("training_scope")
    client_id = manifest.get("client_id")
    num_samples = manifest.get("num_samples")
    if not isinstance(signature, str):
        return False
    if not isinstance(base_sha, str) or not isinstance(scope, str):
        return False
    if not isinstance(client_id, str):
        return False
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        return False
    try:
        expected = compute_manifest_hmac(
            key=key,
            client_sha256=actual_client_sha256,
            base_model_sha256=base_sha,
            training_scope=scope,
            client_id=client_id,
            num_samples=num_samples,
        )
    except ValueError:
        return False
    return hmac.compare_digest(signature, expected)

