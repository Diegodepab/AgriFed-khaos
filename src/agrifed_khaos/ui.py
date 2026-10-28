"""Web user interface template loader for KHAOS Federated Model Hub."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_agnostic_client_page() -> str:
    """Return HTML string for GET / (Data Space Model Hub Catalog)."""
    return (TEMPLATES_DIR / "client.html").read_text(encoding="utf-8")


def render_model_consumer_page(slug: str) -> str:
    """Return HTML string for GET /models/{slug} (Dedicated Consumer Model Page)."""
    content = (TEMPLATES_DIR / "model_detail.html").read_text(encoding="utf-8")
    return content.replace("{{MODEL_SLUG}}", slug)


def render_expert_page(slug: str | None = None) -> str:
    """Return HTML string for GET /expert or GET /models/{slug}/expert."""
    content = (TEMPLATES_DIR / "expert.html").read_text(encoding="utf-8")
    return content.replace("{{MODEL_SLUG}}", slug or "")


def render_master_admin_page() -> str:
    """Return HTML string for GET /master (Data Space Governance Console)."""
    return (TEMPLATES_DIR / "master.html").read_text(encoding="utf-8")


def render_model_create_page() -> str:
    """Return HTML string for GET /master/models/new (Model Creation Studio)."""
    return (TEMPLATES_DIR / "model_create.html").read_text(encoding="utf-8")
