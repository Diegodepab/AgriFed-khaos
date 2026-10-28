"""Comprehensive tests for KHAOS Hub Generic Dynamic Model Catalog & Thumbnail System.

Verifies all 12 test scenarios from Section 10:
1. Model with valid thumbnail.
2. Model without thumbnail (placeholder rendered).
3. Model with empty / null thumbnail URL.
4. Thumbnail loading error fallback handling.
5. Model with missing optional metadata (graceful defaults).
6. Model with zero detectable classes.
7. Multiple models across different domains/categories.
8. Dynamic counter updates on search and filter.
9. Filter reset ("Limpiar todos").
10. Empty results state.
11. Card and list view rendering consistency.
12. Absence of hardcoded "olivo"/"aguacate" specific conditions in generic rendering.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

from agrifed_khaos.catalog import LogicalModelEntry, ModelCatalogManager
from agrifed_khaos.models import build_model, save_checkpoint_bundle
from scripts.serve import create_app


def _create_dummy_checkpoint(path: Path, num_classes: int = 2) -> None:
    """Helper to generate a lightweight valid model checkpoint bundle."""
    m = build_model("resnet18", num_classes=num_classes, pretrained=False)
    save_checkpoint_bundle(
        path,
        m.state_dict(),
        {
            "architecture": "resnet18",
            "num_classes": num_classes,
            "classes": [f"class_{i}" for i in range(num_classes)],
            "img_size": [32, 32],
        },
    )


def test_1_model_with_valid_thumbnail(tmp_path: Path) -> None:
    """Scenario 1: Model with valid thumbnail served via endpoint."""
    model_p = tmp_path / "model1.pt"
    _create_dummy_checkpoint(model_p)

    # Create thumbnail directory and image
    thumb_dir = tmp_path / "models" / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_img = thumb_dir / "test-model.jpg"
    Image.new("RGB", (64, 64), color="green").save(thumb_img, format="JPEG")

    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    entry = LogicalModelEntry(
        slug="test-model",
        title="Valid Thumbnail Model",
        domain="Agricultura",
        crop="Vid",
        target_element="Hoja",
        purpose="Detección de mildiu",
        active_checkpoint=model_p.name,
        thumbnail_url="/api/models/test-model/thumbnail",
    )
    cat_mgr.save_catalog([entry])

    app = create_app(str(model_p), str(tmp_path / "meta.json"))
    app.state.catalog = cat_mgr
    client = TestClient(app)

    # 1. API list returns thumbnail_url
    r_list = client.get("/api/models")
    assert r_list.status_code == 200
    item = next(m for m in r_list.json()["models"] if m["slug"] == "test-model")
    assert item["thumbnail_url"] == "/api/models/test-model/thumbnail"

    # 2. Thumbnail endpoint returns image with correct content type
    # Mocking cand file in root relative path
    thumb_img_app = Path("models/thumbnails/test-model.jpg")
    thumb_img_app.parent.mkdir(parents=True, exist_ok=True)
    try:
        thumb_img_app.write_bytes(thumb_img.read_bytes())
        r_thumb = client.get("/api/models/test-model/thumbnail")
        assert r_thumb.status_code == 200
        assert r_thumb.headers["content-type"] in ("image/jpeg", "image/jpg")
    finally:
        thumb_img_app.unlink(missing_ok=True)


def test_2_model_without_thumbnail(tmp_path: Path) -> None:
    """Scenario 2: Model without thumbnail returns None and triggers Case B placeholder."""
    model_p = tmp_path / "model2.pt"
    _create_dummy_checkpoint(model_p)

    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    entry = LogicalModelEntry(
        slug="no-thumb-model",
        title="Model Without Thumbnail",
        domain="General",
        active_checkpoint=model_p.name,
        thumbnail_url=None,
    )
    cat_mgr.save_catalog([entry])

    app = create_app(str(model_p), str(tmp_path / "meta.json"))
    app.state.catalog = cat_mgr
    client = TestClient(app)

    r_list = client.get("/api/models")
    assert r_list.status_code == 200
    item = next(m for m in r_list.json()["models"] if m["slug"] == "no-thumb-model")
    assert item["thumbnail_url"] is None

    # Check that client template contains the Case B placeholder renderer
    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "getThumbnailPlaceholderHtml" in r_html.text
    assert "KHAOS Model Hub" in r_html.text


def test_3_thumbnail_with_empty_or_null_url(tmp_path: Path) -> None:
    """Scenario 3: Model with empty string or None thumbnail_url validates cleanly."""
    e_null = LogicalModelEntry(
        slug="m-null",
        title="Null Model",
        active_checkpoint="m.pt",
        thumbnail_url=None,
    )
    e_empty = LogicalModelEntry(
        slug="m-empty",
        title="Empty Model",
        active_checkpoint="m.pt",
        thumbnail_url="",
    )
    assert e_null.thumbnail_url is None
    assert e_empty.thumbnail_url == ""

    # Serialization roundtrip
    dump_null = e_null.model_dump()
    dump_empty = e_empty.model_dump()
    assert dump_null["thumbnail_url"] is None
    assert dump_empty["thumbnail_url"] == ""


def test_4_thumbnail_load_error_handling(tmp_path: Path) -> None:
    """Scenario 4: Missing thumbnail file returns 404, and template handles fallback."""
    model_p = tmp_path / "model4.pt"
    _create_dummy_checkpoint(model_p)

    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    entry = LogicalModelEntry(
        slug="error-thumb-model",
        title="Broken Thumbnail Model",
        active_checkpoint=model_p.name,
        thumbnail_url="/api/models/error-thumb-model/thumbnail",
    )
    cat_mgr.save_catalog([entry])

    app = create_app(str(model_p), str(tmp_path / "meta.json"))
    app.state.catalog = cat_mgr
    client = TestClient(app)

    # 404 for missing image on disk
    r_thumb = client.get("/api/models/error-thumb-model/thumbnail")
    assert r_thumb.status_code == 404

    # Template defines handleThumbnailError with console.warn and onerror=null
    r_html = client.get("/")
    assert "handleThumbnailError" in r_html.text
    assert "console.warn" in r_html.text
    assert "img.onerror = null" in r_html.text


def test_5_model_with_missing_optional_metadata(tmp_path: Path) -> None:
    """Scenario 5: Model with absent optional metadata uses neutral graceful defaults."""
    entry = LogicalModelEntry(
        slug="bare-model",
        title="Bare Minimum Model",
        active_checkpoint="chk.pt",
    )
    assert entry.domain == "General"
    assert entry.crop == ""
    assert entry.target_element == ""
    assert entry.purpose == ""
    assert entry.description == ""
    assert entry.classes == []
    assert entry.num_classes == 0
    assert entry.thumbnail_url is None
    assert entry.calibrated_thresholds is None


def test_6_model_with_zero_detectable_classes(tmp_path: Path) -> None:
    """Scenario 6: Model with 0 detectable classes handles without layout breakage."""
    entry = LogicalModelEntry(
        slug="zero-class-model",
        title="Zero Classes Model",
        active_checkpoint="chk0.pt",
        num_classes=0,
        classes=[],
    )
    dumped = entry.model_dump()
    assert dumped["num_classes"] == 0
    assert dumped["classes"] == []

    # Check client template handles 0 classes cleanly
    html = Path("src/agrifed_khaos/templates/client.html").read_text(encoding="utf-8")
    assert "Sin clases catalogadas" in html


def test_7_multiple_models_across_different_domains(tmp_path: Path) -> None:
    """Scenario 7: Catalog indexes and filters across diverse domains."""
    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    m_agro = LogicalModelEntry(
        slug="agro-1",
        title="Agro Model",
        domain="Agricultura",
        crop="Cereal",
        active_checkpoint="a.pt",
    )
    m_ind = LogicalModelEntry(
        slug="ind-1",
        title="Industrial Defect Detection",
        domain="Inspección Industrial",
        crop="Metalurgia",
        active_checkpoint="b.pt",
    )
    m_med = LogicalModelEntry(
        slug="med-1",
        title="Medical Retina Scanner",
        domain="Medicina",
        active_checkpoint="c.pt",
    )
    cat_mgr.save_catalog([m_agro, m_ind, m_med])

    # Filter by domain
    res_ind = cat_mgr.search_and_paginate(domain="Inspección Industrial")
    assert res_ind["total"] == 1
    assert res_ind["models"][0]["slug"] == "ind-1"

    res_med = cat_mgr.search_and_paginate(domain="Medicina")
    assert res_med["total"] == 1
    assert res_med["models"][0]["slug"] == "med-1"

    # All domains aggregated
    all_res = cat_mgr.search_and_paginate()
    assert all_res["total"] == 3
    assert "Inspección Industrial" in all_res["domains"]
    assert "Medicina" in all_res["domains"]
    assert "Agricultura" in all_res["domains"]


def test_8_dynamic_counter_updates_on_search_and_filter(tmp_path: Path) -> None:
    """Scenario 8: Search and filtering dynamically recalculate total count."""
    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    m1 = LogicalModelEntry(
        slug="soldaduras-tubos",
        title="Detección de Defectos en Soldaduras",
        domain="Industrial",
        active_checkpoint="s.pt",
        classes=["poro", "fisura"],
    )
    m2 = LogicalModelEntry(
        slug="radiografias-torax",
        title="Segmentación Pulmonar en Radiografías",
        domain="Salud",
        active_checkpoint="r.pt",
        classes=["neumonia", "normal"],
    )
    cat_mgr.save_catalog([m1, m2])

    # Search for soldaduras
    res_s = cat_mgr.search_and_paginate(q="soldaduras")
    assert res_s["total"] == 1
    assert res_s["models"][0]["slug"] == "soldaduras-tubos"

    # Search for class "poro"
    res_p = cat_mgr.search_and_paginate(q="poro")
    assert res_p["total"] == 1

    # Search for non-existent
    res_none = cat_mgr.search_and_paginate(q="termino_inexistente")
    assert res_none["total"] == 0


def test_9_filter_reset_logic(tmp_path: Path) -> None:
    """Scenario 9: Resetting filters restores full active models collection."""
    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    m1 = LogicalModelEntry(slug="m1", title="Model 1", active_checkpoint="1.pt")
    m2 = LogicalModelEntry(slug="m2", title="Model 2", active_checkpoint="2.pt")
    cat_mgr.save_catalog([m1, m2])

    filtered = cat_mgr.search_and_paginate(domain="Inexistente")
    assert filtered["total"] == 0

    reset = cat_mgr.search_and_paginate(domain=None, q=None)
    assert reset["total"] == 2


def test_10_empty_results_state(tmp_path: Path) -> None:
    """Scenario 10: Search with no matching items produces empty result and UI state."""
    cat_mgr = ModelCatalogManager(
        catalog_path=tmp_path / "catalog.json", models_dir=tmp_path
    )
    m1 = LogicalModelEntry(slug="m1", title="Alpha Model", active_checkpoint="1.pt")
    cat_mgr.save_catalog([m1])

    res = cat_mgr.search_and_paginate(q="zzzzzzzz")
    assert res["total"] == 0
    assert len(res["models"]) == 0

    # Verify template contains empty state container and zero count messaging
    html = Path("src/agrifed_khaos/templates/client.html").read_text(encoding="utf-8")
    assert 'id="emptyCatalogState"' in html
    assert "0 modelos encontrados" in html


def test_11_card_and_list_view_rendering_consistency() -> None:
    """Scenario 11: Three view modes (Grid, Specs, List) exist and expose identical actions."""
    html = Path("src/agrifed_khaos/templates/client.html").read_text(encoding="utf-8")
    assert "function renderGridView" in html
    assert "function renderSpecsView" in html
    assert "function renderListView" in html
    assert "setViewMode('grid')" in html
    assert "setViewMode('specs')" in html
    assert "setViewMode('list')" in html
    assert "openAnalysisStudioBySlug" in html
    assert "Ver detalles &rarr;" in html

    # Verify single help trigger without redundant duplicates
    assert html.count("toggleHelpModal(true)") == 1
    assert "sidebarEmptyNotice" in html
    assert "facetCategorySection" in html
    assert "facetSampleSection" in html
    assert "facetPurposeSection" in html


def test_12_absence_of_hardcoded_examples_in_generic_catalog() -> None:
    """Scenario 12: Generic catalog template contains zero hardcoded olive/avocado conditions."""
    html = Path("src/agrifed_khaos/templates/client.html").read_text(encoding="utf-8")
    lower_html = html.lower()
    # Confirm complete removal of hardcoded conditional branching on specific crops
    assert "descriptive_synonyms" not in lower_html
    assert "olivo" not in lower_html
    assert "aguacate" not in lower_html
