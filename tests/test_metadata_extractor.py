"""Unit tests for metadata extraction, narrative variety, and thumbnail generation."""

# ruff: noqa: E501

from pathlib import Path

from agrifed_khaos.metadata import (
    _is_random_or_opaque_filename,
    clean_filename_title,
    format_class_display_name,
    generate_narrative_description,
    infer_dataset_title_and_slug,
    infer_domain,
    is_healthy_class,
    slugify,
    suggest_dataset_metadata,
)


def test_opaque_filename_detection():
    # Opaque / generic names that should trigger class-driven inference
    assert _is_random_or_opaque_filename("archive.zip") is True
    assert _is_random_or_opaque_filename("archive (1).zip") is True
    assert _is_random_or_opaque_filename("data.zip") is True
    assert _is_random_or_opaque_filename("dataset.zip") is True
    assert _is_random_or_opaque_filename("upload_1793096246_d45a6e7e") is True
    assert _is_random_or_opaque_filename("images.tar.gz") is True
    assert _is_random_or_opaque_filename("train") is True
    assert _is_random_or_opaque_filename("IMG_20240918_123456.jpg") is True

    # Meaningful filenames that should be preserved and cleaned
    assert _is_random_or_opaque_filename("enfermedades_hoja_olivo.zip") is False
    assert _is_random_or_opaque_filename("inspeccion_defectos_piezas.zip") is False
    assert _is_random_or_opaque_filename("mango_fruit_quality.zip") is False


def test_clean_filename_title_and_slug():
    assert clean_filename_title("enfermedades_hoja_v1.0.zip") == "Enfermedades Hoja v1.0"
    assert clean_filename_title("inspeccion_defectos_piezas-1.zip") == "Inspeccion Defectos Piezas"
    assert clean_filename_title("mango_fruit_quality-800x600.zip") == "Mango Fruit Quality"

    assert slugify("Enfermedades Hoja Olivo v1.0") == "enfermedades-hoja-olivo-v10"
    assert slugify("Detección de Crack & Scratch") == "deteccion-de-crack-scratch"


def test_format_class_display_name():
    assert format_class_display_name("aculus_olearius") == "Aculus Olearius"
    assert format_class_display_name("olive_peacock_spot") == "Olive Peacock Spot"
    assert format_class_display_name("cochinilla_tizne") == "Cochinilla Tizne"
    assert format_class_display_name("sano") == "Sano"


def test_is_healthy_class():
    assert is_healthy_class("sano") is True
    assert is_healthy_class("Sana") is True
    assert is_healthy_class("Healthy") is True
    assert is_healthy_class("control_normal") is True
    assert is_healthy_class("baseline") is True

    assert is_healthy_class("aculus_olearius") is False
    assert is_healthy_class("olive_peacock_spot") is False
    assert is_healthy_class("crack") is False
    assert is_healthy_class("scratch") is False


def test_infer_dataset_title_and_slug_from_classes():
    # User's case: Healthy, Aculus Olearius, Olive Peacock Spot with archive (1).zip
    title, slug = infer_dataset_title_and_slug(
        source_name="archive (1).zip",
        classes=["Healthy", "Aculus Olearius", "Olive Peacock Spot"],
        image_counts={"Healthy": 830, "Aculus Olearius": 690, "Olive Peacock Spot": 1200},
    )
    assert title == "Detección de Olive Peacock Spot y Aculus Olearius"
    assert slug == "deteccion-de-olive-peacock-spot-y-aculus-olearius"

    # Single defect + healthy
    title_single, _ = infer_dataset_title_and_slug(
        source_name="data.zip",
        classes=["sano", "roya"],
    )
    assert title_single == "Detección de Roya"

    # Meaningful filename overrides class inference
    title_meaningful, slug_meaningful = infer_dataset_title_and_slug(
        source_name="enfermedades_del_olivo.zip",
        classes=["sano", "repilo"],
    )
    assert title_meaningful == "Enfermedades del Olivo"
    assert slug_meaningful == "enfermedades-del-olivo"


def test_narrative_variety_and_determinism():
    # Test that narrative generation is non-empty, contains contextual class names,
    # and is deterministic for the same dataset while varied across different datasets.
    desc_a1 = generate_narrative_description(
        dataset_path="data/dataset_a/train",
        classes=["Healthy", "Aculus Olearius", "Olive Peacock Spot"],
        total_images=2720,
    )
    desc_a2 = generate_narrative_description(
        dataset_path="data/dataset_a/train",
        classes=["Healthy", "Aculus Olearius", "Olive Peacock Spot"],
        total_images=2720,
    )
    assert desc_a1 == desc_a2
    assert "Olive Peacock Spot" in desc_a1
    assert "Healthy" in desc_a1

    # Different dataset path / counts should produce varied phrasing
    desc_b = generate_narrative_description(
        dataset_path="data/industrial_b/train",
        classes=["Normal", "Crack", "Dent", "Scratch"],
        total_images=5400,
    )
    assert desc_b != desc_a1
    assert "Crack" in desc_b


def test_domain_inference():
    # Agro
    assert infer_domain(["Healthy", "Aculus Olearius", "Olive Peacock Spot"]) == "Agroalimentario & Agricultura"
    assert infer_domain(["sano", "hoja_olivo_repilo"]) == "Agroalimentario & Agricultura"

    # Industry
    assert infer_domain(["normal", "crack", "dent", "scratch"]) == "Industria & Manufactura"
    assert infer_domain(["soldadura_buena", "soldadura_poro"]) == "Industria & Manufactura"

    # Healthcare
    assert infer_domain(["normal", "tumor", "skin_lesion"]) == "Biomedicina & Sanidad"

    # Environment
    assert infer_domain(["forest", "wildfire_smoke"]) == "Medio Ambiente & Recursos"

    # Generic
    assert infer_domain(["clase_a", "clase_b"]) == "Visión Artificial"


def test_suggest_dataset_metadata_with_thumbnail(tmp_path: Path):
    from PIL import Image

    # Create dummy dataset directory structure
    cls_healthy = tmp_path / "Healthy"
    cls_healthy.mkdir()
    cls_defect = tmp_path / "Olive Peacock Spot"
    cls_defect.mkdir()

    # Create dummy images
    img_h = Image.new("RGB", (200, 200), color=(0, 200, 0))
    img_h.save(str(cls_healthy / "h1.jpg"))

    img_d = Image.new("RGB", (300, 300), color=(200, 50, 50))
    img_d.save(str(cls_defect / "d1.jpg"))

    meta = suggest_dataset_metadata(
        dataset_path=str(tmp_path),
        classes=["Healthy", "Olive Peacock Spot"],
        total_images=2,
        image_counts={"Healthy": 1, "Olive Peacock Spot": 1},
        source_name="archive.zip",
    )

    assert meta["suggested_title"] == "Detección de Olive Peacock Spot"
    assert meta["suggested_slug"] == "deteccion-de-olive-peacock-spot"
    assert meta["suggested_domain"] == "Agroalimentario & Agricultura"
    assert "Olive Peacock Spot" in meta["suggested_description"]
    assert meta["thumbnail_b64"] is not None
    assert meta["thumbnail_b64"].startswith("data:image/webp;base64,")
