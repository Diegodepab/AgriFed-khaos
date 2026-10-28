"""Metadata extraction, narrative generation, and thumbnail processing for datasets and models.

Inspired by MetaDataXtract architecture:
- Deterministic narrative variety without external LLM dependencies.
- Intelligent separation of healthy baseline vs defect/anomaly classes.
- Opaque filename detection and class-driven title & slug inference.
- Lightweight embedded WebP Base64 dataset thumbnail generation.
- Domain inference across agricultural, industrial, healthcare, environmental, and general verticals.
"""

# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

# ==============================================================================
# PATTERNS & KEYWORDS
# ==============================================================================

_OPAQUE_FILENAME_PATTERNS = [
    re.compile(r"^(?:img|dsc|p|dcim|vid|mov|wp|dji|gopr|gp|sam)[-_0-9]+$", re.IGNORECASE),
    re.compile(r"^(?:screenshot|captura(?:_de_pantalla)?|pasted[-_]?image|screen[-_]?shot)[-_0-9.\s]*$", re.IGNORECASE),
    re.compile(
        r"^(?:(?:frame|image|images|photo|photos|foto|fotos|video|clip|untitled|archivo|archivos|file|files|test|sample|prueba|dummy|dataset|datasets|data|archive|archives|upload|uploads|train|val|valid|test)[-_0-9.\s()]*)+$",
        re.IGNORECASE,
    ),
    re.compile(r"^upload(?:s)?[-_0-9a-fA-F]+$", re.IGNORECASE),
    re.compile(r"^[0-9a-f]{12,64}$", re.IGNORECASE),
    re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE),
]

HEALTHY_KEYWORDS = frozenset({
    "sano", "sana", "sanos", "sanas",
    "healthy", "health",
    "normal", "normale", "normales",
    "control", "baseline", "background",
    "negativo", "negative",
    "bueno", "buena", "ok", "good",
    "sin_sintomas", "sin_danos",
})

AGRO_KEYWORDS = frozenset({
    "hoja", "hojas", "leaf", "leaves",
    "fruto", "frutos", "fruit", "fruits",
    "olivo", "oliva", "olive", "olives", "aceituna", "aceitunas",
    "mango", "mora", "nuez", "nueces", "almendra", "almendras",
    "crop", "crops", "planta", "plantas", "plant", "plants",
    "trigo", "wheat", "maiz", "corn", "arroz", "rice",
    "pest", "pests", "plaga", "plagas",
    "rust", "roya", "repilo", "antracnosis", "fumagina", "virosis",
    "tuberculosis", "olearius", "aculus", "tizne", "mildiu", "oidio",
    "uva", "grape", "vid", "vineyard", "weed", "malezas",
    "citrus", "citrico", "naranja", "orange", "limon", "lemon",
    "patata", "potato", "tomate", "tomato", "cafe", "coffee",
    "sano", "sana", "agricola", "agronomia", "cultivo",
})

INDUSTRY_KEYWORDS = frozenset({
    "crack", "cracks", "fisura", "grieta", "grietas",
    "dent", "dents", "abolladura", "abolladuras",
    "scratch", "scratches", "rayado", "arana", "aranazo",
    "metal", "metalico", "pieza", "piezas", "part", "component",
    "soldadura", "weld", "welding",
    "defect", "defects", "defecto", "defectos", "flaw", "flaws",
    "screw", "tornillo", "pcb", "chip", "board", "circuito",
    "fabric", "textil", "superficie", "surface",
    "tubo", "pipe", "bearing", "rodamiento", "engranaje", "gear",
    "manufacturing", "manufactura", "industrial", "maquina",
})

HEALTH_KEYWORDS = frozenset({
    "cell", "celula", "celulas", "xray", "rayos", "radiografia",
    "mri", "resonancia", "ct", "scanner",
    "tumor", "tumores", "skin", "piel", "lesion", "lesiones",
    "cancer", "melanoma", "retina", "fondo_ojo",
    "covid", "pulmon", "pulmones", "lung", "tissue", "tejido",
    "histopatologia", "biopsia", "ecografia", "clinica", "medico",
})

ENVIRONMENT_KEYWORDS = frozenset({
    "forest", "bosque", "forestal", "arbol", "tree", "trees",
    "agua", "water", "rio", "river", "glacier", "glaciar",
    "satellite", "satelite", "sentinel", "landsat",
    "vegetacion", "vegetation", "wildfire", "incendio", "fuego",
    "fauna", "animal", "biodiversidad", "ocean", "marino",
})


# ==============================================================================
# FORMATTING & CLEANING HELPERS
# ==============================================================================

def _format_title_word(w: str) -> str:
    """Format an individual word keeping acronyms and version strings clean."""
    if re.match(r"^v?\d+(\.\d+)*$", w, re.IGNORECASE):
        return w.lower()
    if any(c.isupper() for c in w[1:]) and any(c.islower() for c in w):
        return w
    if w.isupper() and len(w) > 1:
        return w
    if w.lower() == "co2":
        return "CO2"
    if w.lower() in ("de", "del", "en", "el", "la", "los", "las", "y", "o", "a", "al", "por", "con", "sin"):
        return w.lower()
    return w.capitalize()


KNOWN_ARCHIVE_EXTS = (
    ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pt", ".pth", ".onnx", ".bin",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
)


def clean_filename_title(name: str) -> str:
    """Transform raw filename into a human-friendly title.

    Strips extensions, resolution stamps (e.g. -350x464), trailing counter indices,
    and applies smart capitalization.
    """
    base = os.path.basename(name).strip()
    for ext in KNOWN_ARCHIVE_EXTS:
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    if not base:
        return "Dataset"
    # Remove resolution stamps
    base = re.sub(r"[-_](?:\d+[-_])?\d+x\d+$", "", base, flags=re.IGNORECASE)
    # Remove trailing catalog index like -1, _1 if preceded by alphanumeric letters
    base = re.sub(r"([a-zA-ZáéíóúÁÉÍÓÚñÑ])[-_]\d+$", r"\1", base)
    words = re.split(r"[-_\s]+", base)
    words_formatted = [_format_title_word(w) for w in words if w]
    if words_formatted:
        # First word should always be capitalized
        words_formatted[0] = words_formatted[0].capitalize()
    cleaned = " ".join(words_formatted).strip()
    return cleaned if cleaned else base.title()


def _is_random_or_opaque_filename(name: str) -> bool:
    """Detect camera counters, screen capture stamps, hashes, or generic names."""
    base = os.path.basename(name).strip()
    for ext in KNOWN_ARCHIVE_EXTS:
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    if not base or len(base) <= 3:
        return True
    for pat in _OPAQUE_FILENAME_PATTERNS:
        if pat.match(base):
            return True
    lower = base.lower()
    vowels = sum(1 for c in lower if c in "aeiou")
    consonants = sum(1 for c in lower if c.isalpha() and c not in "aeiou")
    total_letters = consonants + vowels
    if total_letters >= 7:
        if vowels / total_letters < 0.20:
            return True
        if re.search(r"[asdfjkl]{6,}", lower) or re.search(r"[qweruiop]{6,}", lower):
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower):
            return True
    return False


def format_class_display_name(cls_name: str) -> str:
    """Format a machine class label (e.g. 'aculus_olearius') into a display title ('Aculus Olearius')."""
    cleaned = re.sub(r"[-_]+", " ", cls_name).strip()
    words = cleaned.split()
    if not words:
        return cls_name
    return " ".join(_format_title_word(w) for w in words)


def slugify(text: str) -> str:
    """Convert a human title into a safe URL slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "-", text)
    return slug or "modelo-ia"


def is_healthy_class(cls_name: str) -> bool:
    """Check whether a class represents a normal, healthy, or control baseline."""
    clean = re.sub(r"[^a-zA-Z0-9]+", " ", cls_name).lower().strip()
    tokens = set(clean.split())
    return bool(tokens & HEALTHY_KEYWORDS)


# ==============================================================================
# TITLE & SLUG INFERENCE
# ==============================================================================

def infer_dataset_title_and_slug(
    source_name: str | None,
    classes: list[str],
    image_counts: dict[str, int] | None = None,
    is_yolo: bool = False,
) -> tuple[str, str]:
    """Infer a natural, concise title and slug from the dataset classes or meaningful filename."""
    # Check if source_name is a meaningful, human filename
    if source_name and not _is_random_or_opaque_filename(source_name):
        title = clean_filename_title(source_name)
        return title, slugify(title)

    # Otherwise, infer intelligently from classes
    if not classes:
        return "Nuevo Modelo de Visión", "nuevo-modelo-vision"

    healthy_classes = [c for c in classes if is_healthy_class(c)]
    defect_classes = [c for c in classes if not is_healthy_class(c)]

    # Sort defects by count descending if available
    if image_counts:
        defect_classes = sorted(defect_classes, key=lambda c: image_counts.get(c, 0), reverse=True)

    fmt_defects = [format_class_display_name(c) for c in defect_classes]
    fmt_healthy = [format_class_display_name(c) for c in healthy_classes]

    if is_yolo:
        if len(classes) == 1:
            title = f"Detección de {format_class_display_name(classes[0])}"
        elif len(classes) == 2:
            title = f"Detección de {format_class_display_name(classes[0])} y {format_class_display_name(classes[1])}"
        else:
            title = f"Detección de {format_class_display_name(classes[0])}, {format_class_display_name(classes[1])} y Objetos"
        return title, slugify(title)

    # Classification scenario
    if fmt_defects:
        if len(fmt_defects) == 1:
            if fmt_healthy:
                title = f"Detección de {fmt_defects[0]}"
            else:
                title = f"Clasificador de {fmt_defects[0]}"
        elif len(fmt_defects) == 2:
            title = f"Detección de {fmt_defects[0]} y {fmt_defects[1]}"
        elif len(fmt_defects) == 3:
            title = f"Detección de {fmt_defects[0]}, {fmt_defects[1]} y {fmt_defects[2]}"
        else:
            title = f"Detección de {fmt_defects[0]}, {fmt_defects[1]} y Patologías"
    else:
        # All classes matched healthy / control or are generic
        all_fmt = [format_class_display_name(c) for c in classes]
        if len(all_fmt) == 2:
            title = f"Clasificación de {all_fmt[0]} vs {all_fmt[1]}"
        elif len(all_fmt) <= 3:
            title = f"Clasificación de {', '.join(all_fmt[:-1])} y {all_fmt[-1]}"
        else:
            title = f"Clasificación Multiclase ({all_fmt[0]}, {all_fmt[1]})"

    return title, slugify(title)


# ==============================================================================
# NARRATIVE VARIETY DESCRIPTION GENERATOR (10+ Templates)
# ==============================================================================

def generate_narrative_description(
    dataset_path: str,
    classes: list[str],
    total_images: int,
    image_counts: dict[str, int] | None = None,
    is_yolo: bool = False,
    total_boxes: int = 0,
) -> str:
    """Generate a rich, contextual technical description with narrative variety.

    Uses a deterministic seed based on dataset path and classes to ensure stable reproducibility
    for a given dataset while providing distinct styles across different models.
    """
    if not classes:
        return "Modelo de visión artificial configurado para inferencia predictiva."

    healthy_classes = [c for c in classes if is_healthy_class(c)]
    defect_classes = [c for c in classes if not is_healthy_class(c)]

    if image_counts:
        defect_classes = sorted(defect_classes, key=lambda c: image_counts.get(c, 0), reverse=True)

    fmt_defects = [format_class_display_name(c) for c in defect_classes]
    fmt_healthy = [format_class_display_name(c) for c in healthy_classes]
    all_fmt = [format_class_display_name(c) for c in classes]

    # Helper phrases
    if len(fmt_defects) == 1:
        defects_str = fmt_defects[0]
    elif len(fmt_defects) == 2:
        defects_str = f"{fmt_defects[0]} y {fmt_defects[1]}"
    elif len(fmt_defects) == 3:
        defects_str = f"{fmt_defects[0]}, {fmt_defects[1]} y {fmt_defects[2]}"
    elif len(fmt_defects) > 3:
        extra_count = len(fmt_defects) - 2
        defects_str = f"{fmt_defects[0]}, {fmt_defects[1]} y {extra_count} patologías adicionales"
    else:
        defects_str = "anomalías visuales"

    healthy_str = fmt_healthy[0] if fmt_healthy else "estado de control"

    if len(all_fmt) <= 2:
        classes_preview = " y ".join(all_fmt)
    elif len(all_fmt) == 3:
        classes_preview = f"{all_fmt[0]}, {all_fmt[1]} y {all_fmt[2]}"
    else:
        classes_preview = f"{all_fmt[0]}, {all_fmt[1]} y {len(all_fmt) - 2} clases más"

    # Seed generation: deterministic MD5 hash
    hash_key = f"{dataset_path}:{','.join(sorted(classes))}:{total_images}"
    seed = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)

    # 1. YOLO Object Detection Pool
    if is_yolo:
        yolo_templates = [
            f"Detector visual en tiempo real entrenado para la localización espacial precisa y delimitación de {classes_preview}, totalizando {total_boxes} anotaciones en {total_images} imágenes verificadas.",
            f"Modelo de detección de objetos basado en YOLO para la identificación y coordenadas de {classes_preview}. Permite inferencia simultánea de múltiples instancias por captura.",
            f"Pipeline de visión espacial optimizado para localizar y cuantificar ocurrencias de {classes_preview} sobre {total_images} capturas anotadas.",
            f"Red neuronal de visión artificial especializada en bounding boxes multiobjeto ({classes_preview}). Diseñada para inspección automática y conteo en tiempo real.",
        ]
        return yolo_templates[seed % len(yolo_templates)]

    # 2. Classification with Healthy & Defect classes (12 narrative archetypes)
    if fmt_defects and fmt_healthy:
        templates = [
            # 1. Triaje y Cribado Preventivo
            f"Red neuronal de visión artificial entrenada para el triaje temprano de {defects_str}, contrastada con muestras de control ({healthy_str}). Diseñada para automatizar la inspección visual en entornos de muestreo continuo.",
            # 2. Diagnóstico y Clasificación Especializada
            f"Modelo predictivo de alta precisión orientado a la detección de {defects_str}. Permite clasificar de forma supervisada frente a ejemplares {healthy_str} sobre un corpus verificado de {total_images} imágenes.",
            # 3. Control de Calidad e Inspección
            f"Módulo de inspección visual enfocado en la segregación y alerta rápida ante {defects_str}, evaluado frente a estados normales ({healthy_str}). Minimiza el margen de error subjetivo en inspecciones periódicas.",
            # 4. Monitorización y Detección de Síntomas
            f"Sistema clasificador concebido para la monitorización automática de {defects_str}. Proporciona scoring de confianza calibrado para distinguir síntomas específicos frente a controles ({healthy_str}) en {total_images} capturas analizadas.",
            # 5. Cribado Operativo Multiclase
            f"Algoritmo de visión por computador para la catalogación y triaje multiclase de {defects_str} y ejemplares {healthy_str}. Optimizado para despliegue distribuido y respuesta en milisegundos.",
            # 6. Detección Precoz y Vigilancia Técnica
            f"Herramienta de análisis visual automatizado para la identificación precoz de patrones compatibles con {defects_str}. Valida la ausencia de anomalías frente al estándar {healthy_str}.",
            # 7. Segregación Asistida y Soporte a Decisiones
            f"Pipeline clasificador multiclase desarrollado para asistir a inspectores y técnicos en el reconocimiento certero de {defects_str} frente al estándar {healthy_str}. Cuenta con {len(classes)} clases calibradas con {total_images} imágenes de entrenamiento.",
            # 8. Inferencia Morfológica y Discriminación de Patologías
            f"Modelo convolucional optimizado para discriminar variaciones morfológicas y cromáticas causadas por {defects_str}, con base de comparación en especímenes {healthy_str}.",
            # 9. Verificación Sistemática de Calidad
            f"Solución analítica de visión artificial para la verificación sistemática y detección de anomalías ({defects_str}), contrastada frente a muestras {healthy_str}. Diseñada para operar con baja latencia en flujos de trabajo federados.",
            # 10. Clasificación Comparativa Control/Anomalía
            f"Red discriminativa multivariable para la detección focalizada de {defects_str} frente al estado de referencia ({healthy_str}). Validada sobre un banco exhaustivo de {total_images} capturas procesadas.",
            # 11. Cribado Fenotípico y Sintomatológico
            f"Modelo de clasificación y cribado visual calibrado para la discriminación certera entre {healthy_str} y manifestaciones de {defects_str}, sustentado en {total_images} imágenes etiquetadas.",
            # 12. Vigilancia en Tiempo Real
            f"Arquitectura de visión computacional para la caracterización fenotípica y sintomatológica de {defects_str}, garantizando diferenciación robusta frente a muestras {healthy_str}.",
        ]
        return templates[seed % len(templates)]

    # 3. Multiclass Triage without Explicit Healthy Class
    multiclass_templates = [
        f"Modelo de visión artificial multiclase calibrado para categorizar y separar entre {classes_preview}. Optimizado para catalogación precisa sobre {total_images} muestras analizadas.",
        f"Red neuronal convolucional especializada en la discriminación multiclase de {classes_preview}. Diseñada para triaje automatizado y soporte a decisiones analíticas.",
        f"Sistema de clasificación supervisada entrenado con {total_images} capturas para la segregación sistemática de {classes_preview}.",
        f"Pipeline de análisis visual para la categorización y clasificación de {len(classes)} tipologías ({classes_preview}).",
        f"Modelo predictivo orientado a la diferenciación visual de {classes_preview}, optimizado para inferencia en tiempo real sobre {total_images} imágenes.",
        f"Clasificador multiespectral basado en red profunda para la catalogación y triaje continuo entre {classes_preview}.",
    ]
    return multiclass_templates[seed % len(multiclass_templates)]


# ==============================================================================
# DOMAIN INFERENCE
# ==============================================================================

def infer_domain(
    classes: list[str],
    source_name: str | None = None,
) -> str:
    """Infer the primary domain or sector based on class names and source keywords."""
    combined = " ".join(classes + ([source_name] if source_name else [])).lower()
    clean_tokens = set(re.sub(r"[^a-zA-Z0-9]+", " ", combined).split())

    # 1. Agriculture / Agroalimentario
    if clean_tokens & AGRO_KEYWORDS:
        return "Agroalimentario & Agricultura"

    # 2. Industry & Manufacturing
    if clean_tokens & INDUSTRY_KEYWORDS:
        return "Industria & Manufactura"

    # 3. Healthcare & Biomedicine
    if clean_tokens & HEALTH_KEYWORDS:
        return "Biomedicina & Sanidad"

    # 4. Environment & Forestry
    if clean_tokens & ENVIRONMENT_KEYWORDS:
        return "Medio Ambiente & Recursos"

    # Default versatile
    return "Visión Artificial"


# ==============================================================================
# TAGS INFERENCE
# ==============================================================================

def generate_tags(
    classes: list[str],
    domain: str,
) -> str:
    """Generate search tags based on classes and domain."""
    tags: list[str] = []
    for cls in classes:
        clean = re.sub(r"[-_]+", " ", cls).strip().lower()
        if clean and clean not in tags:
            tags.append(clean)

    # Add domain hint
    if "agro" in domain.lower():
        tags.append("sanidad vegetal")
    elif "industria" in domain.lower():
        tags.append("control de calidad")
    elif "biomedicina" in domain.lower():
        tags.append("diagnóstico clínico")

    tags.append("visión artificial")
    return ", ".join(tags[:8])


# ==============================================================================
# DATASET THUMBNAIL GENERATOR (WebP Base64)
# ==============================================================================

def extract_dataset_thumbnail_b64(
    target_dir: Path,
    classes: list[str],
    is_yolo: bool = False,
) -> str | None:
    """Extract a representative sample image from the dataset and return an inline WebP Base64 URI."""
    try:
        from PIL import Image
    except ImportError:
        return None

    if not target_dir.is_dir():
        return None

    candidate_paths: list[Path] = []
    valid_exts = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

    if is_yolo:
        # Check YOLO image folders
        for sub in [target_dir / "images" / "train", target_dir / "images", target_dir]:
            if sub.is_dir():
                for p in sub.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        candidate_paths.append(p)
                        if len(candidate_paths) >= 5:
                            break
            if candidate_paths:
                break
    else:
        # Prefer defect classes first for more interesting visual thumbnails
        defect_classes = [c for c in classes if not is_healthy_class(c)]
        ordered_classes = defect_classes + [c for c in classes if is_healthy_class(c)]
        if not ordered_classes:
            ordered_classes = classes

        for cls_name in ordered_classes:
            cls_dir = target_dir / cls_name
            if cls_dir.is_dir():
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        candidate_paths.append(p)
                        if len(candidate_paths) >= 3:
                            break
            if candidate_paths:
                break

        # Fallback: search recursively if top-level scan found nothing
        if not candidate_paths:
            try:
                for p in target_dir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        candidate_paths.append(p)
                        if len(candidate_paths) >= 1:
                            break
            except OSError:
                pass

    if not candidate_paths:
        return None

    sample_p = candidate_paths[0]
    try:
        with Image.open(sample_p) as img:
            thumb = img.copy()
            if thumb.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", thumb.size, (255, 255, 255))
                if thumb.mode == "P":
                    thumb = thumb.convert("RGBA")
                bg.paste(thumb, mask=thumb.split()[-1] if thumb.mode == "RGBA" else None)
                thumb = bg
            elif thumb.mode != "RGB":
                thumb = thumb.convert("RGB")

            resample_filter = getattr(getattr(Image, "Resampling", None), "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
            thumb.thumbnail((400, 300), resample_filter)
            buf = io.BytesIO()
            thumb.save(buf, format="WEBP", quality=75, method=4)
            b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/webp;base64,{b64_str}"
    except Exception:
        return None


# ==============================================================================
# HIGH-LEVEL AGGREGATOR
# ==============================================================================

def suggest_dataset_metadata(
    dataset_path: str,
    classes: list[str],
    total_images: int,
    image_counts: dict[str, int] | None = None,
    source_name: str | None = None,
    is_yolo: bool = False,
    total_boxes: int = 0,
) -> dict[str, Any]:
    """Inspect dataset attributes and return inferred title, slug, narrative description, domain, tags, and thumbnail."""
    target_p = Path(dataset_path)
    title, slug = infer_dataset_title_and_slug(
        source_name=source_name or target_p.name,
        classes=classes,
        image_counts=image_counts,
        is_yolo=is_yolo,
    )
    description = generate_narrative_description(
        dataset_path=dataset_path,
        classes=classes,
        total_images=total_images,
        image_counts=image_counts,
        is_yolo=is_yolo,
        total_boxes=total_boxes,
    )
    domain = infer_domain(classes=classes, source_name=source_name or target_p.name)
    tags = generate_tags(classes=classes, domain=domain)
    thumbnail_b64 = extract_dataset_thumbnail_b64(
        target_dir=target_p,
        classes=classes,
        is_yolo=is_yolo,
    )

    return {
        "suggested_title": title,
        "suggested_slug": slug,
        "suggested_description": description,
        "suggested_domain": domain,
        "suggested_tags": tags,
        "thumbnail_b64": thumbnail_b64,
    }
