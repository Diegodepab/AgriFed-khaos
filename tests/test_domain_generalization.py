"""Unit tests for Universal Domain Generalization & In-The-Wild Robustness (Phases 1-4)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from agrifed_khaos.data import RandomBackgroundSwapTransform, build_augmentation_transform
from agrifed_khaos.dataset_service import profile_dataset_health_sync
from agrifed_khaos.inference_service import (
    InferenceEngine,
    build_aspect_letterbox,
    extract_smart_roi_crop,
    fuse_multiview_logits,
)
from agrifed_khaos.models import build_model, save_checkpoint_bundle


def _make_studio_specimen_image(
    bg_rgb: tuple[int, int, int] = (5, 5, 5),
    fg_rgb: tuple[int, int, int] = (210, 165, 45),
    size: tuple[int, int] = (320, 240),
) -> Image.Image:
    """Create a synthetic image with a uniform background and a central specimen."""
    w, h = size
    arr = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    arr[y0:y1, x0:x1] = fg_rgb
    return Image.fromarray(arr)


def test_phase1_smart_roi_crop_and_letterbox() -> None:
    """Verify Smart ROI Crop focuses on foreground specimen and Letterbox preserves aspect ratio."""
    img = _make_studio_specimen_image(bg_rgb=(250, 250, 250), fg_rgb=(180, 90, 30), size=(400, 200))

    roi_crop = extract_smart_roi_crop(img, padding_pct=0.12)
    assert isinstance(roi_crop, Image.Image)
    # Cropped region should be smaller than the original 400x200 frame
    assert roi_crop.size[0] < 400 or roi_crop.size[1] < 200
    # Center of ROI crop should contain the foreground specimen color
    cx, cy = roi_crop.size[0] // 2, roi_crop.size[1] // 2
    center_px = roi_crop.getpixel((cx, cy))
    assert center_px == (180, 90, 30)

    letterboxed = build_aspect_letterbox(img, target_size=(224, 224))
    assert letterboxed.size == (224, 224)


def test_phase1_fuse_multiview_logits_inverse_entropy() -> None:
    """Verify inverse-entropy multi-view fusion prioritizes confident views over uncertain views."""
    # View 0 (Standard): high entropy / uncertain across 3 classes
    # View 1 (ROI Crop): low entropy / high certainty for class 2
    # View 2 (Letterbox): moderate certainty for class 2
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.0],
            [-2.0, -1.5, 4.5],
            [-1.0, -0.5, 2.5],
        ],
        dtype=torch.float32,
    )
    fused = fuse_multiview_logits(logits, temperature=1.0)
    assert fused.shape == (3,)
    assert int(torch.argmax(fused).item()) == 2


def test_phase1_inference_engine_multiview_infer_single(tmp_path: Path) -> None:
    """Verify InferenceEngine.infer_single runs multi-view batch inference seamlessly."""
    ckpt_path = tmp_path / "test_model.pt"
    net = build_model("resnet18", num_classes=3, pretrained=False)
    save_checkpoint_bundle(
        ckpt_path,
        net.state_dict(),
        metadata={
            "architecture": "resnet18",
            "num_classes": 3,
            "classes": ["almendra", "nuez", "pistacho"],
            "img_size": 224,
        },
    )
    engine = InferenceEngine(model_path=str(ckpt_path), metadata_path=str(tmp_path / "none.json"))

    img = _make_studio_specimen_image(size=(360, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = engine.infer_single(buf.getvalue(), "specimen.jpg")

    assert res["predicted_class"] in ["almendra", "nuez", "pistacho"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 3


def test_phase2_dataset_health_background_bias_detector(tmp_path: Path) -> None:
    """Verify dataset health profiler detects studio background bias vs diverse scenes."""
    # 1. Dark studio dataset (e.g., frutossecos style)
    dark_dir = tmp_path / "dark_studio_ds"
    for cls_name in ("clase_a", "clase_b"):
        cdir = dark_dir / cls_name
        cdir.mkdir(parents=True)
        for i in range(6):
            im = _make_studio_specimen_image(bg_rgb=(4, 4, 4), fg_rgb=(190, 140, 60))
            im.save(cdir / f"sample_{i}.jpg")

    hp_dark = profile_dataset_health_sync(dark_dir)
    assert hp_dark["dominant_bg_type"] == "dark_studio"
    assert hp_dark["background_uniformity_pct"] >= 65.0
    assert hp_dark["domain_shift_risk"] == "high"
    assert hp_dark["recommended_augmentation"] == "in_the_wild"

    # 2. Natural diverse dataset
    nat_dir = tmp_path / "natural_ds"
    rng = np.random.default_rng(123)
    for cls_name in ("clase_a", "clase_b"):
        cdir = nat_dir / cls_name
        cdir.mkdir(parents=True)
        for i in range(6):
            arr = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cdir / f"nat_{i}.png")

    hp_nat = profile_dataset_health_sync(nat_dir)
    assert hp_nat["dominant_bg_type"] == "natural_diverse"
    assert hp_nat["domain_shift_risk"] == "low"


def test_phase3_random_background_swap_and_in_the_wild_pipeline() -> None:
    """Verify RandomBackgroundSwapTransform replaces uniform borders while preserving object."""
    studio_img = _make_studio_specimen_image(
        bg_rgb=(0, 0, 0), fg_rgb=(220, 180, 50), size=(160, 160)
    )
    swap = RandomBackgroundSwapTransform(p=1.0, seed=42)
    out_img = swap(studio_img)
    out_arr = np.asarray(out_img)

    # Border was pure black (0, 0, 0); after swap it should be non-zero synthetic background
    assert float(out_arr[:8, :, :].mean()) > 15.0
    # Foreground center should remain untouched (220, 180, 50)
    center_rgb = out_arr[80, 80]
    assert np.allclose(center_rgb, [220, 180, 50], atol=5)

    # Full 'in_the_wild' transform produces normalized (3, 224, 224) tensor
    tf = build_augmentation_transform("in_the_wild", img_size=224)
    tensor_out = tf(studio_img)
    assert isinstance(tensor_out, torch.Tensor)
    assert tuple(tensor_out.shape) == (3, 224, 224)


@pytest.mark.parametrize("arch", ["convnext_tiny", "swin_t", "eva02_tiny_patch14_224"])
def test_phase4_domain_invariant_backbones_instantiation(arch: str) -> None:
    """Verify domain-invariant backbones instantiate with custom num_classes."""
    model = build_model(architecture=arch, num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (2, 7)

