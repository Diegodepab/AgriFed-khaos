#!/usr/bin/env python3
"""Ultralytics YOLO Object Detection Training Pipeline for AgriFed-KHAOS.

Supports both COCO JSON and Ultralytics YOLO dataset layouts.
Automatically converts COCO datasets into YOLO format if required, trains
YOLOv8/YOLO11 models, captures mAP50, mAP50-95, precision and recall metrics,
and saves production checkpoints and KHAOS metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics YOLO Object Detection model on COCO/YOLO datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory (COCO JSON or YOLO format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/detection_model_base.pt",
        help="Path where output checkpoint (.pt) will be written",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/detection_model.meta.json",
        help="Path where model metadata (.meta.json) will be written",
    )
    parser.add_argument(
        "--arch",
        "--model",
        dest="model",
        type=str,
        default="yolov8n",
        help="YOLO model architecture / weights (e.g. yolov8n, yolov8s, yolo11n)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16; <=0 = AutoBatch sized to the VRAM cap)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate lr0 (default: 0.01)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('cpu', '0', 'cuda', 'auto')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers (default: 2)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        choices=["none", "ram", "disk"],
        default="none",
        help="Cache decoded images to speed up epochs (default: none)",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Keep the Ultralytics run folder (plots, last.pt) instead of cleaning it up",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max images per split for fast testing (0 = unlimited)",
    )
    return parser.parse_args()


def compute_greedy_matched_mean_iou(
    model: Any,
    data_yaml_path: Path,
    split: str = "val",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    max_eval_images: int = 150,
) -> tuple[float, int]:
    """Compute 1-to-1 greedy bipartite matching Mean IoU across validation/test split.

    Adheres strictly to the criteria:
    - Same-class matching only.
    - 1-to-1 greedy bipartite matching ordered by descending prediction confidence.
    - Confidence threshold >= conf_thresh (default 0.25).
    - IoU threshold >= iou_thresh (default 0.50).
    - Average calculated exclusively over matched pairs.
    - Returns (mean_iou, matched_detections_count). If matched count is 0, returns (0.0, 0).
    """
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    except Exception:
        yaml_data = {}

    base_dir = Path(yaml_data.get("path", data_yaml_path.parent)).resolve()
    split_entry = yaml_data.get(split) or f"images/{split}"
    if isinstance(split_entry, list):
        split_entry = split_entry[0] if split_entry else f"images/{split}"

    img_dir_cand = (base_dir / split_entry).resolve()
    if not img_dir_cand.is_dir():
        for fb in [
            base_dir / split / "images",
            base_dir / "images" / split,
            base_dir / split,
            base_dir / "images",
        ]:
            if fb.is_dir():
                img_dir_cand = fb
                break

    if not img_dir_cand.is_dir():
        return 0.0, 0

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_files = sorted([p for p in img_dir_cand.iterdir() if p.suffix.lower() in image_exts])
    if not img_files:
        return 0.0, 0

    if len(img_files) > max_eval_images:
        img_files = img_files[:max_eval_images]

    lbl_dir_cand = None
    if "images" in img_dir_cand.parts:
        parts = list(img_dir_cand.parts)
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        cand = Path(*parts)
        if cand.is_dir():
            lbl_dir_cand = cand
    if lbl_dir_cand is None or not lbl_dir_cand.is_dir():
        for fb_l in [
            base_dir / "labels" / split,
            base_dir / split / "labels",
            base_dir / "labels",
        ]:
            if fb_l.is_dir():
                lbl_dir_cand = fb_l
                break

    if lbl_dir_cand is None or not lbl_dir_cand.is_dir():
        return 0.0, 0

    matched_ious: list[float] = []

    for img_p in img_files:
        lbl_p = lbl_dir_cand / f"{img_p.stem}.txt"
        if not lbl_p.is_file():
            continue

        gt_boxes: list[list[float]] = []
        gt_classes: list[int] = []
        try:
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_id = int(float(parts[0]))
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gx1 = max(0.0, xc - bw / 2.0)
                    gy1 = max(0.0, yc - bh / 2.0)
                    gx2 = min(1.0, xc + bw / 2.0)
                    gy2 = min(1.0, yc + bh / 2.0)
                    gt_boxes.append([gx1, gy1, gx2, gy2])
                    gt_classes.append(c_id)
        except Exception:
            continue

        if not gt_boxes:
            continue

        try:
            preds = model(str(img_p), conf=conf_thresh, verbose=False)
            if not preds:
                continue
            pred_boxes_res = preds[0].boxes
            if pred_boxes_res is None or len(pred_boxes_res) == 0:
                continue

            p_xyxyn = pred_boxes_res.xyxyn.cpu().numpy()
            p_cls = pred_boxes_res.cls.cpu().numpy().astype(int)
            p_conf = pred_boxes_res.conf.cpu().numpy()
        except Exception:
            continue

        order = np.argsort(-p_conf)
        matched_gt_indices: set[int] = set()

        for idx in order:
            if float(p_conf[idx]) < conf_thresh:
                continue
            pred_cls = int(p_cls[idx])
            pred_b = p_xyxyn[idx]

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, (gt_b, gt_c) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_c != pred_cls or g_idx in matched_gt_indices:
                    continue

                ix1 = max(pred_b[0], gt_b[0])
                iy1 = max(pred_b[1], gt_b[1])
                ix2 = min(pred_b[2], gt_b[2])
                iy2 = min(pred_b[3], gt_b[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih

                area_p = max(0.0, pred_b[2] - pred_b[0]) * max(0.0, pred_b[3] - pred_b[1])
                area_g = max(0.0, gt_b[2] - gt_b[0]) * max(0.0, gt_b[3] - gt_b[1])
                union = area_p + area_g - inter
                iou = inter / union if union > 1e-7 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_ious.append(float(best_iou))

    matched_count = len(matched_ious)
    mean_iou = float(np.mean(matched_ious)) if matched_count > 0 else 0.0
    return round(mean_iou, 4), matched_count


def main() -> None:
    args = parse_args()

    data_dir_p = Path(args.data_dir).resolve()
    if not data_dir_p.exists():
        logger.error("Dataset directory does not exist: %s", data_dir_p)
        sys.exit(1)

    output_p = Path(args.output).resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    metadata_p = Path(args.metadata).resolve()
    metadata_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve / Prepare YOLO dataset data.yaml
    from agrifed_khaos.coco import prepare_detection_dataset

    logger.info("Resolving and preparing detection dataset from: %s", data_dir_p)
    try:
        data_yaml_path, class_names = prepare_detection_dataset(data_dir_p)
    except Exception as exc:
        logger.error("Failed preparing detection dataset: %s", exc)
        sys.exit(1)

    logger.info(
        "Detection dataset ready at: %s with %d class(es): %s",
        data_yaml_path,
        len(class_names),
        class_names,
    )

    # 2. Select Device
    if args.device == "auto":
        device_str = "0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    logger.info("Training on device: %s", device_str)

    # Shared-GPU etiquette: cap VRAM share and size AutoBatch to that share.
    batch_arg: int | float = max(1, args.batch_size)
    if device_str != "cpu" and torch.cuda.is_available():
        from agrifed_khaos.hardware import apply_gpu_memory_cap

        cap = apply_gpu_memory_cap()
        if cap is not None:
            logger.info("GPU memory cap active: %.0f%% of device VRAM", cap * 100)
        if args.batch_size <= 0:
            # Ultralytics AutoBatch: a float in (0, 1) means "fraction of VRAM to target".
            batch_arg = round(min(0.9, max(0.3, (cap or 0.7) * 0.85)), 2)
            logger.info("AutoBatch enabled targeting %.0f%% of VRAM", batch_arg * 100)
    elif args.batch_size <= 0:
        batch_arg = 8

    # 3. Model Architecture
    raw_model = args.model.strip()
    if not raw_model.endswith(".pt") and not raw_model.endswith(".yaml"):
        model_name = f"{raw_model}.pt"
    else:
        model_name = raw_model

    # 4. Import Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.error(
            "Ultralytics is not installed. Please install with: pip install ultralytics (%s)",
            exc,
        )
        sys.exit(1)

    logger.info("Initializing YOLO model: %s", model_name)
    model = YOLO(model_name)

    # 5. Training
    run_name = output_p.stem
    project_dir = repo_root / "models" / "runs_detection"
    project_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting YOLO training: epochs=%d, batch=%d, imgsz=%d, lr0=%f, patience=%d",
        args.epochs,
        args.batch_size,
        args.img_size,
        args.lr,
        args.patience,
    )

    results = model.train(
        data=str(data_yaml_path),
        epochs=max(1, args.epochs),
        batch=batch_arg,
        imgsz=args.img_size,
        device=device_str,
        project=str(project_dir),
        name=run_name,
        patience=args.patience,
        lr0=args.lr,
        workers=max(0, args.workers),
        exist_ok=True,
        cos_lr=True,
        close_mosaic=min(10, max(1, args.epochs // 5)),
        cache=False if args.cache == "none" else args.cache,
        plots=True,
        save=True,
        verbose=True,
    )

    # 6. Locate trained weights
    weights_dir = project_dir / run_name / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    src_weights = None
    if best_weights.is_file():
        src_weights = best_weights
    elif last_weights.is_file():
        src_weights = last_weights

    if src_weights is not None:
        logger.info("Copying best weights from %s to %s", src_weights, output_p)
        shutil.copy2(src_weights, output_p)
    else:
        logger.warning(
            "Could not find best.pt or last.pt in %s. Saving model state directly.",
            weights_dir,
        )
        model.save(str(output_p))

    # 7. Resolve Evaluation Split and Run Detailed Evaluation
    eval_split = "val"
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    except Exception:
        yaml_data = {}

    if "test" in yaml_data and yaml_data["test"]:
        eval_split = "test"
    elif "val" in yaml_data and yaml_data["val"]:
        eval_split = "val"
    else:
        eval_split = "train_fallback"

    logger.info("Evaluating detection model on split: %s", eval_split)
    val_results = None
    try:
        val_results = model.val(
            data=str(data_yaml_path),
            split=eval_split if eval_split != "train_fallback" else "train",
            project=str(project_dir),
            name=f"{run_name}_val",
            plots=True,
            save_json=False,
            device=device_str,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("Dedicated val step encountered error: %s; falling back to train results", exc)
        val_results = results

    # 8. Extract Evaluation Metrics
    map50 = 0.0
    map50_95 = 0.0
    map75 = 0.0
    prec = 0.0
    rec = 0.0

    eval_obj = val_results if val_results is not None else results
    if eval_obj is not None:
        try:
            r_dict = getattr(eval_obj, "results_dict", {})
            box_m = getattr(eval_obj, "box", None)
            map50 = float(
                r_dict.get(
                    "metrics/mAP50(B)",
                    getattr(box_m, "map50", 0.0) if box_m else 0.0,
                )
            )
            map50_95 = float(
                r_dict.get(
                    "metrics/mAP50-95(B)",
                    getattr(box_m, "map", 0.0) if box_m else 0.0,
                )
            )
            map75 = float(
                r_dict.get(
                    "metrics/mAP75(B)",
                    getattr(box_m, "map75", 0.0) if box_m else 0.0,
                )
            )
            prec = float(
                r_dict.get(
                    "metrics/precision(B)",
                    getattr(box_m, "mp", getattr(box_m, "p", 0.0)) if box_m else 0.0,
                )
            )
            rec = float(
                r_dict.get(
                    "metrics/recall(B)",
                    getattr(box_m, "mr", getattr(box_m, "r", 0.0)) if box_m else 0.0,
                )
            )
        except Exception as exc:
            logger.warning("Could not extract detailed metrics from results: %s", exc)

    # 9. Extract per-class summary from val_results.summary()
    per_class_summary: list[dict[str, Any]] = []
    if eval_obj is not None and hasattr(eval_obj, "summary"):
        try:
            raw_summary = eval_obj.summary(normalize=True, decimals=4)
            for idx, row in enumerate(raw_summary):
                c_name = str(row.get("Class", f"class_{idx}"))
                per_class_summary.append({
                    "class_name": c_name,
                    "class_id": idx,
                    "images": int(row.get("Images", 0)),
                    "instances": int(row.get("Instances", 0)),
                    "precision": round(float(row.get("Box-P", 0.0)), 4),
                    "recall": round(float(row.get("Box-R", 0.0)), 4),
                    "f1_score": round(float(row.get("Box-F1", 0.0)), 4),
                    "mAP50": round(float(row.get("mAP50", 0.0)), 4),
                    "mAP75": round(float(row.get("mAP75", 0.0)), 4),
                    "mAP50_95": round(float(row.get("mAP50-95", 0.0)), 4),
                })
        except Exception as exc:
            logger.warning("Could not extract per-class summary: %s", exc)

    # 10. Compute 1-to-1 Greedy Bipartite Matching Mean IoU
    mean_iou, matched_count = compute_greedy_matched_mean_iou(
        model=model,
        data_yaml_path=data_yaml_path,
        split=eval_split if eval_split != "train_fallback" else "train",
        conf_thresh=0.25,
        iou_thresh=0.50,
    )
    logger.info(
        "Greedy 1-to-1 matched Mean IoU (conf>=0.25, iou>=0.50): %.4f across %d matched detections",
        mean_iou,
        matched_count,
    )

    # 11. Copy and Compress Evaluation Preview Mosaic (val_batch0_pred.jpg)
    slug = metadata_p.name.replace(".meta.json", "")
    eval_preview_dst = output_p.parent / f"{slug}_eval_preview.jpg"
    preview_candidates = [
        project_dir / f"{run_name}_val" / "val_batch0_pred.jpg",
        project_dir / run_name / "val_batch0_pred.jpg",
        project_dir / f"{run_name}_val" / "val_batch0_labels.jpg",
        project_dir / run_name / "val_batch0_labels.jpg",
    ]
    eval_preview_created = False
    for cand_p in preview_candidates:
        if cand_p.is_file():
            try:
                resample_filter = getattr(getattr(Image, "Resampling", None), "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
                with Image.open(cand_p) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    max_dim = 1280
                    if max(rgb_img.size) > max_dim:
                        scale = max_dim / float(max(rgb_img.size))
                        new_size = (int(rgb_img.width * scale), int(rgb_img.height * scale))
                        rgb_img = rgb_img.resize(new_size, resample_filter)
                    rgb_img.save(eval_preview_dst, format="JPEG", quality=85, optimize=True)
                    eval_preview_created = True
                    logger.info("Saved compressed eval preview mosaic to %s", eval_preview_dst)

                    # Also ensure catalog thumbnail exists if not present
                    models_thumb_dir = output_p.parent / "thumbnails"
                    models_thumb_dir.mkdir(parents=True, exist_ok=True)
                    thumb_target = models_thumb_dir / f"{slug}.jpg"
                    if not thumb_target.exists():
                        t_img = rgb_img.copy()
                        t_img.thumbnail((480, 360), resample_filter)
                        t_img.save(thumb_target, format="JPEG", quality=80, optimize=True)
                        logger.info("Saved catalog thumbnail from eval preview to %s", thumb_target)
                    break
            except Exception as exc:
                logger.warning("Failed processing eval preview mosaic from %s: %s", cand_p, exc)

    # 12. Write models/{slug}_evaluation_report.json
    eval_report_payload = {
        "slug": slug,
        "model_name": model_name,
        "task": "detection",
        "eval_split": eval_split,
        "metrics": {
            "mAP50": round(map50 * 100.0, 2),
            "mAP50_95": round(map50_95 * 100.0, 2),
            "mAP75": round(map75 * 100.0, 2),
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "mean_iou": round(mean_iou, 4),
            "matched_detections": matched_count,
        },
        "per_class": per_class_summary,
        "eval_preview_available": eval_preview_created,
        "eval_preview_filename": eval_preview_dst.name if eval_preview_created else None,
    }
    eval_report_path = output_p.parent / f"{slug}_evaluation_report.json"
    eval_report_path.write_text(json.dumps(eval_report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Evaluation report saved to %s", eval_report_path)

    # 13. Write Metadata JSON
    metrics_payload = {
        "mAP50": round(map50 * 100.0, 2),
        "mAP50_95": round(map50_95 * 100.0, 2),
        "mAP75": round(map75 * 100.0, 2),
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "mean_iou": round(mean_iou, 4),
        "matched_detections": matched_count,
        "accuracy": round(map50 * 100.0, 2),  # for UI table compatibility
    }

    metadata = {
        "task": "detection",
        "architecture": model_name.replace(".pt", ""),
        "num_classes": len(class_names),
        "classes": class_names,
        "img_size": args.img_size,
        "weights_file": str(output_p),
        "eval_split": eval_split,
        "metrics": metrics_payload,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": batch_arg,
            "learning_rate": args.lr,
            "img_size": args.img_size,
            "model": model_name,
            "patience": args.patience,
        },
        "dataset_source": str(data_dir_p),
    }
    if eval_preview_created:
        metadata["eval_preview"] = eval_preview_dst.name

    metadata_p.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata saved to %s", metadata_p)

    # 14. Cleanup temporary run directories unless keep_runs requested
    if not args.keep_runs:
        shutil.rmtree(project_dir / run_name, ignore_errors=True)
        shutil.rmtree(project_dir / f"{run_name}_val", ignore_errors=True)

    logger.info("Object detection training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
