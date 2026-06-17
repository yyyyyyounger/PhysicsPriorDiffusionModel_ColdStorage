"""Shared utilities for Cold-Fog SAM inference and comparison."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam_vit_h_4b8939.pth"
DEFAULT_INFER_OUTPUT = SCRIPT_DIR / "results/infer"
DEFAULT_COMPARE_OUTPUT = SCRIPT_DIR / "results/compare"
RECALL_THRESHOLDS = (0.5,)
SPLIT_SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def read_rgb(path: Path, resize_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize_hw is not None:
        height, width = resize_hw
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def extract_masks(
    annotations: list[dict[str, Any]],
    min_area: int,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for ann in annotations:
        if ann["area"] < min_area:
            continue
        masks.append(np.asarray(ann["segmentation"], dtype=bool))
    return masks


def match_against_reference(
    reference_masks: list[np.ndarray],
    query_masks: list[np.ndarray],
    recall_thresholds: tuple[float, ...],
) -> dict[str, float | int | None]:
    if not reference_masks:
        return {
            "reference_num_masks": 0,
            "matched_iou_vs_reference": None,
            **{
                f"mask_recall_at_{str(th).replace('.', '_')}": None
                for th in recall_thresholds
            },
        }

    best_ious: list[float] = []
    for ref_mask in reference_masks:
        if not query_masks:
            best_ious.append(0.0)
            continue
        best_ious.append(max(mask_iou(ref_mask, query_mask) for query_mask in query_masks))

    recalls = {
        f"mask_recall_at_{str(th).replace('.', '_')}": float(
            np.mean(np.array(best_ious) >= th)
        )
        for th in recall_thresholds
    }
    return {
        "reference_num_masks": len(reference_masks),
        "matched_iou_vs_reference": float(np.mean(best_ious)),
        **recalls,
    }


def query_metrics(
    annotations: list[dict[str, Any]],
    reference_masks: list[np.ndarray],
    min_area: int,
    recall_thresholds: tuple[float, ...],
) -> dict[str, float | int | None]:
    query_mask_list = extract_masks(annotations, min_area=min_area)
    stabilities = [
        ann["stability_score"]
        for ann in annotations
        if ann["area"] >= min_area
    ]

    match_stats = match_against_reference(
        reference_masks, query_mask_list, recall_thresholds
    )
    return {
        "num_valid_masks": len(query_mask_list),
        "mean_stability": float(np.mean(stabilities)) if stabilities else None,
        **match_stats,
    }


def infer_stats(annotations: list[dict[str, Any]], min_area: int) -> dict[str, float | int]:
    valid = [ann for ann in annotations if ann["area"] >= min_area]
    stabilities = [ann["stability_score"] for ann in valid]
    return {
        "num_valid_masks": len(valid),
        "mean_stability": float(np.mean(stabilities)) if stabilities else None,
        "total_masks": len(annotations),
    }


def build_mask_generator(
    checkpoint: Path,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    crop_n_layers: int,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )


def resolve_device(device: str, gpu_id: int | None) -> str:
    if device == "cpu":
        if gpu_id is not None:
            raise ValueError("--gpu-id/--gpu-ids cannot be used together with --device cpu")
        return "cpu"

    if not torch.cuda.is_available():
        if gpu_id is not None:
            print("Warning: CUDA unavailable; ignoring GPU selection and using cpu.")
        return "cpu"

    if gpu_id is None:
        return "cuda"

    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise ValueError(
            f"Invalid GPU id {gpu_id}; available GPU IDs: 0..{device_count - 1}"
        )
    torch.cuda.set_device(gpu_id)
    return f"cuda:{gpu_id}"


def parse_gpu_ids(gpu_ids: str | None, gpu_id: int | None) -> list[int] | None:
    if gpu_ids is not None and gpu_id is not None:
        raise ValueError("Use either --gpu-id or --gpu-ids, not both.")

    if gpu_ids is not None:
        parsed = [int(part.strip()) for part in gpu_ids.split(",") if part.strip()]
        if not parsed:
            raise ValueError("--gpu-ids must contain at least one GPU index.")
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"Duplicate GPU ids in --gpu-ids: {gpu_ids}")
        return parsed

    if gpu_id is not None:
        return [gpu_id]
    return None


def split_items(items: list[Any], num_shards: int, seed: int = SPLIT_SEED) -> list[list[Any]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    shuffled = list(items)

    random.Random(seed).shuffle(shuffled)

    shards: list[list[Any]] = [[] for _ in range(num_shards)]
    for index, item in enumerate(shuffled):
        shards[index % num_shards].append(item)
    return shards


def list_image_paths(input_dir: Path, manifest_path: Path | None = None) -> list[Path]:
    if manifest_path is not None:
        with manifest_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        sample_ids = payload["sample_ids"]
        paths: list[Path] = []
        for sample_id in sample_ids:
            matched = None
            for ext in IMAGE_EXTENSIONS:
                candidate = input_dir / f"{sample_id}{ext}"
                if candidate.is_file():
                    matched = candidate
                    break
            if matched is None:
                raise FileNotFoundError(
                    f"No image found for sample_id {sample_id!r} under {input_dir}"
                )
            paths.append(matched)
        return paths

    paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found under {input_dir}")
    return paths


def image_stem(path: Path) -> str:
    return path.stem


def save_annotations_npz(path: Path, annotations: list[dict[str, Any]]) -> None:
    if not annotations:
        np.savez_compressed(
            path,
            segmentations=np.zeros((0, 1, 1), dtype=bool),
            areas=np.zeros((0,), dtype=np.int32),
            stability_scores=np.zeros((0,), dtype=np.float32),
            predicted_ious=np.zeros((0,), dtype=np.float32),
        )
        return

    segmentations = np.stack(
        [np.asarray(ann["segmentation"], dtype=bool) for ann in annotations],
        axis=0,
    )
    areas = np.asarray([ann["area"] for ann in annotations], dtype=np.int32)
    stability_scores = np.asarray(
        [ann["stability_score"] for ann in annotations], dtype=np.float32
    )
    predicted_ious = np.asarray(
        [ann.get("predicted_iou", 0.0) for ann in annotations], dtype=np.float32
    )
    np.savez_compressed(
        path,
        segmentations=segmentations,
        areas=areas,
        stability_scores=stability_scores,
        predicted_ious=predicted_ious,
    )


def load_annotations_npz(path: Path) -> list[dict[str, Any]]:
    data = np.load(path)
    segmentations = data["segmentations"]
    areas = data["areas"]
    stability_scores = data["stability_scores"]
    predicted_ious = data["predicted_ious"]

    annotations: list[dict[str, Any]] = []
    for index in range(len(areas)):
        annotations.append({
            "segmentation": segmentations[index],
            "area": int(areas[index]),
            "stability_score": float(stability_scores[index]),
            "predicted_iou": float(predicted_ious[index]),
        })
    return annotations


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_overlay_png(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    output_path: Path,
    seed: int = 42,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not annotations:
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return

    sorted_anns = sorted(annotations, key=lambda ann: ann["area"], reverse=True)
    overlay = image.copy().astype(np.float32)
    rng = np.random.default_rng(seed)
    for ann in sorted_anns:
        color = rng.random(3) * 255.0
        mask = ann["segmentation"]
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"sample_count": len(rows)}
    for key in metric_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return summary


def save_compare_plot(
    summaries: dict[str, dict[str, float | int | None]],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()

    metrics = [
        ("num_valid_masks_mean", "Valid masks"),
        ("mean_stability_mean", "Mean stability"),
        ("matched_iou_vs_reference_mean", "Matched IoU vs reference"),
        ("mask_recall_at_0_5_mean", "Mask Recall@0.5"),
    ]

    labels = list(summaries.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    x = np.arange(len(labels))
    width = 0.55

    for ax, (key, subplot_title) in zip(axes.ravel(), metrics):
        means = [summaries[label].get(key) for label in labels]
        std_key = key.replace("_mean", "_std")
        stds = [summaries[label].get(std_key, 0.0) for label in labels]
        valid = [mean is not None for mean in means]
        plot_x = x[valid]
        plot_means = [mean for mean, ok in zip(means, valid) if ok]
        plot_stds = [std for std, ok in zip(stds, valid) if ok]
        plot_labels = [label for label, ok in zip(labels, valid) if ok]

        ax.bar(plot_x, plot_means, width, yerr=plot_stds, capsize=4, color="#4C72B0")
        ax.set_xticks(plot_x)
        ax.set_xticklabels(plot_labels, rotation=15, ha="right")
        ax.set_title(subplot_title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_table_tex(
    summaries: dict[str, dict[str, float | int | None]],
    output_path: Path,
    caption: str,
) -> None:
    rows = []
    for label, summary in summaries.items():
        rows.append(
            " & ".join([
                label,
                _format_metric(summary.get("matched_iou_vs_reference_mean")),
                _format_metric(summary.get("mask_recall_at_0_5_mean")),
                _format_metric(summary.get("num_valid_masks_mean")),
                _format_metric(summary.get("mean_stability_mean")),
            ])
            + r" \\"
        )

    content = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & Matched IoU & Recall@0.5 & Valid masks & Stability \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    output_path.write_text(content + "\n", encoding="utf-8")


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def load_infer_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json in infer run: {run_dir}")

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    masks_dir = run_dir / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks/ in infer run: {run_dir}")

    mask_paths = {
        path.stem: path for path in sorted(masks_dir.glob("*.npz"))
    }
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found under {masks_dir}")

    return meta, mask_paths


def add_resize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resize-h",
        type=int,
        default=None,
        help="Resize input height (pixels) before SAM; requires --resize-w.",
    )
    parser.add_argument(
        "--resize-w",
        type=int,
        default=None,
        help="Resize input width (pixels) before SAM; requires --resize-h.",
    )


def resize_hw_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.resize_h is None and args.resize_w is None:
        return None
    if args.resize_h is None or args.resize_w is None:
        raise ValueError("--resize-h and --resize-w must be specified together.")
    if args.resize_h <= 0 or args.resize_w <= 0:
        raise ValueError(
            f"resize dimensions must be positive, got H={args.resize_h}, W={args.resize_w}"
        )
    return (args.resize_h, args.resize_w)


def add_amg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-score-thresh", type=float, default=0.90)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=int, default=100)


def amg_kwargs_from_args(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
