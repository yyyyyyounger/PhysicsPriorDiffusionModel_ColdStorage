#!/usr/bin/env python3
"""Batch SAM evaluation on Cold-Fog test set (sam_eval).

Metrics per fog level (reference = SAM masks on clear GT, i.e. GT-SAM):
  - num_valid_masks
  - mean_stability
  - matched_iou_vs_gt_sam  (mean best IoU per GT-SAM mask)
  - mask_recall_at_0_5     (fraction of GT-SAM masks with best IoU >= 0.5)

Results are written under coldfog_test/results/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = REPO_ROOT / "data/sam_eval/manifest.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam_vit_h_4b8939.pth"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"
FOG_LEVELS = ("light", "medium", "heavy")
RECALL_THRESHOLDS = (0.5,)
SPLIT_SEED = 42


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


def match_against_gt_sam(
    gt_sam_masks: list[np.ndarray],
    fog_masks: list[np.ndarray],
    recall_thresholds: tuple[float, ...],
) -> dict[str, float | int | None]:
    if not gt_sam_masks:
        return {
            "gt_sam_num_masks": 0,
            "matched_iou_vs_gt_sam": None,
            **{f"mask_recall_at_{str(t).replace('.', '_')}": None for t in recall_thresholds},
        }

    best_ious: list[float] = []
    for gt_mask in gt_sam_masks:
        if not fog_masks:
            best_ious.append(0.0)
            continue
        best_ious.append(max(mask_iou(gt_mask, fog_mask) for fog_mask in fog_masks))

    recalls = {
        f"mask_recall_at_{str(th).replace('.', '_')}": float(np.mean(np.array(best_ious) >= th))
        for th in recall_thresholds
    }
    return {
        "gt_sam_num_masks": len(gt_sam_masks),
        "matched_iou_vs_gt_sam": float(np.mean(best_ious)),
        **recalls,
    }


def fog_metrics(
    annotations: list[dict[str, Any]],
    gt_sam_masks: list[np.ndarray],
    min_area: int,
    recall_thresholds: tuple[float, ...],
) -> dict[str, float | int | None]:
    fog_masks = extract_masks(annotations, min_area=min_area)
    stabilities = [ann["stability_score"] for ann in annotations if ann["area"] >= min_area]

    match_stats = match_against_gt_sam(gt_sam_masks, fog_masks, recall_thresholds)
    return {
        "num_valid_masks": len(fog_masks),
        "mean_stability": float(np.mean(stabilities)) if stabilities else None,
        **match_stats,
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


def load_sample_ids(manifest_path: Path, limit: int | None) -> list[str]:
    with manifest_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    sample_ids = payload["sample_ids"]
    if limit is not None:
        sample_ids = sample_ids[:limit]
    return sample_ids


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


def split_sample_ids(
    sample_ids: list[str],
    num_shards: int,
    seed: int = SPLIT_SEED,
) -> list[list[str]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)

    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for index, sample_id in enumerate(shuffled):
        shards[index % num_shards].append(sample_id)
    return shards


def sort_rows_by_manifest(rows: list[dict[str, Any]], sample_ids: list[str]) -> list[dict[str, Any]]:
    sample_order = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    fog_order = {level: index for index, level in enumerate(FOG_LEVELS)}
    return sorted(
        rows,
        key=lambda row: (sample_order[row["sample_id"]], fog_order[row["fog_level"]]),
    )


def amg_kwargs_from_args(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }


def evaluate_sample_ids(
    sample_ids: list[str],
    data_root: Path,
    checkpoint: Path,
    device: str,
    min_match_area: int,
    amg_kwargs: dict[str, int | float],
    progress_prefix: str = "",
) -> list[dict[str, Any]]:
    mask_generator = build_mask_generator(checkpoint, device, **amg_kwargs)
    rows: list[dict[str, Any]] = []
    total = len(sample_ids)

    for index, sample_id in enumerate(sample_ids, start=1):
        clear_path = data_root / "clear" / f"{sample_id}.png"
        if not clear_path.is_file():
            raise FileNotFoundError(clear_path)

        clear_anns = mask_generator.generate(read_rgb(clear_path))
        gt_sam_masks = extract_masks(clear_anns, min_area=min_match_area)

        for fog_level in FOG_LEVELS:
            fog_path = data_root / fog_level / f"{sample_id}.png"
            if not fog_path.is_file():
                raise FileNotFoundError(fog_path)

            fog_anns = mask_generator.generate(read_rgb(fog_path))
            stats = fog_metrics(
                fog_anns,
                gt_sam_masks,
                min_area=min_match_area,
                recall_thresholds=RECALL_THRESHOLDS,
            )
            rows.append({
                "sample_id": sample_id,
                "fog_level": fog_level,
                **stats,
            })

        label = f"{progress_prefix} " if progress_prefix else ""
        print(f"{label}[{index}/{total}] {sample_id}", flush=True)

    return rows


def _worker_eval_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    gpu_id = payload["gpu_id"]
    device = resolve_device("cuda", gpu_id)
    prefix = f"[GPU {gpu_id}]"
    return evaluate_sample_ids(
        sample_ids=payload["sample_ids"],
        data_root=Path(payload["data_root"]),
        checkpoint=Path(payload["checkpoint"]),
        device=device,
        min_match_area=payload["min_match_area"],
        amg_kwargs=payload["amg_kwargs"],
        progress_prefix=prefix,
    )


def aggregate_by_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    metric_keys = [
        "num_valid_masks",
        "mean_stability",
        "matched_iou_vs_gt_sam",
        "mask_recall_at_0_5",
        "gt_sam_num_masks",
    ]

    for level in FOG_LEVELS:
        level_rows = [row for row in rows if row["fog_level"] == level]
        summary[level] = {"sample_count": len(level_rows)}
        for key in metric_keys:
            values = [row[key] for row in level_rows if row.get(key) is not None]
            if not values:
                summary[level][f"{key}_mean"] = None
                summary[level][f"{key}_std"] = None
                continue
            arr = np.asarray(values, dtype=np.float64)
            summary[level][f"{key}_mean"] = float(arr.mean())
            summary[level][f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return summary


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(summary: dict[str, dict[str, float | int]], output_path: Path) -> None:
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

    metrics = [
        ("num_valid_masks_mean", "Valid masks"),
        ("mean_stability_mean", "Mean stability"),
        ("matched_iou_vs_gt_sam_mean", "Matched IoU vs GT-SAM"),
        ("mask_recall_at_0_5_mean", "Mask Recall@0.5"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    x = np.arange(len(FOG_LEVELS))
    width = 0.55

    for ax, (key, title) in zip(axes.ravel(), metrics):
        means = [summary[level].get(key) for level in FOG_LEVELS]
        stds = [summary[level].get(key.replace("_mean", "_std"), 0.0) for level in FOG_LEVELS]
        valid = [m is not None for m in means]
        plot_x = x[valid]
        plot_means = [m for m, ok in zip(means, valid) if ok]
        plot_stds = [s for s, ok in zip(stds, valid) if ok]
        plot_labels = [level for level, ok in zip(FOG_LEVELS, valid) if ok]

        ax.bar(plot_x, plot_means, width, yerr=plot_stds, capsize=4, color="#4C72B0")
        ax.set_xticks(plot_x)
        ax.set_xticklabels(plot_labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SAM Metrics by Fog Level (GT-SAM reference on clear)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_eval(args: argparse.Namespace) -> Path:
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = load_sample_ids(manifest_path, args.limit)
    gpu_ids = parse_gpu_ids(args.gpu_ids, args.gpu_id)
    amg_kwargs = amg_kwargs_from_args(args)

    if args.device == "cpu" or not torch.cuda.is_available():
        device = resolve_device(args.device, gpu_ids[0] if gpu_ids else None)
        rows = evaluate_sample_ids(
            sample_ids=sample_ids,
            data_root=data_root,
            checkpoint=args.checkpoint.resolve(),
            device=device,
            min_match_area=args.min_match_area,
            amg_kwargs=amg_kwargs,
        )
        split_info = None
    elif gpu_ids is None:
        device = resolve_device(args.device, None)
        rows = evaluate_sample_ids(
            sample_ids=sample_ids,
            data_root=data_root,
            checkpoint=args.checkpoint.resolve(),
            device=device,
            min_match_area=args.min_match_area,
            amg_kwargs=amg_kwargs,
        )
        split_info = None
    elif len(gpu_ids) == 1:
        device = resolve_device(args.device, gpu_ids[0])
        rows = evaluate_sample_ids(
            sample_ids=sample_ids,
            data_root=data_root,
            checkpoint=args.checkpoint.resolve(),
            device=device,
            min_match_area=args.min_match_area,
            amg_kwargs=amg_kwargs,
        )
        split_info = None
    else:
        if args.device != "cuda":
            raise ValueError("Multi-GPU mode requires --device cuda.")

        device_count = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id < 0 or gpu_id >= device_count:
                raise ValueError(
                    f"Invalid GPU id {gpu_id}; available GPU IDs: 0..{device_count - 1}"
                )

        shards = split_sample_ids(sample_ids, num_shards=len(gpu_ids), seed=SPLIT_SEED)
        device = ",".join(f"cuda:{gpu_id}" for gpu_id in gpu_ids)
        split_info = {
            "seed": SPLIT_SEED,
            "gpu_ids": gpu_ids,
            "shard_sizes": [len(shard) for shard in shards],
            "shards": {
                str(gpu_id): shard for gpu_id, shard in zip(gpu_ids, shards)
            },
        }

        print(
            f"Multi-GPU eval on {gpu_ids} "
            f"(seed={SPLIT_SEED}, shard sizes={split_info['shard_sizes']})",
            flush=True,
        )

        payloads = [
            {
                "gpu_id": gpu_id,
                "sample_ids": shard,
                "data_root": str(data_root),
                "checkpoint": str(args.checkpoint.resolve()),
                "min_match_area": args.min_match_area,
                "amg_kwargs": amg_kwargs,
            }
            for gpu_id, shard in zip(gpu_ids, shards)
        ]

        ctx = get_context("spawn")
        with ctx.Pool(processes=len(gpu_ids)) as pool:
            shard_rows = pool.map(_worker_eval_shard, payloads)

        rows = sort_rows_by_manifest(
            [row for shard in shard_rows for row in shard],
            sample_ids,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = run_dir / "per_sample.csv"
    save_csv(per_sample_csv, rows)

    summary = aggregate_by_level(rows)
    config = {
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device,
        "gpu_id": gpu_ids[0] if gpu_ids and len(gpu_ids) == 1 else None,
        "gpu_ids": gpu_ids,
        "split": split_info,
        "sample_count": len(sample_ids),
        "fog_levels": list(FOG_LEVELS),
        "reference": "SAM automatic masks on clear GT (GT-SAM)",
        "amg": amg_kwargs,
        "min_match_area": args.min_match_area,
        "recall_thresholds": list(RECALL_THRESHOLDS),
    }

    summary_payload = {"config": config, "by_fog_level": summary}
    summary_json = run_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    save_summary_plot(summary, run_dir / "summary_metrics.pdf")

    latest_link = output_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    print(f"Per-sample CSV: {per_sample_csv}")
    print(f"Summary JSON:   {summary_json}")
    print(f"Summary plot:   {run_dir / 'summary_metrics.pdf'}")
    print(f"Latest link:    {latest_link} -> {run_dir.name}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM on Cold-Fog sam_eval test set."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to sam_eval manifest.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_MANIFEST.parent,
        help="Root directory containing clear/light/medium/heavy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"SAM ViT-H checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for eval outputs (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SAM inference: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Single CUDA device index, e.g. 0 (maps to cuda:N). Ignored on cpu.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device indices for parallel eval, e.g. 0,1. "
            f"Samples are shuffled with seed {SPLIT_SEED} and split evenly across GPUs. "
            "Mutually exclusive with --gpu-id."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (for quick tests).",
    )
    parser.add_argument(
        "--min-match-area",
        type=int,
        default=100,
        help="Ignore masks smaller than this area when matching GT-SAM.",
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-score-thresh", type=float, default=0.90)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
