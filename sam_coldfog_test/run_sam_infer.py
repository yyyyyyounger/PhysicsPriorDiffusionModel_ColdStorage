#!/usr/bin/env python3
"""Run SAM automatic mask generation on all images in a directory.

Each run writes masks, per-image stats, optional overlay PNGs, and meta.json.
Use run_sam_compare.py afterward to compare two or more infer runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sam_core import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INFER_OUTPUT,
    SPLIT_SEED,
    amg_kwargs_from_args,
    add_amg_args,
    add_resize_args,
    build_mask_generator,
    image_stem,
    infer_stats,
    list_image_paths,
    parse_gpu_ids,
    read_rgb,
    resize_hw_from_args,
    resolve_device,
    save_annotations_npz,
    save_csv,
    save_overlay_png,
    split_items,
)


def infer_image_paths(
    image_paths: list[Path],
    run_dir: Path,
    checkpoint: Path,
    device: str,
    min_area: int,
    amg_kwargs: dict[str, int | float],
    save_overlays: bool,
    resize_hw: tuple[int, int] | None = None,
    progress_prefix: str = "",
) -> list[dict[str, Any]]:
    mask_generator = build_mask_generator(checkpoint, device, **amg_kwargs)
    masks_dir = run_dir / "masks"
    overlays_dir = run_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        stem = image_stem(image_path)
        image = read_rgb(image_path, resize_hw=resize_hw)
        annotations = mask_generator.generate(image)
        stats = infer_stats(annotations, min_area=min_area)

        save_annotations_npz(masks_dir / f"{stem}.npz", annotations)
        if save_overlays:
            save_overlay_png(image, annotations, overlays_dir / f"{stem}.png")

        rows.append({
            "sample_id": stem,
            "image_path": str(image_path.resolve()),
            **stats,
        })

        label = f"{progress_prefix} " if progress_prefix else ""
        print(f"{label}[{index}/{total}] {stem}", flush=True)

    return rows


def _worker_infer_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    gpu_id = payload["gpu_id"]
    device = resolve_device("cuda", gpu_id)
    prefix = f"[GPU {gpu_id}]"
    return infer_image_paths(
        image_paths=[Path(path) for path in payload["image_paths"]],
        run_dir=Path(payload["run_dir"]),
        checkpoint=Path(payload["checkpoint"]),
        device=device,
        min_area=payload["min_area"],
        amg_kwargs=payload["amg_kwargs"],
        save_overlays=payload["save_overlays"],
        resize_hw=tuple(payload["resize_hw"]) if payload["resize_hw"] else None,
        progress_prefix=prefix,
    )


def sort_rows_by_paths(rows: list[dict[str, Any]], image_paths: list[Path]) -> list[dict[str, Any]]:
    order = {image_stem(path): index for index, path in enumerate(image_paths)}
    return sorted(rows, key=lambda row: order[row["sample_id"]])


def run_infer(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(input_dir, args.manifest.resolve() if args.manifest else None)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    gpu_ids = parse_gpu_ids(args.gpu_ids, args.gpu_id)
    amg_kwargs = amg_kwargs_from_args(args)
    resize_hw = resize_hw_from_args(args)
    tag = args.tag or input_dir.name

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    split_info = None
    if args.device == "cpu" or not torch.cuda.is_available():
        device = resolve_device(args.device, gpu_ids[0] if gpu_ids else None)
        rows = infer_image_paths(
            image_paths=image_paths,
            run_dir=run_dir,
            checkpoint=args.checkpoint.resolve(),
            device=device,
            min_area=args.min_area,
            amg_kwargs=amg_kwargs,
            save_overlays=args.save_overlays,
            resize_hw=resize_hw,
        )
    elif gpu_ids is None:
        device = resolve_device(args.device, None)
        rows = infer_image_paths(
            image_paths=image_paths,
            run_dir=run_dir,
            checkpoint=args.checkpoint.resolve(),
            device=device,
            min_area=args.min_area,
            amg_kwargs=amg_kwargs,
            save_overlays=args.save_overlays,
            resize_hw=resize_hw,
        )
    elif len(gpu_ids) == 1:
        device = resolve_device(args.device, gpu_ids[0])
        rows = infer_image_paths(
            image_paths=image_paths,
            run_dir=run_dir,
            checkpoint=args.checkpoint.resolve(),
            device=device,
            min_area=args.min_area,
            amg_kwargs=amg_kwargs,
            save_overlays=args.save_overlays,
            resize_hw=resize_hw,
        )
    else:
        if args.device != "cuda":
            raise ValueError("Multi-GPU mode requires --device cuda.")

        device_count = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id < 0 or gpu_id >= device_count:
                raise ValueError(
                    f"Invalid GPU id {gpu_id}; available GPU IDs: 0..{device_count - 1}"
                )

        shards = split_items([str(path) for path in image_paths], num_shards=len(gpu_ids), seed=SPLIT_SEED)
        device = ",".join(f"cuda:{gpu_id}" for gpu_id in gpu_ids)
        split_info = {
            "seed": SPLIT_SEED,
            "gpu_ids": gpu_ids,
            "shard_sizes": [len(shard) for shard in shards],
            "shards": {str(gpu_id): shard for gpu_id, shard in zip(gpu_ids, shards)},
        }

        print(
            f"Multi-GPU infer on {gpu_ids} "
            f"(seed={SPLIT_SEED}, shard sizes={split_info['shard_sizes']})",
            flush=True,
        )

        payloads = [
            {
                "gpu_id": gpu_id,
                "image_paths": shard,
                "run_dir": str(run_dir),
                "checkpoint": str(args.checkpoint.resolve()),
                "min_area": args.min_area,
                "amg_kwargs": amg_kwargs,
                "save_overlays": args.save_overlays,
                "resize_hw": list(resize_hw) if resize_hw else None,
            }
            for gpu_id, shard in zip(gpu_ids, shards)
        ]

        ctx = get_context("spawn")
        with ctx.Pool(processes=len(gpu_ids)) as pool:
            shard_rows = pool.map(_worker_infer_shard, payloads)

        rows = sort_rows_by_paths(
            [row for shard in shard_rows for row in shard],
            image_paths,
        )

    index_csv = run_dir / "index.csv"
    save_csv(index_csv, rows)

    meta = {
        "tag": tag,
        "input_dir": str(input_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device,
        "gpu_id": gpu_ids[0] if gpu_ids and len(gpu_ids) == 1 else None,
        "gpu_ids": gpu_ids,
        "split": split_info,
        "sample_count": len(image_paths),
        "save_overlays": args.save_overlays,
        "min_area": args.min_area,
        "resize_h": resize_hw[0] if resize_hw else None,
        "resize_w": resize_hw[1] if resize_hw else None,
        "amg": amg_kwargs,
        "created_at": timestamp,
    }
    meta_path = run_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    latest_link = output_root / f"latest_{tag}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    print(f"Run directory: {run_dir}")
    print(f"Index CSV:     {index_csv}")
    print(f"Meta JSON:     {meta_path}")
    if args.save_overlays:
        print(f"Overlays:      {run_dir / 'overlays'}")
    print(f"Latest link:   {latest_link} -> {run_dir.name}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM on all images in a directory and save masks/overlays."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input images (e.g. data/sam_eval/clear)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Label for this run (default: basename of --input-dir)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.json to control sample order/filtering",
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
        default=DEFAULT_INFER_OUTPUT,
        help=f"Root directory for infer runs (default: {DEFAULT_INFER_OUTPUT})",
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
            "Comma-separated CUDA device indices for parallel infer, e.g. 0,1. "
            f"Images are shuffled with seed {SPLIT_SEED} and split evenly across GPUs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (for quick tests).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum mask area used in index.csv stats.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save per-image SAM overlay PNGs under overlays/",
    )
    add_resize_args(parser)
    add_amg_args(parser)
    return parser.parse_args()


def main() -> None:
    run_infer(parse_args())


if __name__ == "__main__":
    main()
