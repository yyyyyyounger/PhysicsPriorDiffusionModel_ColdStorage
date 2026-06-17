#!/usr/bin/env python3
"""Depth-stratified comparison of SAM infer runs against clear GT-SAM.

This is an offline analysis script. It reuses saved mask NPZ files from
run_sam_infer.py and metric depth maps generated from the clear images.
Reference masks are assigned to Near/Middle/Far bins by median depth.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_DEPTH_DIR = PROJECT_ROOT / "data/cold_depth_metric_vitl_980"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sam_core import (
    DEFAULT_COMPARE_OUTPUT,
    RECALL_THRESHOLDS,
    extract_masks,
    load_annotations_npz,
    load_infer_run,
    mask_iou,
    save_csv,
)

DEPTH_BINS = (
    ("Near", None, 4.0),
    ("Middle", 4.0, 10.0),
    ("Far", 10.0, None),
)


def depth_path_for_sample(depth_dir: Path, sample_id: str, suffix: str) -> Path:
    return depth_dir / f"{sample_id}{suffix}"


def resize_depth_to_mask(depth: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == mask_shape:
        return depth

    height, width = mask_shape
    return cv2.resize(
        depth.astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def median_depth_for_mask(depth: np.ndarray, mask: np.ndarray) -> float | None:
    values = depth[mask]
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def depth_bin_label(
    depth_m: float | None,
    near_max_m: float,
    far_min_m: float,
) -> str | None:
    if depth_m is None:
        return None
    if depth_m <= near_max_m:
        return "Near"
    if depth_m <= far_min_m:
        return "Middle"
    return "Far"


def bin_rows_template() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "depth_bin": label,
            "reference_num_masks": 0,
            "query_num_valid_masks": 0,
            "sample_ids": set(),
            "ref_depth_values": [],
            "query_stability_values": [],
            "best_ious": [],
        }
        for label, _, _ in DEPTH_BINS
    }


def summarize_bin(acc: dict[str, Any], recall_thresholds: tuple[float, ...]) -> dict[str, Any]:
    ref_count = int(acc["reference_num_masks"])
    query_count = int(acc["query_num_valid_masks"])
    best_ious = np.asarray(acc["best_ious"], dtype=np.float64)
    ref_depths = np.asarray(acc["ref_depth_values"], dtype=np.float64)
    stabilities = np.asarray(acc["query_stability_values"], dtype=np.float64)

    row: dict[str, Any] = {
        "depth_bin": acc["depth_bin"],
        "sample_count": len(acc["sample_ids"]),
        "reference_num_masks": ref_count,
        "query_num_valid_masks": query_count,
        "query_valid_mask_ratio": (
            float(query_count / ref_count) if ref_count > 0 else None
        ),
        "matched_iou_vs_reference": (
            float(best_ious.mean()) if best_ious.size else None
        ),
        "mean_reference_depth_m": (
            float(ref_depths.mean()) if ref_depths.size else None
        ),
        "median_reference_depth_m": (
            float(np.median(ref_depths)) if ref_depths.size else None
        ),
        "query_mean_stability": (
            float(stabilities.mean()) if stabilities.size else None
        ),
    }

    for threshold in recall_thresholds:
        key = f"mask_recall_at_{str(threshold).replace('.', '_')}"
        missing_key = f"missing_mask_rate_at_{str(threshold).replace('.', '_')}"
        recall = (
            float(np.mean(best_ious >= threshold)) if best_ious.size else None
        )
        row[key] = recall
        row[missing_key] = 1.0 - recall if recall is not None else None

    return row


def compare_query_by_depth(
    reference_mask_paths: dict[str, Path],
    query_meta: dict[str, Any],
    query_mask_paths: dict[str, Path],
    sample_ids: list[str],
    depth_dir: Path,
    depth_suffix: str,
    min_match_area: int,
    recall_thresholds: tuple[float, ...],
    near_max_m: float,
    far_min_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_mask_rows: list[dict[str, Any]] = []
    bins = bin_rows_template()

    for sample_id in sample_ids:
        ref_anns = load_annotations_npz(reference_mask_paths[sample_id])
        query_anns = load_annotations_npz(query_mask_paths[sample_id])
        ref_masks = extract_masks(ref_anns, min_area=min_match_area)
        query_masks = extract_masks(query_anns, min_area=min_match_area)
        depth_path = depth_path_for_sample(depth_dir, sample_id, depth_suffix)
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)

        raw_depth = np.load(depth_path)
        if ref_masks:
            depth = resize_depth_to_mask(raw_depth, ref_masks[0].shape)
        elif query_masks:
            depth = resize_depth_to_mask(raw_depth, query_masks[0].shape)
        else:
            continue

        for ann in query_anns:
            if ann["area"] < min_match_area:
                continue
            query_mask = np.asarray(ann["segmentation"], dtype=bool)
            query_depth = median_depth_for_mask(depth, query_mask)
            query_bin = depth_bin_label(query_depth, near_max_m, far_min_m)
            if query_bin is None:
                continue
            bins[query_bin]["query_num_valid_masks"] += 1
            bins[query_bin]["query_stability_values"].append(ann["stability_score"])

        for ref_index, ref_mask in enumerate(ref_masks):
            ref_depth = median_depth_for_mask(depth, ref_mask)
            ref_bin = depth_bin_label(ref_depth, near_max_m, far_min_m)
            if ref_bin is None:
                continue

            if query_masks:
                best_iou = max(mask_iou(ref_mask, query_mask) for query_mask in query_masks)
            else:
                best_iou = 0.0

            bins[ref_bin]["reference_num_masks"] += 1
            bins[ref_bin]["sample_ids"].add(sample_id)
            bins[ref_bin]["ref_depth_values"].append(ref_depth)
            bins[ref_bin]["best_ious"].append(best_iou)

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "query_tag": query_meta["tag"],
                "query_run": query_meta.get("run_dir", ""),
                "depth_bin": ref_bin,
                "reference_mask_index": ref_index,
                "reference_mask_area": int(ref_mask.sum()),
                "reference_median_depth_m": ref_depth,
                "best_iou": best_iou,
            }
            for threshold in recall_thresholds:
                row[f"matched_at_{str(threshold).replace('.', '_')}"] = (
                    int(best_iou >= threshold)
                )
            per_mask_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for label, _, _ in DEPTH_BINS:
        row = summarize_bin(bins[label], recall_thresholds)
        row = {
            "query_tag": query_meta["tag"],
            "query_run": query_meta.get("run_dir", ""),
            **row,
        }
        summary_rows.append(row)

    return per_mask_rows, summary_rows


def make_compare_name(ref_tag: str, query_tags: list[str], output_name: str | None) -> str:
    if output_name:
        return output_name
    if len(query_tags) == 1:
        return f"{ref_tag}_vs_{query_tags[0]}_depth"
    return "_vs_".join([ref_tag, *query_tags]) + "_depth"


def save_depth_summary_table_tex(
    rows: list[dict[str, Any]],
    output_path: Path,
    caption: str,
) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "--"
        return f"{float(value):.3f}"

    body = []
    for row in rows:
        body.append(
            " & ".join([
                str(row["query_tag"]),
                str(row["depth_bin"]),
                fmt(row["matched_iou_vs_reference"]),
                fmt(row["mask_recall_at_0_5"]),
                fmt(row["query_valid_mask_ratio"]),
                fmt(row["missing_mask_rate_at_0_5"]),
            ])
            + r" \\"
        )

    content = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Fog level & Depth bin & Matched IoU & Recall@0.5 & Valid mask ratio & Missing mask rate \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    output_path.write_text(content + "\n", encoding="utf-8")


def run_depth_compare(args: argparse.Namespace) -> Path:
    reference_dir = args.reference.resolve()
    query_dirs = [path.resolve() for path in args.query]
    depth_dir = args.depth_dir.resolve()
    output_root = args.output.resolve()

    ref_meta, ref_mask_paths = load_infer_run(reference_dir)
    ref_meta["run_dir"] = str(reference_dir)

    query_payloads: list[tuple[dict[str, Any], dict[str, Path], str]] = []
    for query_dir in query_dirs:
        query_meta, query_mask_paths = load_infer_run(query_dir)
        query_meta["run_dir"] = str(query_dir)
        query_payloads.append((query_meta, query_mask_paths, query_meta["tag"]))

    sample_ids = sorted(set(ref_mask_paths) & set.intersection(
        *[set(paths.keys()) for _, paths, _ in query_payloads]
    ))
    if not sample_ids:
        raise ValueError(
            "No overlapping sample_id found between reference and query runs."
        )

    missing_depth = [
        sample_id
        for sample_id in sample_ids
        if not depth_path_for_sample(depth_dir, sample_id, args.depth_suffix).is_file()
    ]
    if missing_depth:
        preview = ", ".join(missing_depth[:5])
        raise FileNotFoundError(
            f"Missing depth maps for {len(missing_depth)} samples under {depth_dir}: "
            f"{preview}"
        )

    all_mask_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []

    for query_meta, query_mask_paths, _ in query_payloads:
        mask_rows, summary_rows = compare_query_by_depth(
            reference_mask_paths=ref_mask_paths,
            query_meta=query_meta,
            query_mask_paths=query_mask_paths,
            sample_ids=sample_ids,
            depth_dir=depth_dir,
            depth_suffix=args.depth_suffix,
            min_match_area=args.min_match_area,
            recall_thresholds=tuple(args.recall_thresholds),
            near_max_m=args.near_max_m,
            far_min_m=args.far_min_m,
        )
        all_mask_rows.extend(mask_rows)
        all_summary_rows.extend(summary_rows)

    compare_name = make_compare_name(
        ref_meta["tag"],
        [tag for _, _, tag in query_payloads],
        args.output_name,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    compare_dir = output_root / f"{compare_name}_{timestamp}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    per_mask_csv = compare_dir / "per_reference_mask.csv"
    per_depth_bin_csv = compare_dir / "per_depth_bin.csv"
    save_csv(per_mask_csv, all_mask_rows)
    save_csv(per_depth_bin_csv, all_summary_rows)

    summary_payload = {
        "config": {
            "reference": {
                "run_dir": str(reference_dir),
                "tag": ref_meta["tag"],
                "input_dir": ref_meta.get("input_dir"),
                "amg": ref_meta.get("amg"),
            },
            "queries": [
                {
                    "run_dir": query_meta["run_dir"],
                    "tag": query_tag,
                    "input_dir": query_meta.get("input_dir"),
                    "amg": query_meta.get("amg"),
                }
                for query_meta, _, query_tag in query_payloads
            ],
            "depth_dir": str(depth_dir),
            "depth_suffix": args.depth_suffix,
            "depth_bins": [
                {"label": "Near", "rule": f"d <= {args.near_max_m:g} m"},
                {
                    "label": "Middle",
                    "rule": f"{args.near_max_m:g} m < d <= {args.far_min_m:g} m",
                },
                {"label": "Far", "rule": f"d > {args.far_min_m:g} m"},
            ],
            "sample_count": len(sample_ids),
            "min_match_area": args.min_match_area,
            "recall_thresholds": list(args.recall_thresholds),
            "depth_resize": "Depth maps are resized to the SAM mask size when shapes differ.",
        },
        "by_query_depth_bin": all_summary_rows,
    }
    summary_json = compare_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    if args.save_tex:
        tex_path = compare_dir / "summary_depth_table.tex"
        save_depth_summary_table_tex(
            all_summary_rows,
            tex_path,
            caption=f"Depth-stratified SAM metrics relative to {ref_meta['tag']} reference.",
        )

    latest_link = output_root / f"latest_{compare_name}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(compare_dir.name)

    print(f"Depth compare directory: {compare_dir}")
    print(f"Per-mask CSV:            {per_mask_csv}")
    print(f"Per-depth-bin CSV:       {per_depth_bin_csv}")
    print(f"Summary JSON:            {summary_json}")
    if args.save_tex:
        print(f"LaTeX table:             {compare_dir / 'summary_depth_table.tex'}")
    print(f"Latest link:             {latest_link} -> {compare_dir.name}")
    return compare_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth-stratified SAM comparison against a clear reference run."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference infer run directory (typically clear GT-SAM).",
    )
    parser.add_argument(
        "--query",
        type=Path,
        nargs="+",
        required=True,
        help="One or more query infer run directories to compare.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=DEFAULT_DEPTH_DIR,
        help=f"Directory containing metric depth NPY files (default: {DEFAULT_DEPTH_DIR}).",
    )
    parser.add_argument(
        "--depth-suffix",
        type=str,
        default="_raw_depth_meter.npy",
        help="Depth filename suffix appended to sample_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_COMPARE_OUTPUT,
        help=f"Root directory for compare outputs (default: {DEFAULT_COMPARE_OUTPUT}).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional name prefix for the compare output directory.",
    )
    parser.add_argument(
        "--min-match-area",
        type=int,
        default=100,
        help="Ignore masks smaller than this area when matching and binning.",
    )
    parser.add_argument(
        "--recall-thresholds",
        type=float,
        nargs="+",
        default=list(RECALL_THRESHOLDS),
        help="IoU thresholds for mask recall metrics.",
    )
    parser.add_argument(
        "--near-max-m",
        type=float,
        default=4.0,
        help="Near bin upper bound in meters.",
    )
    parser.add_argument(
        "--far-min-m",
        type=float,
        default=10.0,
        help="Far bin lower bound in meters; Middle is near-max < d <= far-min.",
    )
    parser.add_argument(
        "--save-tex",
        action="store_true",
        help="Also write summary_depth_table.tex for paper tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.near_max_m >= args.far_min_m:
        raise ValueError("--near-max-m must be smaller than --far-min-m.")
    run_depth_compare(args)


if __name__ == "__main__":
    main()
