#!/usr/bin/env python3
"""Compare SAM infer runs against a reference run (typically clear GT-SAM).

Reads saved mask NPZ files from run_sam_infer.py outputs and computes:
  - matched_iou_vs_reference
  - mask_recall_at_0_5
  - num_valid_masks
  - mean_stability
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sam_core import (
    DEFAULT_COMPARE_OUTPUT,
    RECALL_THRESHOLDS,
    aggregate_metric_rows,
    extract_masks,
    load_annotations_npz,
    load_infer_run,
    query_metrics,
    save_compare_plot,
    save_csv,
    save_summary_table_tex,
)

METRIC_KEYS = [
    "num_valid_masks",
    "mean_stability",
    "matched_iou_vs_reference",
    "mask_recall_at_0_5",
    "reference_num_masks",
]


def compare_one_query(
    reference_masks: dict[str, list[np.ndarray]],
    query_meta: dict[str, Any],
    query_mask_paths: dict[str, Path],
    sample_ids: list[str],
    min_match_area: int,
    recall_thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        ref_masks = reference_masks[sample_id]
        query_path = query_mask_paths[sample_id]
        query_anns = load_annotations_npz(query_path)
        stats = query_metrics(
            query_anns,
            ref_masks,
            min_area=min_match_area,
            recall_thresholds=recall_thresholds,
        )
        rows.append({
            "sample_id": sample_id,
            "query_tag": query_meta["tag"],
            "query_run": query_meta.get("run_dir", ""),
            **stats,
        })
    return rows


def run_compare(args: argparse.Namespace) -> Path:
    reference_dir = args.reference.resolve()
    query_dirs = [path.resolve() for path in args.query]
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

    reference_masks = {
        sample_id: extract_masks(
            load_annotations_npz(ref_mask_paths[sample_id]),
            min_area=args.min_match_area,
        )
        for sample_id in sample_ids
    }

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float | int | None]] = {}

    for query_meta, query_mask_paths, query_tag in query_payloads:
        rows = compare_one_query(
            reference_masks=reference_masks,
            query_meta=query_meta,
            query_mask_paths=query_mask_paths,
            sample_ids=sample_ids,
            min_match_area=args.min_match_area,
            recall_thresholds=tuple(args.recall_thresholds),
        )
        all_rows.extend(rows)
        summaries[query_tag] = aggregate_metric_rows(rows, METRIC_KEYS)

    if args.output_name:
        compare_name = args.output_name
    elif len(query_payloads) == 1:
        compare_name = f"{ref_meta['tag']}_vs_{query_payloads[0][2]}"
    else:
        joined = "_vs_".join([ref_meta["tag"], *[tag for _, _, tag in query_payloads]])
        compare_name = joined

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    compare_dir = output_root / f"{compare_name}_{timestamp}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = compare_dir / "per_sample.csv"
    save_csv(per_sample_csv, all_rows)

    config = {
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
        "sample_count": len(sample_ids),
        "min_match_area": args.min_match_area,
        "recall_thresholds": list(args.recall_thresholds),
    }

    summary_payload = {
        "config": config,
        "by_query": summaries,
    }
    summary_json = compare_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    plot_path = compare_dir / "summary_metrics.pdf"
    save_compare_plot(
        summaries,
        plot_path,
        title=f"SAM Metrics vs Reference ({ref_meta['tag']})",
    )

    if args.save_tex:
        tex_path = compare_dir / "summary_table.tex"
        save_summary_table_tex(
            summaries,
            tex_path,
            caption=f"SAM segmentation metrics relative to {ref_meta['tag']} reference.",
        )

    latest_link = output_root / f"latest_{compare_name}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(compare_dir.name)

    print(f"Compare directory: {compare_dir}")
    print(f"Per-sample CSV:    {per_sample_csv}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Summary plot:      {plot_path}")
    if args.save_tex:
        print(f"LaTeX table:       {compare_dir / 'summary_table.tex'}")
    print(f"Latest link:       {latest_link} -> {compare_dir.name}")
    return compare_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAM infer runs against a reference run."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference infer run directory (typically clear GT-SAM)",
    )
    parser.add_argument(
        "--query",
        type=Path,
        nargs="+",
        required=True,
        help="One or more query infer run directories to compare",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_COMPARE_OUTPUT,
        help=f"Root directory for compare outputs (default: {DEFAULT_COMPARE_OUTPUT})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional name prefix for the compare output directory",
    )
    parser.add_argument(
        "--min-match-area",
        type=int,
        default=100,
        help="Ignore masks smaller than this area when matching.",
    )
    parser.add_argument(
        "--recall-thresholds",
        type=float,
        nargs="+",
        default=list(RECALL_THRESHOLDS),
        help="IoU thresholds for mask recall metrics.",
    )
    parser.add_argument(
        "--save-tex",
        action="store_true",
        help="Also write summary_table.tex for paper tables.",
    )
    return parser.parse_args()


def main() -> None:
    run_compare(parse_args())


if __name__ == "__main__":
    main()
