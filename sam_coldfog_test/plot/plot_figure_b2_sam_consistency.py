#!/usr/bin/env python3
"""Plot Figure B.2: qualitative SAM consistency after PPDM dehazing.

Layout:
    Row 1: Clear image | Depth map
    Row 2: Hazy masks | PPDM-dehazed masks

Green and red overlays denote matched and missing clear-image reference masks
across Near, Middle, and Far depth bins. Extra query-only masks are omitted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_qualitative_depth_case import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    LEGEND_COLORS,
    LEGEND_LABELS,
    TEXTWIDTH_IN,
    configure_matplotlib,
    crop_box_4_3,
    image_path_by_sample,
    load_annotations_npz,
    load_image,
    mask_iou,
    one_to_one_match_flags,
    overlay_mask_status,
    read_index,
    resize_depth_to_mask,
    resize_display_image,
    resize_image_to_shape,
    resolve_run_dir,
    selected_reference_masks,
    style_image_panel,
    valid_masks,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COLDFOG_TEST_DIR = SCRIPT_DIR.parent
SEGMENT_ANYTHING_DIR = COLDFOG_TEST_DIR.parent
PROJECT_ROOT = SEGMENT_ANYTHING_DIR.parent

DEFAULT_INFER_ROOT = COLDFOG_TEST_DIR / "results/infer"
DEFAULT_DEPTH_DIR = PROJECT_ROOT / "data/cold_depth_metric_vitl_980"
DEFAULT_SAMPLE_ID = "ggl_20260202_0013"
DEFAULT_DEHAZED_TAG = "dehazed_ddim100_physical_v1"
DEPTH_BINS = ("Near", "Middle", "Far")
FIGURE_LEGEND_COLORS = LEGEND_COLORS[:2]
FIGURE_LEGEND_LABELS = LEGEND_LABELS[:2]


def reference_match_metrics(
    reference_masks: list[np.ndarray],
    query_masks: list[np.ndarray],
    recall_threshold: float,
) -> tuple[float, float]:
    best_ious: list[float] = []
    for ref_mask in reference_masks:
        if not query_masks:
            best_ious.append(0.0)
            continue
        best_ious.append(
            max(mask_iou(ref_mask, query_mask) for query_mask in query_masks)
        )
    iou_values = np.asarray(best_ious, dtype=np.float64)
    return float(iou_values.mean()), float(np.mean(iou_values >= recall_threshold))


def comparison_overlay(
    image: np.ndarray,
    reference_masks: list[np.ndarray],
    query_masks: list[np.ndarray],
    iou_threshold: float,
    alpha: float,
) -> np.ndarray:
    ref_flags, _query_flags = one_to_one_match_flags(
        reference_masks,
        query_masks,
        iou_threshold,
    )
    return overlay_mask_status(
        image,
        reference_masks,
        ref_flags,
        extra_query_masks=[],
        alpha=alpha,
    )


def draw_mask_row_title(ax: plt.Axes, title: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.0,
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        transform=ax.transAxes,
    )


def draw_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="0.25", linewidth=0.6)
        for color in FIGURE_LEGEND_COLORS
    ]
    ax.legend(
        handles,
        FIGURE_LEGEND_LABELS,
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=1.2,
        fontsize=9,
    )
    ax.axis("off")


def draw_case(args: argparse.Namespace) -> tuple[Path, dict[str, tuple[float, float]]]:
    configure_matplotlib()

    run_dirs = {
        "clear": resolve_run_dir(args.infer_root, "clear", args.clear_run),
        "hazy": resolve_run_dir(args.infer_root, args.hazy_tag, args.hazy_run),
        "dehazed": resolve_run_dir(
            args.infer_root,
            args.dehazed_tag,
            args.dehazed_run,
        ),
    }
    indices = {
        condition: read_index(run_dir / "index.csv")
        for condition, run_dir in run_dirs.items()
    }

    sample_id = args.sample_id
    image_paths = {
        condition: image_path_by_sample(index_rows, sample_id)
        for condition, index_rows in indices.items()
    }

    images = {
        condition: load_image(path)
        for condition, path in image_paths.items()
    }

    clear_mask_path = run_dirs["clear"] / "masks" / f"{sample_id}.npz"
    hazy_mask_path = run_dirs["hazy"] / "masks" / f"{sample_id}.npz"
    dehazed_mask_path = run_dirs["dehazed"] / "masks" / f"{sample_id}.npz"

    reference_masks_all = valid_masks(
        load_annotations_npz(clear_mask_path),
        args.min_area,
    )
    hazy_masks = valid_masks(load_annotations_npz(hazy_mask_path), args.min_area)
    dehazed_masks = valid_masks(load_annotations_npz(dehazed_mask_path), args.min_area)
    if not reference_masks_all:
        raise ValueError(f"No valid reference masks for sample {sample_id!r}.")

    mask_shape = reference_masks_all[0].shape
    images = {
        condition: resize_image_to_shape(image, mask_shape)
        for condition, image in images.items()
    }

    raw_depth = np.load(args.depth_dir / f"{sample_id}{args.depth_suffix}")
    depth_for_masks = resize_depth_to_mask(raw_depth, mask_shape)
    reference_masks = selected_reference_masks(
        reference_masks_all,
        depth_for_masks,
        near_max_m=args.near_max_m,
        far_min_m=args.far_min_m,
        depth_bins=args.depth_bins,
        max_masks=args.max_masks,
    )
    if not reference_masks:
        raise ValueError(
            f"No reference masks found for {sample_id!r} in depth bins: "
            f"{', '.join(args.depth_bins)}."
        )

    hazy_metrics = reference_match_metrics(
        reference_masks,
        hazy_masks,
        recall_threshold=args.iou_threshold,
    )
    dehazed_metrics = reference_match_metrics(
        reference_masks,
        dehazed_masks,
        recall_threshold=args.iou_threshold,
    )
    metrics = {
        "Hazy": hazy_metrics,
        "PPDM": dehazed_metrics,
    }

    hazy_overlay = comparison_overlay(
        images["hazy"],
        reference_masks,
        hazy_masks,
        iou_threshold=args.iou_threshold,
        alpha=args.overlay_alpha,
    )
    dehazed_overlay = comparison_overlay(
        images["dehazed"],
        reference_masks,
        dehazed_masks,
        iou_threshold=args.iou_threshold,
        alpha=args.overlay_alpha,
    )
    depth_display = resize_depth_to_mask(raw_depth, mask_shape)
    display_crop_box = crop_box_4_3(mask_shape)
    top_row_images = {
        "Clear image": resize_display_image(images["clear"], display_crop_box),
        "Depth map": resize_display_image(depth_display, display_crop_box),
    }
    mask_row_images = {
        "Hazy masks": resize_display_image(hazy_overlay, display_crop_box),
        "PPDM-dehazed masks": resize_display_image(dehazed_overlay, display_crop_box),
    }

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.6))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.08, 0.10],
        width_ratios=[1.0, 1.08],
        hspace=0.30,
        wspace=0.04,
    )

    for col_index, (title, image) in enumerate(top_row_images.items()):
        ax = fig.add_subplot(grid[0, col_index])
        if title == "Depth map":
            depth_artist = ax.imshow(image, cmap=args.depth_cmap)
        else:
            ax.imshow(image)
        ax.set_title(title, pad=3)
        style_image_panel(ax)

    for col_index, (title, image) in enumerate(mask_row_images.items()):
        mask_cell = grid[1, col_index].subgridspec(
            2,
            1,
            height_ratios=[0.10, 1.0],
            hspace=0.10,
        )
        title_ax = fig.add_subplot(mask_cell[0, 0])
        draw_mask_row_title(title_ax, title)
        mask_ax = fig.add_subplot(mask_cell[1, 0])
        mask_ax.imshow(image)
        style_image_panel(mask_ax)

    cbar = fig.colorbar(
        depth_artist,
        ax=fig.axes[1],
        orientation="vertical",
        fraction=0.046,
        pad=0.02,
    )
    cbar.ax.set_ylabel("m", rotation=0, labelpad=6, fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)

    legend_ax = fig.add_subplot(grid[2, :])
    draw_legend(legend_ax)

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = SCRIPT_DIR / f"figure_b2_sam_consistency_{sample_id}"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_stem.with_suffix(".png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the four-panel Figure B.2 SAM consistency case.",
    )
    parser.add_argument(
        "--sample-id",
        default=DEFAULT_SAMPLE_ID,
        help=f"Sample id to visualize (default: {DEFAULT_SAMPLE_ID}).",
    )
    parser.add_argument(
        "--infer-root",
        type=Path,
        default=DEFAULT_INFER_ROOT,
        help=f"Root directory containing SAM inference runs (default: {DEFAULT_INFER_ROOT}).",
    )
    parser.add_argument("--clear-run", type=Path, default=None)
    parser.add_argument("--hazy-run", type=Path, default=None)
    parser.add_argument("--dehazed-run", type=Path, default=None)
    parser.add_argument(
        "--hazy-tag",
        default="heavy",
        help="SAM run tag for the hazy input (default: heavy).",
    )
    parser.add_argument(
        "--dehazed-tag",
        default=DEFAULT_DEHAZED_TAG,
        help=f"SAM run tag for PPDM-dehazed input (default: {DEFAULT_DEHAZED_TAG}).",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=DEFAULT_DEPTH_DIR,
        help=f"Directory containing metric depth files (default: {DEFAULT_DEPTH_DIR}).",
    )
    parser.add_argument(
        "--depth-suffix",
        default="_raw_depth_meter.npy",
        help="Depth filename suffix appended to sample_id.",
    )
    parser.add_argument(
        "--depth-bins",
        nargs="+",
        choices=DEPTH_BINS,
        default=list(DEPTH_BINS),
        help="Reference-mask depth bins to draw (default: Near Middle Far).",
    )
    parser.add_argument(
        "--near-max-m",
        type=float,
        default=4.0,
        help="Near masks have median depth <= this value.",
    )
    parser.add_argument(
        "--far-min-m",
        type=float,
        default=10.0,
        help="Far masks have median depth > this value.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for one-to-one matched masks.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Ignore masks smaller than this area.",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=0,
        help="Optional cap on selected reference masks, largest first; 0 means no cap.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.42,
        help="Transparency for mask fills.",
    )
    parser.add_argument(
        "--depth-cmap",
        default="cividis",
        help="Matplotlib colormap for the depth map.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without extension. PNG is written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.near_max_m >= args.far_min_m:
        raise ValueError("--near-max-m must be smaller than --far-min-m.")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be within [0, 1].")
    output_path, metrics = draw_case(args)
    for label in ("Hazy", "PPDM"):
        mean_iou, recall = metrics[label]
        print(f"{label}: IoU / Recall = {mean_iou:.2f} / {recall:.2f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
