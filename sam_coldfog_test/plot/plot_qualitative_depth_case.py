#!/usr/bin/env python3
"""Plot a qualitative case for depth-dependent SAM degradation.

Layout:
    Row 1: Clear image, Depth map
    Row 2: Column headers Light / Medium / Heavy
    Row 3: Mask overlays only

Green masks are one-to-one matched reference masks.
Red masks are unmatched reference masks.
Blue masks are unmatched extra query masks from the fog image.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

TEXTWIDTH_IN = 5.768
DISPLAY_ASPECT = 4 / 3
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = int(round(DISPLAY_WIDTH / DISPLAY_ASPECT))

SCRIPT_DIR = Path(__file__).resolve().parent
COLDFOG_TEST_DIR = SCRIPT_DIR.parent
SEGMENT_ANYTHING_DIR = COLDFOG_TEST_DIR.parent
PROJECT_ROOT = SEGMENT_ANYTHING_DIR.parent

DEFAULT_INFER_ROOT = COLDFOG_TEST_DIR / "results/infer"
DEFAULT_DEPTH_DIR = PROJECT_ROOT / "data/cold_depth_metric_vitl_980"

CONDITIONS = ("clear", "light", "medium", "heavy")
QUERY_CONDITIONS = ("light", "medium", "heavy")
QUERY_COLUMN_LABELS = ("Light", "Medium", "Heavy")
MASK_ROW_LABEL = "Masks"
DEPTH_BINS = ("Near", "Middle", "Far")

LEGEND_COLORS = (
    (0.05, 0.70, 0.20),
    (0.90, 0.08, 0.06),
    (0.00, 0.32, 0.95),
)
LEGEND_LABELS = ("matched", "missing", "extra")


def configure_matplotlib() -> None:
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


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_run_dir(infer_root: Path, condition: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit

    latest = infer_root / f"latest_{condition}"
    if latest.exists():
        return latest.resolve()

    candidates = sorted(
        path for path in infer_root.glob(f"{condition}_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directory found for condition {condition!r} under {infer_root}"
        )
    return candidates[-1].resolve()


def load_image(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image = image.astype(np.float32, copy=False)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def resize_image_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    height, width = shape
    return cv2.resize(
        image.astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def crop_box_4_3(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    if width / height > DISPLAY_ASPECT:
        crop_width = int(round(height * DISPLAY_ASPECT))
        x0 = (width - crop_width) // 2
        return 0, height, x0, x0 + crop_width
    crop_height = int(round(width / DISPLAY_ASPECT))
    y0 = (height - crop_height) // 2
    return y0, y0 + crop_height, 0, width


def crop_to_box(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = crop_box
    return image[y0:y1, x0:x1]


def resize_display_image(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    cropped = crop_to_box(image, crop_box)
    return cv2.resize(
        cropped.astype(np.float32, copy=False),
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def load_annotations_npz(path: Path) -> list[dict[str, np.ndarray | int | float]]:
    data = np.load(path)
    segmentations = data["segmentations"]
    areas = data["areas"]
    stability_scores = data["stability_scores"]
    predicted_ious = data["predicted_ious"]

    annotations: list[dict[str, np.ndarray | int | float]] = []
    for index in range(len(areas)):
        annotations.append({
            "segmentation": np.asarray(segmentations[index], dtype=bool),
            "area": int(areas[index]),
            "stability_score": float(stability_scores[index]),
            "predicted_iou": float(predicted_ious[index]),
        })
    return annotations


def valid_masks(
    annotations: Iterable[dict[str, np.ndarray | int | float]],
    min_area: int,
) -> list[np.ndarray]:
    return [
        np.asarray(ann["segmentation"], dtype=bool)
        for ann in annotations
        if int(ann["area"]) >= min_area
    ]


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


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def iou_matrix(
    reference_masks: list[np.ndarray],
    query_masks: list[np.ndarray],
) -> np.ndarray:
    ious = np.zeros((len(reference_masks), len(query_masks)), dtype=np.float32)
    for ref_index, ref_mask in enumerate(reference_masks):
        for query_index, query_mask in enumerate(query_masks):
            ious[ref_index, query_index] = mask_iou(ref_mask, query_mask)
    return ious


def depth_bin_for_mask(
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


def selected_reference_masks(
    reference_masks: list[np.ndarray],
    depth: np.ndarray,
    near_max_m: float,
    far_min_m: float,
    depth_bins: list[str],
    max_masks: int,
) -> list[np.ndarray]:
    selected_bins = set(depth_bins)
    selected_payload: list[tuple[int, np.ndarray]] = []
    for mask in reference_masks:
        depth_m = median_depth_for_mask(depth, mask)
        depth_bin = depth_bin_for_mask(depth_m, near_max_m, far_min_m)
        if depth_bin in selected_bins:
            selected_payload.append((int(mask.sum()), mask))

    selected_payload.sort(key=lambda item: item[0], reverse=True)
    if max_masks > 0:
        selected_payload = selected_payload[:max_masks]
    return [mask for _, mask in selected_payload]


def one_to_one_match_flags(
    reference_masks: list[np.ndarray],
    query_masks: list[np.ndarray],
    threshold: float,
) -> tuple[list[bool], list[bool]]:
    ref_matched = [False] * len(reference_masks)
    query_matched = [False] * len(query_masks)
    if not reference_masks or not query_masks:
        return ref_matched, query_matched

    ious = iou_matrix(reference_masks, query_masks)
    candidate_order = [
        sorted(
            (
                query_index
                for query_index in range(len(query_masks))
                if ious[ref_index, query_index] >= threshold
            ),
            key=lambda query_index: float(ious[ref_index, query_index]),
            reverse=True,
        )
        for ref_index in range(len(reference_masks))
    ]
    ref_order = sorted(
        range(len(reference_masks)),
        key=lambda ref_index: max(
            (
                float(ious[ref_index, query_index])
                for query_index in candidate_order[ref_index]
            ),
            default=0.0,
        ),
        reverse=True,
    )
    match_query_to_ref = [-1] * len(query_masks)

    def try_match(ref_index: int, seen_query: list[bool]) -> bool:
        for query_index in candidate_order[ref_index]:
            if seen_query[query_index]:
                continue
            seen_query[query_index] = True
            current_ref = match_query_to_ref[query_index]
            if current_ref == -1 or try_match(current_ref, seen_query):
                match_query_to_ref[query_index] = ref_index
                return True
        return False

    for ref_index in ref_order:
        try_match(ref_index, [False] * len(query_masks))

    for query_index, ref_index in enumerate(match_query_to_ref):
        if ref_index == -1:
            continue
        ref_matched[ref_index] = True
        query_matched[query_index] = True

    return ref_matched, query_matched


def overlay_mask_status(
    image: np.ndarray,
    reference_masks: list[np.ndarray],
    reference_matched_flags: list[bool],
    extra_query_masks: list[np.ndarray],
    alpha: float,
) -> np.ndarray:
    canvas = image.copy()
    green = np.array(LEGEND_COLORS[0], dtype=np.float32)
    red = np.array(LEGEND_COLORS[1], dtype=np.float32)
    blue = np.array(LEGEND_COLORS[2], dtype=np.float32)

    for mask in extra_query_masks:
        canvas[mask] = canvas[mask] * (1.0 - alpha) + blue * alpha

    for mask, matched in zip(reference_masks, reference_matched_flags):
        color = green if matched else red
        canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha

    outline = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    for mask in extra_query_masks:
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(outline, contours, -1, (0, 82, 242), thickness=2)

    for mask, matched in zip(reference_masks, reference_matched_flags):
        color_bgr = (35, 190, 45) if matched else (230, 20, 20)
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(outline, contours, -1, color_bgr, thickness=2)

    return outline.astype(np.float32) / 255.0


def draw_mask_legend(ax: plt.Axes) -> None:
    ax.axis("off")
    handles = [
        Patch(facecolor=color, edgecolor="0.25", linewidth=0.6)
        for color in LEGEND_COLORS
    ]
    ax.legend(
        handles,
        LEGEND_LABELS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        handlelength=0.9,
        handleheight=0.9,
        labelspacing=0.7,
        borderpad=0.0,
        fontsize=9,
    )


def style_image_panel(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, DISPLAY_WIDTH)
    ax.set_ylim(DISPLAY_HEIGHT, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(DISPLAY_HEIGHT / DISPLAY_WIDTH)


def draw_row_label(ax: plt.Axes, label: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
        transform=ax.transAxes,
    )


def draw_column_headers(fig: plt.Figure, grid_spec: GridSpec) -> None:
    header = grid_spec.subgridspec(1, len(QUERY_COLUMN_LABELS), wspace=0.02)
    for column_index, label in enumerate(QUERY_COLUMN_LABELS):
        ax = fig.add_subplot(header[0, column_index])
        ax.axis("off")
        ax.text(
            0.5,
            0.15,
            label,
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )


def sample_options(index_rows: list[dict[str, str]]) -> list[str]:
    return [row["sample_id"] for row in index_rows]


def choose_sample_id(args: argparse.Namespace, clear_index: list[dict[str, str]]) -> str:
    options = sample_options(clear_index)
    if args.list_samples:
        for index, sample_id in enumerate(options):
            print(f"{index:03d}  {sample_id}")
        raise SystemExit(0)

    if args.sample_id:
        if args.sample_id not in options:
            raise ValueError(f"Sample {args.sample_id!r} not found in clear index.")
        return args.sample_id

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(options):
            raise ValueError(
                f"--sample-index must be in [0, {len(options) - 1}], got {args.sample_index}."
            )
        return options[args.sample_index]

    return options[0]


def image_path_by_sample(index_rows: list[dict[str, str]], sample_id: str) -> Path:
    for row in index_rows:
        if row["sample_id"] == sample_id:
            return Path(row["image_path"])
    raise ValueError(f"Sample {sample_id!r} not found in index.")


def draw_case(args: argparse.Namespace) -> Path:
    configure_matplotlib()

    run_dirs = {
        "clear": resolve_run_dir(args.infer_root, "clear", args.clear_run),
        "light": resolve_run_dir(args.infer_root, "light", args.light_run),
        "medium": resolve_run_dir(args.infer_root, "medium", args.medium_run),
        "heavy": resolve_run_dir(args.infer_root, "heavy", args.heavy_run),
    }
    indices = {
        condition: read_index(run_dir / "index.csv")
        for condition, run_dir in run_dirs.items()
    }

    sample_id = choose_sample_id(args, indices["clear"])
    image_paths = {
        condition: image_path_by_sample(index_rows, sample_id)
        for condition, index_rows in indices.items()
    }
    images = {condition: load_image(path) for condition, path in image_paths.items()}

    clear_mask_path = run_dirs["clear"] / "masks" / f"{sample_id}.npz"
    clear_anns = load_annotations_npz(clear_mask_path)
    reference_masks = valid_masks(clear_anns, args.min_area)
    if not reference_masks:
        raise ValueError(f"No valid reference masks for sample {sample_id!r}.")
    mask_shape = reference_masks[0].shape
    images = {
        condition: resize_image_to_shape(image, mask_shape)
        for condition, image in images.items()
    }

    raw_depth = np.load(args.depth_dir / f"{sample_id}{args.depth_suffix}")
    depth_for_masks = resize_depth_to_mask(raw_depth, reference_masks[0].shape)
    depth_masks = selected_reference_masks(
        reference_masks,
        depth_for_masks,
        near_max_m=args.near_max_m,
        far_min_m=args.far_min_m,
        depth_bins=args.depth_bins,
        max_masks=args.max_masks,
    )
    if not depth_masks:
        raise ValueError(
            f"No reference masks found for sample {sample_id!r} in depth bins: "
            f"{', '.join(args.depth_bins)}."
        )

    overlay_images: dict[str, np.ndarray] = {}
    for condition in QUERY_CONDITIONS:
        query_mask_path = run_dirs[condition] / "masks" / f"{sample_id}.npz"
        query_anns = load_annotations_npz(query_mask_path)
        query_masks = valid_masks(query_anns, args.min_area)
        ref_flags, query_flags = one_to_one_match_flags(
            depth_masks,
            query_masks,
            args.iou_threshold,
        )
        extra_query_masks = [
            query_mask
            for query_mask, matched in zip(query_masks, query_flags)
            if not matched
        ]
        overlay_images[condition] = overlay_mask_status(
            images[condition],
            depth_masks,
            ref_flags,
            extra_query_masks,
            alpha=args.overlay_alpha,
        )

    depth_display = resize_depth_to_mask(raw_depth, images["clear"].shape[:2])
    display_crop_box = crop_box_4_3(reference_masks[0].shape)
    clear_display = resize_display_image(images["clear"], display_crop_box)
    depth_display = resize_display_image(depth_display, display_crop_box)
    overlay_displays = {
        condition: resize_display_image(overlay_images[condition], display_crop_box)
        for condition in QUERY_CONDITIONS
    }

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.35))
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.0, 0.12, 1.0],
        width_ratios=[0.14, 1.0, 1.0, 1.0],
        hspace=0.16,
        wspace=0.02,
    )

    draw_mask_legend(fig.add_subplot(grid[0, 0]))

    top_content = grid[0, 1:4].subgridspec(
        1,
        5,
        width_ratios=[0.55, 1.0, 0.05, 1.10, 0.55],
        wspace=0.02,
    )
    fig.add_subplot(top_content[0, 0]).axis("off")
    fig.add_subplot(top_content[0, 4]).axis("off")

    ax_clear = fig.add_subplot(top_content[0, 1])
    ax_clear.imshow(clear_display)
    ax_clear.set_title("Clear image")
    style_image_panel(ax_clear)

    depth_cell = top_content[0, 3].subgridspec(1, 2, width_ratios=[1.0, 0.10], wspace=0.03)
    ax_depth = fig.add_subplot(depth_cell[0, 0])
    depth_image = ax_depth.imshow(depth_display, cmap=args.depth_cmap)
    ax_depth.set_title("Depth map")
    style_image_panel(ax_depth)
    ax_cbar = fig.add_subplot(depth_cell[0, 1])

    draw_column_headers(fig, grid[1, 1:4])
    fig.add_subplot(grid[1, 0]).axis("off")

    draw_row_label(fig.add_subplot(grid[2, 0]), MASK_ROW_LABEL)

    for column_index, condition in enumerate(QUERY_CONDITIONS):
        ax_masks = fig.add_subplot(grid[2, 1 + column_index])
        ax_masks.imshow(overlay_displays[condition])
        style_image_panel(ax_masks)

    cbar = fig.colorbar(depth_image, cax=ax_cbar)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel("")
    cbar.ax.text(
        0.5,
        1.03,
        "m",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = SCRIPT_DIR / f"qualitative_depth_case_{sample_id}"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return output_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a five-column qualitative depth case visualization."
    )
    parser.add_argument(
        "--infer-root",
        type=Path,
        default=DEFAULT_INFER_ROOT,
        help=f"Root directory containing infer runs (default: {DEFAULT_INFER_ROOT}).",
    )
    parser.add_argument("--clear-run", type=Path, default=None)
    parser.add_argument("--light-run", type=Path, default=None)
    parser.add_argument("--medium-run", type=Path, default=None)
    parser.add_argument("--heavy-run", type=Path, default=None)
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample id to visualize, e.g. sdm_20260212_0292.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Choose a sample by its row index in clear/index.csv.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="Print available sample ids and exit.",
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
        "--far-min-m",
        type=float,
        default=10.0,
        help="Far masks are defined by median depth greater than this value.",
    )
    parser.add_argument(
        "--near-max-m",
        type=float,
        default=4.0,
        help="Near masks are defined by median depth less than or equal to this value.",
    )
    parser.add_argument(
        "--depth-bins",
        nargs="+",
        choices=DEPTH_BINS,
        default=list(DEPTH_BINS),
        help="Depth bins to display. Default: Near Middle Far.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matched masks.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Ignore masks smaller than this area.",
    )
    parser.add_argument(
        "--max-masks",
        "--max-far-masks",
        dest="max_masks",
        type=int,
        default=0,
        help="Optional cap on selected reference masks, largest first; 0 means no cap.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.42,
        help="Transparency for red/green far mask fill.",
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
    if args.sample_id and args.sample_index is not None:
        raise ValueError("Use either --sample-id or --sample-index, not both.")
    if args.near_max_m >= args.far_min_m:
        raise ValueError("--near-max-m must be smaller than --far-min-m.")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be within [0, 1].")
    output_stem = draw_case(args)
    print(f"Saved: {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
