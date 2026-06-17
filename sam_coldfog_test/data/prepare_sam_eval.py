#!/usr/bin/env python3
"""Prepare SAM evaluation data under segment-anything/data/sam_eval/.

Reads sample IDs from DehazeDDPM test_metadata.json and copies (without
modifying the source split) the paired clear / light / medium / heavy
images from data/splits/test into:

    sam_eval/
      clear/    <- gt/
      light/    <- hazy/light/
      medium/   <- hazy/medium/
      heavy/    <- hazy/heavy/

All outputs use a unified {sample_id}.png filename so downstream scripts
(e.g. coldfog_test/run_sam_overlay.py) can reference the same image_id
across folders.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FOG_LEVELS = frozenset({"heavy", "medium", "low", "light"})
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_METADATA = (
    Path(__file__).resolve().parents[2]
    / "DehazeDDPM/plot/test_metadata.json"
)
DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2] / "data/splits/test"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sam_eval"
HAZY_LEVELS = ("light", "medium", "heavy")


def parse_sample_id(filename: str) -> str:
    """Extract sample ID by stripping fog-level suffix/prefix from filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    suffix = parts[-1].lower()
    if suffix in FOG_LEVELS:
        return "_".join(parts[:-1])
    prefix = parts[0].lower()
    if prefix in FOG_LEVELS:
        return "_".join(parts[1:])
    return stem


def find_gt_image(gt_dir: Path, sample_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = gt_dir / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def copy_as_png(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            f"Need OpenCV to convert {src} to PNG; install opencv-python or "
            "ensure GT files are already PNG."
        ) from exc

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"Failed to write PNG: {dst}")


def load_sample_ids(metadata_path: Path) -> list[str]:
    with metadata_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"No 'samples' list found in {metadata_path}")

    sample_ids = [parse_sample_id(item["filename"]) for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found in metadata.")

    return sample_ids


def prepare_sam_eval(
    metadata_path: Path,
    source_root: Path,
    output_root: Path,
    clean: bool = False,
) -> dict[str, int]:
    gt_dir = source_root / "gt"
    hazy_dirs = {
        level: source_root / "hazy" / level for level in HAZY_LEVELS
    }

    required_dirs = [gt_dir, *hazy_dirs.values()]
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing source directory: {path}")

    sample_ids = load_sample_ids(metadata_path)

    out_dirs = {"clear": output_root / "clear"}
    out_dirs.update({level: output_root / level for level in HAZY_LEVELS})
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    copied = {name: 0 for name in out_dirs}
    missing: list[str] = []

    for sample_id in sample_ids:
        gt_src = find_gt_image(gt_dir, sample_id)
        hazy_srcs = {
            level: hazy_dirs[level] / f"{sample_id}.png" for level in HAZY_LEVELS
        }

        if gt_src is None or any(not src.is_file() for src in hazy_srcs.values()):
            missing.append(sample_id)
            continue

        copy_as_png(gt_src, out_dirs["clear"] / f"{sample_id}.png")
        for level, src in hazy_srcs.items():
            copy_as_png(src, out_dirs[level] / f"{sample_id}.png")
            copied[level] += 1
        copied["clear"] += 1

    if missing:
        raise FileNotFoundError(
            "Missing paired files for sample IDs: " + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    manifest = {
        "metadata": str(metadata_path.resolve()),
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "sample_count": len(sample_ids),
        "copied_per_folder": copied,
        "sample_ids": sample_ids,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy test split images into sam_eval/clear|light|medium|heavy."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Path to test_metadata.json (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source test split root (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output sam_eval root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directory before copying.",
    )
    args = parser.parse_args()

    copied = prepare_sam_eval(
        metadata_path=args.metadata,
        source_root=args.source,
        output_root=args.output,
        clean=args.clean,
    )

    print(f"Prepared SAM eval data at: {args.output.resolve()}")
    for folder, count in copied.items():
        print(f"  {folder}: {count} images")
    print(f"Manifest: {(args.output / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
