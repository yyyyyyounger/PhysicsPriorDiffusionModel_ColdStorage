#!/usr/bin/env python3
"""Prepare DehazeDDPM infer outputs for SAM evaluation.

DehazeDDPM ``infer.py`` writes ``{step}_{index}_out.png`` where *index* is the
1-based dataloader index from ``test_metadata.json``.  Downstream SAM scripts
(``run_sam_infer.py`` with ``--manifest``) expect unified filenames
``{sample_id}.png`` under ``data/sam_eval/``.

This script maps dehaze result files into:

    sam_eval/dehazed/       <- {sample_id}.png (symlink or copy)
    sam_eval/hazy_input/    <- optional; per-sample actual hazy input

Usage::

    python data/prepare_dehazed_sam_eval.py \\
        --dehaze-results /path/to/DehazeDDPM/experiments/.../results

Then run SAM on the staged folder::

    python coldfog_test/run_sam_infer.py \\
        --input-dir data/sam_eval/dehazed \\
        --tag dehazed \\
        --manifest data/sam_eval/manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from prepare_sam_eval import DEFAULT_METADATA, DEFAULT_OUTPUT, parse_sample_id

DEFAULT_DEHAZED_DIR = DEFAULT_OUTPUT / "dehazed"
DEFAULT_HAZY_INPUT_DIR = DEFAULT_OUTPUT / "hazy_input"
FOG_LEVEL_TO_FOLDER = {"low": "light", "light": "light", "medium": "medium", "heavy": "heavy"}


def load_metadata_samples(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"No 'samples' list found in {metadata_path}")

    required = {"index", "filename", "fog_level"}
    for item in samples:
        if not required.issubset(item):
            raise ValueError(
                f"Each sample in {metadata_path} must contain {sorted(required)}; "
                f"got keys {sorted(item)}"
            )

    sample_ids = [parse_sample_id(item["filename"]) for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found in metadata.")

    return samples


def dehaze_result_path(dehaze_dir: Path, step: int, index: int) -> Path:
    return dehaze_dir / f"{step}_{index}_out.png"


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        dst.symlink_to(src.resolve())
        return

    shutil.copy2(src, dst)


def fog_folder_name(fog_level: str) -> str:
    folder = FOG_LEVEL_TO_FOLDER.get(fog_level.lower())
    if folder is None:
        raise ValueError(
            f"Unsupported fog_level {fog_level!r}; expected one of "
            f"{sorted(FOG_LEVEL_TO_FOLDER)}"
        )
    return folder


def prepare_dehazed_sam_eval(
    dehaze_results: Path,
    metadata_path: Path,
    output_dehazed: Path,
    step: int = 0,
    use_symlink: bool = True,
    with_hazy_input: bool = False,
    hazy_input_dir: Path | None = None,
    sam_eval_root: Path | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    if not dehaze_results.is_dir():
        raise FileNotFoundError(f"Dehaze results directory not found: {dehaze_results}")

    samples = load_metadata_samples(metadata_path)

    if clean:
        if output_dehazed.exists():
            shutil.rmtree(output_dehazed)
        if with_hazy_input and hazy_input_dir is not None and hazy_input_dir.exists():
            shutil.rmtree(hazy_input_dir)

    output_dehazed.mkdir(parents=True, exist_ok=True)
    if with_hazy_input:
        if hazy_input_dir is None:
            raise ValueError("hazy_input_dir is required when with_hazy_input=True")
        hazy_input_dir.mkdir(parents=True, exist_ok=True)

    sam_eval_root = sam_eval_root or DEFAULT_OUTPUT
    missing_dehaze: list[str] = []
    missing_hazy: list[str] = []
    mapping_rows: list[dict[str, Any]] = []

    for sample in samples:
        index = int(sample["index"])
        filename = sample["filename"]
        fog_level = sample["fog_level"]
        sample_id = parse_sample_id(filename)

        src_dehaze = dehaze_result_path(dehaze_results, step, index)
        dst_dehaze = output_dehazed / f"{sample_id}.png"

        if not src_dehaze.is_file():
            missing_dehaze.append(f"{sample_id} <- {src_dehaze.name}")
            continue

        link_or_copy(src_dehaze, dst_dehaze, use_symlink=use_symlink)

        row: dict[str, Any] = {
            "index": index,
            "sample_id": sample_id,
            "filename": filename,
            "fog_level": fog_level,
            "dehaze_source": str(src_dehaze.resolve()),
            "dehazed_output": str(dst_dehaze.resolve()),
        }

        if with_hazy_input:
            fog_folder = fog_folder_name(fog_level)
            src_hazy = sam_eval_root / fog_folder / f"{sample_id}.png"
            dst_hazy = hazy_input_dir / f"{sample_id}.png"
            if not src_hazy.is_file():
                missing_hazy.append(f"{sample_id} ({fog_level}) <- {src_hazy}")
            else:
                link_or_copy(src_hazy, dst_hazy, use_symlink=use_symlink)
                row["hazy_source"] = str(src_hazy.resolve())
                row["hazy_output"] = str(dst_hazy.resolve())
                row["hazy_folder"] = fog_folder

        mapping_rows.append(row)

    if missing_dehaze:
        raise FileNotFoundError(
            "Missing dehaze result files for: "
            + ", ".join(missing_dehaze[:10])
            + (" ..." if len(missing_dehaze) > 10 else "")
        )

    if missing_hazy:
        raise FileNotFoundError(
            "Missing hazy input files under sam_eval (run prepare_sam_eval.py first): "
            + ", ".join(missing_hazy[:10])
            + (" ..." if len(missing_hazy) > 10 else "")
        )

    mapping_payload = {
        "metadata": str(metadata_path.resolve()),
        "dehaze_results": str(dehaze_results.resolve()),
        "step": step,
        "use_symlink": use_symlink,
        "output_dehazed": str(output_dehazed.resolve()),
        "sample_count": len(mapping_rows),
        "mapping": mapping_rows,
    }
    if with_hazy_input and hazy_input_dir is not None:
        mapping_payload["output_hazy_input"] = str(hazy_input_dir.resolve())
        mapping_payload["sam_eval_root"] = str(sam_eval_root.resolve())

    mapping_path = output_dehazed / "dehazed_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2, ensure_ascii=False)

    return {
        "dehazed": len(mapping_rows),
        "hazy_input": len(mapping_rows) if with_hazy_input else 0,
        "mapping_path": mapping_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map DehazeDDPM infer outputs ({step}_{index}_out.png) to "
            "sam_eval/dehazed/{sample_id}.png for SAM inference."
        )
    )
    parser.add_argument(
        "--dehaze-results",
        type=Path,
        required=True,
        help="DehazeDDPM infer results directory containing {step}_{index}_out.png",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Path to test_metadata.json (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Training step prefix in infer output filenames (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DEHAZED_DIR,
        help=f"Output directory for staged dehazed images (default: {DEFAULT_DEHAZED_DIR})",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (default: symlink).",
    )
    parser.add_argument(
        "--with-hazy-input",
        action="store_true",
        help=(
            "Also stage sam_eval/hazy_input/ using each sample's fog_level "
            "from test_metadata.json (requires prepare_sam_eval.py first)."
        ),
    )
    parser.add_argument(
        "--hazy-input-dir",
        type=Path,
        default=DEFAULT_HAZY_INPUT_DIR,
        help=f"Output directory for staged hazy inputs (default: {DEFAULT_HAZY_INPUT_DIR})",
    )
    parser.add_argument(
        "--sam-eval-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Root of sam_eval folders for --with-hazy-input (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directories before staging.",
    )
    args = parser.parse_args()

    result = prepare_dehazed_sam_eval(
        dehaze_results=args.dehaze_results.resolve(),
        metadata_path=args.metadata.resolve(),
        output_dehazed=args.output.resolve(),
        step=args.step,
        use_symlink=not args.copy,
        with_hazy_input=args.with_hazy_input,
        hazy_input_dir=args.hazy_input_dir.resolve() if args.with_hazy_input else None,
        sam_eval_root=args.sam_eval_root.resolve(),
        clean=args.clean,
    )

    print(f"Staged dehazed images: {result['dehazed']} -> {args.output.resolve()}")
    if args.with_hazy_input:
        print(f"Staged hazy inputs:  {result['hazy_input']} -> {args.hazy_input_dir.resolve()}")
    print(f"Mapping JSON:          {result['mapping_path'].resolve()}")
    print()
    print("Next steps:")
    print("  python coldfog_test/run_sam_infer.py \\")
    print(f"    --input-dir {args.output} \\")
    print("    --tag dehazed \\")
    print("    --manifest data/sam_eval/manifest.json")


if __name__ == "__main__":
    main()
