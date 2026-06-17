#!/usr/bin/env python3
"""Build flat symlink folders for DehazeDDPM fine-tuning.

The original DehazeDDPM loader expects paired hazy / GT image names. This
script keeps that layout and additionally writes per-sample depth symlinks plus
metadata so later model variants can consume physical priors.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

FOG_LEVELS = ("light", "medium", "heavy")
BETA_MAP = {
    "light": 0.06,
    "medium": 0.20,
    "heavy": 0.50,
}
DEPTH_SUFFIX = "_raw_depth_meter.npy"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _repo_data_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_gt(gt_dir: str, stem: str) -> Optional[str]:
    for ext in IMAGE_SUFFIXES:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _resolve_depth(gt_dir: str, stem: str, entry: Optional[Dict[str, Any]]) -> Optional[str]:
    if entry is not None:
        depth_path = entry.get("depth")
        if isinstance(depth_path, str) and os.path.isfile(depth_path):
            return depth_path

    p = os.path.join(gt_dir, stem + DEPTH_SUFFIX)
    if os.path.isfile(p):
        return p
    return None


def _find_split_manifest(splits_root: str, manifest_path: Optional[str]) -> Optional[str]:
    if manifest_path is not None:
        return manifest_path

    candidates = sorted(
        os.path.join(splits_root, name)
        for name in os.listdir(splits_root)
        if name.startswith("split_manifest_seed") and name.endswith(".json")
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        joined = "\n  ".join(candidates)
        raise ValueError(
            "Found multiple split manifests. Pass --manifest explicitly:\n"
            f"  {joined}"
        )
    return candidates[0]


def _load_split_manifest(splits_root: str, manifest_path: Optional[str]) -> Optional[Dict[str, Any]]:
    manifest_path = _find_split_manifest(splits_root, manifest_path)
    if manifest_path is None:
        print("No split manifest found; using fallback depth suffix and BETA_MAP.")
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Using split manifest: {manifest_path}")
    return manifest


def _manifest_entries_by_id(
    manifest: Optional[Dict[str, Any]],
    split: str,
) -> Dict[str, Dict[str, Any]]:
    if manifest is None:
        return {}

    entries = manifest.get("splits", {}).get(split, [])
    return {
        entry["id"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }


def _beta_for_level(
    level: str,
    entry: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> float:
    if entry is not None:
        beta = entry.get("beta", {}).get(level)
        if beta is not None:
            return float(beta)

    if manifest is not None:
        beta = manifest.get("input", {}).get("beta_by_fog_level", {}).get(level)
        if beta is not None:
            return float(beta)

    return float(BETA_MAP[level])


def _symlink_relative(target: str, link_path: str) -> None:
    link_dir = os.path.dirname(link_path)
    rel = os.path.relpath(target, link_dir)
    if os.path.lexists(link_path):
        if os.path.islink(link_path) or os.path.isfile(link_path):
            os.remove(link_path)
        else:
            raise IsADirectoryError(f"Refuse to replace non-file: {link_path}")
    os.symlink(rel, link_path)


def prepare_split(
    splits_root: str,
    split: str,
    out_root: str,
    *,
    manifest: Optional[Dict[str, Any]],
    dry_run: bool,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Returns (n_hazy_links, n_gt_links, n_depth_links, metadata_rows)."""
    n_hazy = 0
    n_gt = 0
    n_depth = 0
    out_hazy = os.path.join(out_root, f"{split}_hazy")
    out_gt = os.path.join(out_root, f"{split}_gt")
    out_depth = os.path.join(out_root, f"{split}_depth")
    rows: List[Dict[str, Any]] = []
    manifest_entries = _manifest_entries_by_id(manifest, split)

    if not dry_run:
        os.makedirs(out_hazy, exist_ok=True)
        os.makedirs(out_gt, exist_ok=True)
        os.makedirs(out_depth, exist_ok=True)

    for level in FOG_LEVELS:
        hazy_dir = os.path.join(splits_root, split, "hazy", level)
        gt_dir = os.path.join(splits_root, split, "gt")
        if not os.path.isdir(hazy_dir):
            raise FileNotFoundError(hazy_dir)
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(gt_dir)

        for name in sorted(os.listdir(hazy_dir)):
            if not name.lower().endswith(".png"):
                continue
            stem, _ = os.path.splitext(name)
            src_hazy = os.path.join(hazy_dir, name)
            dst_name = f"{level}_{stem}.png"
            dst_hazy = os.path.join(out_hazy, dst_name)
            src_gt = _resolve_gt(gt_dir, stem)
            if src_gt is None:
                print(f"Missing GT for {split}/{level} {stem}", file=sys.stderr)
                sys.exit(1)
            dst_gt = os.path.join(out_gt, dst_name)

            entry = manifest_entries.get(stem)
            src_depth = _resolve_depth(gt_dir, stem, entry)
            if src_depth is None:
                print(f"Missing depth npy for {split}/{level} {stem}", file=sys.stderr)
                sys.exit(1)
            dst_depth_name = f"{level}_{stem}.npy"
            dst_depth = os.path.join(out_depth, dst_depth_name)
            beta = _beta_for_level(level, entry, manifest)

            if dry_run:
                print(f"would link {dst_hazy} -> {src_hazy}")
                print(f"would link {dst_gt} -> {src_gt}")
                print(f"would link {dst_depth} -> {src_depth}")
            else:
                _symlink_relative(src_hazy, dst_hazy)
                _symlink_relative(src_gt, dst_gt)
                _symlink_relative(src_depth, dst_depth)
            n_hazy += 1
            n_gt += 1
            n_depth += 1

            rows.append(
                {
                    "split": split,
                    "level": level,
                    "id": stem,
                    "sample_name": dst_name,
                    "beta": beta,
                    "hazy": os.path.relpath(dst_hazy, out_root),
                    "gt": os.path.relpath(dst_gt, out_root),
                    "depth": os.path.relpath(dst_depth, out_root),
                    "source_hazy": os.path.relpath(src_hazy, out_root),
                    "source_gt": os.path.relpath(src_gt, out_root),
                    "source_depth": os.path.relpath(src_depth, out_root),
                }
            )

    return n_hazy, n_gt, n_depth, rows


def _write_metadata(out_root: str, rows: List[Dict[str, Any]], dry_run: bool) -> None:
    csv_path = os.path.join(out_root, "metadata.csv")
    jsonl_path = os.path.join(out_root, "metadata.jsonl")

    if dry_run:
        print(f"would write {len(rows)} rows to {csv_path}")
        print(f"would write {len(rows)} rows to {jsonl_path}")
        return

    os.makedirs(out_root, exist_ok=True)
    fieldnames = [
        "split",
        "level",
        "id",
        "sample_name",
        "beta",
        "hazy",
        "gt",
        "depth",
        "source_hazy",
        "source_gt",
        "source_depth",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"metadata: {csv_path}")
    print(f"metadata: {jsonl_path}")


def _clean_managed_outputs(out_root: str, dry_run: bool) -> None:
    managed_dirs = [
        "train_hazy",
        "train_gt",
        "train_depth",
        "val_hazy",
        "val_gt",
        "val_depth",
    ]
    managed_files = ["metadata.csv", "metadata.jsonl"]

    for name in managed_dirs:
        path = os.path.join(out_root, name)
        if not os.path.exists(path):
            continue
        if dry_run:
            print(f"would remove directory {path}")
        else:
            shutil.rmtree(path)

    for name in managed_files:
        path = os.path.join(out_root, name)
        if not os.path.lexists(path):
            continue
        if dry_run:
            print(f"would remove file {path}")
        else:
            os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-root",
        default=None,
        help="Path to splits/ (default: <this_dir>/splits)",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root for finetune/ (default: <this_dir>/finetune)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Split manifest JSON (default: auto-detect split_manifest_seed*.json under splits-root)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove managed finetune outputs before rebuilding",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = _repo_data_dir()
    splits_root = args.splits_root or os.path.join(data_dir, "splits")
    out_root = args.out_root or os.path.join(data_dir, "finetune")
    manifest = _load_split_manifest(splits_root, args.manifest)

    if not args.no_clean:
        _clean_managed_outputs(out_root, dry_run=args.dry_run)

    all_rows: List[Dict[str, Any]] = []
    for split in ("train", "val"):
        nh, ng, nd, rows = prepare_split(
            splits_root,
            split,
            out_root,
            manifest=manifest,
            dry_run=args.dry_run,
        )
        all_rows.extend(rows)
        print(f"{split}: {nh} hazy + {ng} gt + {nd} depth symlinks under {out_root}")

    _write_metadata(out_root, all_rows, dry_run=args.dry_run)

    if args.dry_run:
        print("(dry-run; no files created)")


if __name__ == "__main__":
    main()
