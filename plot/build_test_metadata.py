#!/usr/bin/env python3
"""Build test_metadata from dataset paths (same logic as infer.py dataloader).

infer.py calls ``Data.create_dataset(dataset_opt, phase)`` which builds
``LRHRDataset.paths`` via ``paired_paths_from_metadata`` (if configured) or
``paired_paths_from_folder`` (scandir order on hazy folder). Val dataloader
does not shuffle; infer output files use 1-based index: ``{step}_{index}_out.png``.

Usage (default: val test paths, config lines 35-37, same as infer.py)::

    python plot/build_test_metadata.py \\
        -c config/test_ColdFog_finetune_netH_physical_ddim20.json

Usage (train split)::

    python plot/build_test_metadata.py \\
        -c config/test_ColdFog_finetune_netH_physical_ddim20.json \\
        --phase train
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from os import path as osp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_NAME = 'Cold Storage Dehazing Dataset'
FOG_LEVELS = frozenset({'heavy', 'medium', 'low', 'light'})


def scandir_files(dir_path, suffix=None, recursive=False, full_path=False):
    """Mirror LRHRDataset.scandir (relative paths under dir_path)."""
    root = dir_path

    def _scandir(dp, suf, rec):
        for entry in os.scandir(dp):
            if not entry.name.startswith('.') and entry.is_file():
                return_path = entry.path if full_path else osp.relpath(entry.path, root)
                if suf is None or return_path.endswith(suf):
                    yield return_path
            elif entry.is_dir() and rec:
                yield from _scandir(entry.path, suf, rec)

    return list(_scandir(dir_path, suffix, recursive))


def paired_paths_from_folder(datarootlq, dataroothq):
    """Same pairing order as LRHRDataset.paired_paths_from_folder."""
    input_paths = scandir_files(datarootlq)
    paths = []
    for lq_path in input_paths:
        basename, ext = osp.splitext(osp.basename(lq_path))
        input_name = basename + ext
        paths.append({
            'lq_path': osp.join(datarootlq, lq_path),
            'gt_path': osp.join(dataroothq, input_name),
        })
    return paths


def parse_fog_level(filename: str) -> str:
    """Parse fog level from filename (prefix or suffix style)."""
    stem = osp.splitext(osp.basename(filename))[0]
    parts = stem.split('_')
    suffix = parts[-1].lower()
    if suffix in FOG_LEVELS:
        return suffix
    prefix = parts[0].lower()
    if prefix in FOG_LEVELS:
        return prefix
    return suffix


def paths_to_metadata(paths) -> list[dict]:
    records = []
    for idx, sample in enumerate(paths):
        filename = osp.basename(sample['lq_path'])
        records.append({
            'index': idx + 1,
            'filename': filename,
            'fog_level': parse_fog_level(filename),
        })
    return records


def load_dataset_opt(config_path: str, phase: str) -> dict:
    with open(config_path, encoding='utf-8') as f:
        opt = json.load(f)
    if phase not in opt['datasets']:
        raise KeyError('Phase {!r} not found in config datasets'.format(phase))
    return opt['datasets'][phase]


def build_paths(dataset_opt, phase: str):
    """Build paths the same way LRHRDataset does (metadata first, else folder)."""
    try:
        import data.util as Util
    except ImportError:
        Util = None

    if Util is not None:
        paths = Util.paired_paths_from_metadata(
            metadata_csv=dataset_opt.get('metadata_csv'),
            metadata_jsonl=dataset_opt.get('metadata_jsonl'),
            finetune_root=dataset_opt.get('finetune_root'),
            split=phase)
        if paths is not None:
            return paths

    datarootlq = dataset_opt.get('datarootlq')
    dataroothq = dataset_opt.get('dataroothq')
    if not datarootlq or not dataroothq:
        raise ValueError('datarootlq and dataroothq are required when metadata is absent')
    return paired_paths_from_folder(datarootlq, dataroothq)


def save_json(path: str, payload: dict) -> None:
    out_dir = osp.dirname(osp.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str, records: list[dict]) -> None:
    out_dir = osp.dirname(osp.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = ['index', 'filename', 'fog_level']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rebuild dataloader path table and export test_metadata.')
    parser.add_argument(
        '-c', '--config',
        default=osp.join(
            ROOT, 'config/test_ColdFog_finetune_netH_physical_ddim20.json'),
        help='JSON config (reads datasets[phase] paths).')
    parser.add_argument(
        '--phase', default='val', choices=['train', 'val'],
        help='Dataset split key in config. Default val matches infer.py and '
             'config datarootlq/dataroothq at lines 35-37 (hazy_test/gt_test).')
    parser.add_argument(
        '-o', '--output',
        default=osp.join(ROOT, 'plot/test_metadata.json'),
        help='Output JSON path (CSV written alongside with .csv suffix).')
    parser.add_argument(
        '--dataset-name',
        default=DEFAULT_DATASET_NAME,
        help='Dataset display name in metadata (default: Cold Storage Dehazing Dataset).')
    args = parser.parse_args()

    config_path = osp.abspath(args.config)
    dataset_opt = load_dataset_opt(config_path, args.phase)
    paths = build_paths(dataset_opt, args.phase)

    data_len = dataset_opt.get('len', -1)
    if data_len is not None and int(data_len) > 0:
        paths = paths[:int(data_len)]

    records = paths_to_metadata(paths)

    payload = {
        'config': config_path,
        'phase': args.phase,
        'dataset_name': args.dataset_name,
        'datarootlq': dataset_opt.get('datarootlq'),
        'dataroothq': dataset_opt.get('dataroothq'),
        'count': len(records),
        'samples': records,
    }

    save_json(args.output, payload)
    csv_path = osp.splitext(args.output)[0] + '.csv'
    # save_csv(csv_path, records)

    print('Saved {} samples to {}'.format(len(records), args.output))
    print('CSV: {}'.format(csv_path))
    if records:
        print('First 3 (dataloader order):')
        for row in records[:3]:
            print('  [{index}] {filename} ({fog_level})'.format(**row))


if __name__ == '__main__':
    main()
