#!/usr/bin/env python3
"""Parse DehazeDDPM-style train.log and plot training / validation curves.

Saves figure next to the log file: <log_stem>_curves.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

# English labels, Times New Roman (user preference)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "Nimbus Roman"]
plt.rcParams["axes.unicode_minus"] = False

_LINE_L_PIX = re.compile(
    r"<epoch:\s*(\d+),\s*iter:\s*([\d,\s]+)>\s+l_pix:\s*([\deE.+-]+)"
)
_LINE_PSNR = re.compile(
    r"<epoch:\s*(\d+),\s*iter:\s*([\d,\s]+)>\s+psnr:\s*([\deE.+-]+)"
)


def _parse_iter(s: str) -> int:
    return int(re.sub(r"[\s,]", "", s))


def parse_train_log(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    iters_l: list[int] = []
    l_pix: list[float] = []
    iters_v: list[int] = []
    psnr: list[float] = []
    for line in text.splitlines():
        m = _LINE_L_PIX.search(line)
        if m:
            iters_l.append(_parse_iter(m.group(2)))
            l_pix.append(float(m.group(3)))
            continue
        m = _LINE_PSNR.search(line)
        if m:
            iters_v.append(_parse_iter(m.group(2)))
            psnr.append(float(m.group(3)))
    return iters_l, l_pix, iters_v, psnr


def plot_curves(
    log_path: Path,
    iters_l: list[int],
    l_pix: list[float],
    iters_v: list[int],
    psnr: list[float],
) -> Path:
    out_path = log_path.parent / f"{log_path.stem}_curves.png"

    n_plots = 0
    if iters_l:
        n_plots += 1
    if iters_v:
        n_plots += 1
    if n_plots == 0:
        raise SystemExit(f"No plottable metrics found in {log_path}")

    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3.5 * n_plots), squeeze=False)
    ax_list = axes.flatten().tolist()
    i_ax = 0

    if iters_l:
        ax = ax_list[i_ax]
        i_ax += 1
        ax.plot(iters_l, l_pix, color="#1f77b4", linewidth=1.0, alpha=0.9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("l_pix")
        ax.set_title(f"Training loss ({log_path.name})")
        ax.grid(True, linestyle=":", alpha=0.6)

    if iters_v:
        ax = ax_list[i_ax]
        ax.plot(iters_v, psnr, color="#2ca02c", marker="o", markersize=3, linewidth=1.0)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR")
        ax.set_title("Validation PSNR")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot DehazeDDPM train.log curves.")
    p.add_argument(
        "log_file",
        type=Path,
        help="Path to train.log (or any log with the same <epoch, iter> l_pix / psnr lines)",
    )
    args = p.parse_args()
    log_path = args.log_file.expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(f"File not found: {log_path}")

    iters_l, l_pix, iters_v, psnr = parse_train_log(log_path)
    out = plot_curves(log_path, iters_l, l_pix, iters_v, psnr)
    print(f"Saved: {out}")
    print(
        f"Parsed {len(iters_l)} l_pix points, {len(iters_v)} validation PSNR points."
    )


if __name__ == "__main__":
    main()
