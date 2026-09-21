#!/usr/bin/env python3
"""Run each remaining Goal 1 cohort and checkpoint it before continuing."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-sizes",
        default="100000,250000,500000,750000,1000000",
    )
    args = parser.parse_args()
    sizes = tuple(
        int(value.strip())
        for value in args.cohort_sizes.split(",")
        if value.strip()
    )
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("cohort sizes must be positive")
    for size in sizes:
        subprocess.run(
            (
                sys.executable,
                str(ROOT / "scripts/run_ann_scale_goal1_measurements.py"),
                "measure",
                "--cohort-sizes",
                str(size),
            ),
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            (
                sys.executable,
                str(ROOT / "scripts/analyze_ann_scale_goal1.py"),
                "--cohort-sizes",
                str(size),
            ),
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
