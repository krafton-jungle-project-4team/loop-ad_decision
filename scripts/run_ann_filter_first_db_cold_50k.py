#!/usr/bin/env python3
"""Restart local PostgreSQL before each 50k filter-first cold sample."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PLANS = ("exact_all", "filter_first_exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--scenario-id",
        default="intent-all-months-confirmation-a",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.output.exists() and not args.append:
        raise ValueError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    append = args.append or args.output.exists()
    completed = {plan: 0 for plan in PLANS}
    for iteration in range(args.iterations):
        order = PLANS if iteration % 2 == 0 else tuple(reversed(PLANS))
        for plan in order:
            _restart_and_wait(args.container)
            command = [
                sys.executable,
                "scripts/benchmark_audience_search.py",
                "run-cell",
                "--env-file",
                str(args.env_file),
                "--manifest",
                str(args.manifest),
                "--scenario-id",
                args.scenario_id,
                "--phase",
                "confirmation",
                "--cache-mode",
                "db_cold",
                "--warmups",
                "0",
                "--repetitions",
                "1",
                "--sample-sizes",
                "50000",
                "--plan",
                plan,
                "--output",
                str(args.output),
                "--confirm-disposable-postgres",
            ]
            if append:
                command.append("--append")
            environment = dict(os.environ)
            environment["ANN_POSTGRES_CONTAINER"] = args.container
            subprocess.run(command, check=True, env=environment)
            append = True
            completed[plan] += 1
            print(
                json.dumps(
                    {
                        "iteration": iteration + 1,
                        "plan": plan,
                        "completed": completed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return 0


def _restart_and_wait(container: str) -> None:
    subprocess.run(
        ("docker", "restart", container),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        ready = subprocess.run(
            ("docker", "exec", container, "pg_isready", "-q"),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ready.returncode == 0:
            return
        time.sleep(0.25)
    raise TimeoutError("PostgreSQL did not become ready within 60 seconds")


if __name__ == "__main__":
    raise SystemExit(main())
