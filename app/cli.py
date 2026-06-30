from __future__ import annotations

import argparse
from datetime import date

from app.config import load_settings
from app.jobs.decision_job import DecisionRunRequest, build_daily_decision_job_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop-Ad Decision API jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-daily-decision")
    run_parser.add_argument("--project-key", required=True)
    run_parser.add_argument("--analysis-date", required=True, type=date.fromisoformat)
    run_parser.add_argument("--mode", choices=("normal", "demo", "backfill"), default="normal")
    run_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "run-daily-decision":
        settings = load_settings()
        service = build_daily_decision_job_service(settings)
        run = service.start_run(
            DecisionRunRequest(
                project_key=args.project_key,
                analysis_date=args.analysis_date,
                mode=args.mode,
                force=args.force,
                run_type="manual_cli",
                trigger_source="cli",
            )
        )
        service.execute_run(run.run_id)


if __name__ == "__main__":
    main()
