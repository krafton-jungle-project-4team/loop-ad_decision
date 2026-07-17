#!/usr/bin/env python3
"""Add registered segment_audience.v1 bindings to explicit legacy segment IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.segment_audience_backfill import (  # noqa: E402
    plan_segment_audience_backfill,
)
from app.config import load_settings  # noqa: E402
from app.db import create_postgres_connection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or apply registered V2 audience bindings to explicitly "
            "listed AI segment definitions."
        )
    )
    parser.add_argument(
        "--segment-id",
        action="append",
        required=True,
        dest="segment_ids",
        help="Segment ID to validate; repeat for every intended row.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist all validated rows in one transaction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    segment_ids = tuple(dict.fromkeys(args.segment_ids))
    if len(segment_ids) != len(args.segment_ids):
        raise SystemExit("duplicate --segment-id values are not allowed")
    connection = create_postgres_connection(load_settings())
    try:
        lock_clause = "FOR UPDATE" if args.apply else ""
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""
                SELECT segment_id, source, rule_json, profile_json
                FROM segment_definitions
                WHERE segment_id = ANY(%s)
                ORDER BY segment_id ASC
                {lock_clause}
                """,
                (list(segment_ids),),
            )
            plans = plan_segment_audience_backfill(
                list(cursor.fetchall()),
                requested_segment_ids=segment_ids,
            )
            if args.apply:
                for plan in plans:
                    if not plan.changed:
                        continue
                    cursor.execute(
                        """
                        UPDATE segment_definitions
                        SET rule_json = %s, updated_at = now()
                        WHERE segment_id = %s
                        """,
                        (Jsonb(plan.rule_json), plan.segment_id),
                    )
        summary = {
            "mode": "apply" if args.apply else "dry-run",
            "validated_segment_ids": [plan.segment_id for plan in plans],
            "changed_segment_ids": [
                plan.segment_id for plan in plans if plan.changed
            ],
        }
        if args.apply:
            connection.commit()
        else:
            connection.rollback()
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
