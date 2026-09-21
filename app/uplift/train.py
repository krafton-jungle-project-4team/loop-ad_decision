from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from typing import Sequence

from app.config import load_settings
from app.db import create_clickhouse_client, create_postgres_connection
from app.decision.repositories import PsycopgPostgresExecutor
from app.uplift.dataset import (
    ClickHouseOutcomeEventRepository,
    PostgresUpliftUnitSourceRepository,
    UpliftDatasetBuilder,
)
from app.uplift.registry import (
    UpliftModelLifecycleService,
    UpliftModelRegistryRepository,
)
from app.uplift.training import UpliftOneShotTrainingService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one immutable LoopAd Uplift dataset."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--reference-time", required=True, type=_reference_time)
    args = parser.parse_args(argv)

    settings = load_settings()
    connection = create_postgres_connection(settings)
    clickhouse_client = create_clickhouse_client(settings)
    try:
        postgres = PsycopgPostgresExecutor(connection)
        result = UpliftOneShotTrainingService(
            dataset_builder=UpliftDatasetBuilder(
                unit_reader=PostgresUpliftUnitSourceRepository(postgres),
                outcome_reader=ClickHouseOutcomeEventRepository(
                    clickhouse_client
                ),
            ),
            lifecycle_service=UpliftModelLifecycleService(
                UpliftModelRegistryRepository(postgres)
            ),
        ).run(
            project_id=args.project_id.strip(),
            reference_time=args.reference_time,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        close = getattr(clickhouse_client, "close", None)
        if callable(close):
            close()
    print(json.dumps(result.to_json(), ensure_ascii=False, sort_keys=True))
    return 0


def _reference_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "reference time must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "reference time must include a timezone"
        )
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
