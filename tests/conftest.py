from __future__ import annotations

import pytest
from psycopg.conninfo import conninfo_to_dict


_LOCAL_POSTGRES_HOSTS = {"localhost", "127.0.0.1", "::1"}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("loopad")
    group.addoption(
        "--loopad-test-postgres-dsn",
        action="store",
        default=None,
        dest="loopad_test_postgres_dsn",
        metavar="LOCAL_DSN",
        help=(
            "run opt-in PostgreSQL locking tests against a disposable local "
            "database"
        ),
    )


@pytest.fixture
def loopad_test_postgres_dsn(pytestconfig: pytest.Config) -> str:
    value = pytestconfig.getoption("loopad_test_postgres_dsn")
    if not isinstance(value, str) or not value.strip():
        pytest.skip(
            "pass --loopad-test-postgres-dsn to run PostgreSQL locking tests"
        )

    dsn = value.strip()
    try:
        connection_parameters = conninfo_to_dict(dsn)
    except Exception:
        raise pytest.UsageError(
            "--loopad-test-postgres-dsn must be a valid PostgreSQL DSN"
        ) from None

    if connection_parameters.get("service"):
        raise pytest.UsageError(
            "--loopad-test-postgres-dsn must not use a libpq service"
        )
    for key in ("host", "hostaddr"):
        addresses = connection_parameters.get(key)
        if addresses and not _all_postgres_addresses_are_local(addresses, key=key):
            raise pytest.UsageError(
                "--loopad-test-postgres-dsn must target localhost or a local "
                "Unix socket"
            )
    return dsn


def _all_postgres_addresses_are_local(addresses: str, *, key: str) -> bool:
    for address in addresses.split(","):
        candidate = address.strip()
        if not candidate:
            continue
        if key == "host" and candidate.startswith("/"):
            continue
        if candidate not in _LOCAL_POSTGRES_HOSTS:
            return False
    return True
