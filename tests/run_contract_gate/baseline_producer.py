"""Manual producer; invoked only with immutable baseline source, never normal Gate."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from .baseline import export_rows, restore_rows
from .seed import REQUEST
from .test_run_db import fresh_database, post, observe


def test_produce_and_verify():
    output = Path(os.environ['RCG_OUTPUT'])
    with fresh_database() as database:
        with psycopg.connect(**database) as connection:
            initial = export_rows(connection)
        response = post(database, REQUEST)
        assert response.status_code == 200, response.text
        with psycopg.connect(**database) as connection:
            rows = export_rows(connection)
            # No silently omitted application tables from this dedicated synthetic DB.
            for (table,) in connection.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'"):
                from psycopg import sql
                count = connection.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                assert count == 0 or table in rows, table
    with fresh_database(with_seed=False) as database:
        with psycopg.connect(**database) as connection:
            restore_rows(connection, {'initial': initial, 'committed': rows})
        before = observe(database)
        restored = post(database, REQUEST)
        assert restored.status_code == 200, restored.text
        assert restored.json() == response.json()
        assert observe(database) == before
    for name, value in [('rows.json', {'initial': initial, 'committed': rows}), ('expected.json', {'request': REQUEST, 'response': response.json()})]:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    provenance = dict(producer_sha='e1de8b29b902b54df3a58f21f1daa27c1171fe80',
        contract_sha='0ec2cef0290f4659ad21ccc1dd2a20df2801ff50', synthetic_only=True,
        baseline_restore_verified=True, generated_at=datetime.now(timezone.utc).isoformat(),
        procedure='Immutable git archive; synthetic seed; actual POST /runs; commit; export before/after images; fresh DDL restore using valid lifecycle SQL transitions; same baseline API reuse.',
        files={name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('rows.json','expected.json')},
        runner_image=os.environ['RCG_RUNNER_IMAGE'], architecture=os.environ['RCG_ARCHITECTURE'],
        python_image='python:3.12-slim@sha256:b699c2a51f4f834fa1a5f7f7cba0356e71d82fae37d4628223b11d93b71e9ebe',
        postgres_image='pgvector/pgvector:0.8.0-pg16@sha256:a132765ec351c65111b5b675928a3a0515a466a40f97277329db8b8209ad8bc9',
        lock_sha256=hashlib.sha256(Path('/requirements.lock').read_bytes()).hexdigest(),
        seed_sha256=hashlib.sha256(Path(__file__).with_name('seed.py').read_bytes()).hexdigest(),
        procedure_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ddl_sha256=hashlib.sha256(Path(os.environ['RCG_SCHEMA']).read_bytes()).hexdigest())
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
