"""Export/restore only the non-personal synthetic fixture, with constraints enabled."""
import hashlib
import json
from pathlib import Path
from psycopg import sql
from psycopg.types.json import Jsonb

TABLES = ('projects', 'campaigns', 'promotions', 'promotion_analyses', 'segment_definitions',
          'user_behavior_vector_search_generations', 'segment_vectors',
          'segment_audience_allocation_plans', 'segment_audience_snapshots', 'segment_audience_members',
          'promotion_target_segments', 'promotion_audience_exclusion_state',
          'promotion_audience_exclusion_members', 'generation_runs', 'content_candidates',
          'promotion_runs', 'ad_experiments', 'promotion_run_target_bindings')


def export_rows(connection):
    result = {}
    for table in TABLES:
        rows = [row[0] for row in connection.execute(sql.SQL('SELECT to_jsonb(t) FROM {} t').format(sql.Identifier(table)))]
        # Canonical DDL already owns its system fallback row; this fixture never creates it.
        if table == 'segment_definitions':
            rows = [row for row in rows if row['segment_id'] != 'seg_existing_all']
        rows.sort(key=lambda row: (row.get('snapshot_kind') == 'final', json.dumps(row, sort_keys=True)))
        result[table] = rows
    return result


def insert_rows(connection, rows):
    assert set(rows) == set(TABLES), 'fixture table manifest mismatch'
    for table in TABLES:
        columns = {row[0] for row in connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s", (table,))}
        for row in rows[table]:
            assert set(row) <= columns, f'fixture columns removed from {table}: {set(row) - columns}'
            names = sql.SQL(',').join(map(sql.Identifier, row))
            connection.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM jsonb_populate_record(NULL::{}, %s)').format(
                sql.Identifier(table), names, names, sql.Identifier(table)), (Jsonb(row),))
    connection.commit()


def read_fixture(directory):
    directory = Path(directory)
    provenance = json.loads((directory / 'provenance.json').read_text())
    assert provenance['producer_sha'] == 'e1de8b29b902b54df3a58f21f1daa27c1171fe80'
    assert provenance['contract_sha'] == '0ec2cef0290f4659ad21ccc1dd2a20df2801ff50'
    assert provenance['synthetic_only'] is True and provenance['baseline_restore_verified'] is True
    for name in ('rows.json', 'expected.json'):
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == provenance['files'][name]
    return json.loads((directory / 'rows.json').read_text()), json.loads((directory / 'expected.json').read_text())


def restore_rows(connection, fixture):
    # Lifecycle INSERT triggers require finalized/reserved initial states. Replay the
    # stored baseline before/after images through valid SQL transitions, never a writer.
    insert_rows(connection, fixture['initial'])
    for table in TABLES:
        if fixture['initial'][table] == fixture['committed'][table]:
            continue
        keys = [row[0] for row in connection.execute("""
            SELECT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey)
            WHERE i.indrelid=%s::regclass AND i.indisprimary ORDER BY a.attnum
        """, (table,))]
        assert keys, table
        before = {tuple(row[key] for key in keys): row for row in fixture['initial'][table]}
        for row in fixture['committed'][table]:
            old = before.get(tuple(row[key] for key in keys))
            names = sql.SQL(',').join(map(sql.Identifier, row))
            if old is None:
                connection.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM jsonb_populate_record(NULL::{}, %s)').format(
                    sql.Identifier(table), names, names, sql.Identifier(table)), (Jsonb(row),))
            elif old != row:
                assignment = sql.SQL(',').join(sql.SQL('{}=r.{}').format(sql.Identifier(k),sql.Identifier(k)) for k in row)
                predicate = sql.SQL(' AND ').join(sql.SQL('t.{}=r.{}').format(sql.Identifier(k),sql.Identifier(k)) for k in keys)
                connection.execute(sql.SQL('UPDATE {} t SET {} FROM jsonb_populate_record(NULL::{}, %s) r WHERE {}').format(
                    sql.Identifier(table), assignment, sql.Identifier(table), predicate), (Jsonb(row),))
    connection.commit()
    assert export_rows(connection) == fixture['committed'], 'restored rows differ from immutable baseline output'
