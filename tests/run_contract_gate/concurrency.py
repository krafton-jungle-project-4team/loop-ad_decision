"""Bounded, test-only scheduling around real requests, SQL and transactions."""
from contextvars import ContextVar
import json
import os
from pathlib import Path
import threading
import time

import psycopg
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from app.decision import router
from app.decision.audience_snapshots import AudienceSnapshotRepository
from app.decision.repositories import PromotionRunRepository
from app.decision.service import RunConflictError
from .seed import PROMOTION, REQUEST, SEGMENTS
from .responses import capture
from tools.run_contract_gate.contention import verify_contention, join_workers

REQUEST_ID = ContextVar('rcg_request_id')
WAIT_SECONDS = 12
PATH = '/decision/v1/promotions/' + PROMOTION + '/runs'


def run_contention(database, monkeypatch, mode, make_app, observe):
    """Timeouts are preparation errors; returned responses/rows are asserted by tests."""
    case = {'commit': 'RCG-10', 'rollback': 'RCG-11', 'overlap': 'RCG-12', 'overlap-reverse': 'RCG-12-reverse'}[mode]
    timeline, responses, errors, connections = [], {}, {}, {}
    mutex = threading.Lock()
    both_absent = threading.Barrier(2, timeout=WAIT_SECONDS)
    written, release = threading.Event(), threading.Event()
    result = dict(case=case, mode=mode, timeline=timeline, responses=responses,
                  errors=errors, limits=dict(event_seconds=WAIT_SECONDS,
                  statement_timeout_ms=20000, lock_timeout_ms=18000), before=observe(database))

    def record(request_id, event, **details):
        with mutex:
            timeline.append(dict(sequence=len(timeline), monotonic_ns=time.monotonic_ns(),
                                 request_id=request_id, event=event, **details))

    def wait(event, label):
        if not event.wait(WAIT_SECONDS):
            record('scheduler', 'preparation_error', stage=label)
            raise TimeoutError('contention preparation timed out: ' + label)

    class Connection:
        def __init__(self, actual, request_id):
            self.actual, self.request_id = actual, request_id
        def __getattr__(self, name):
            return getattr(self.actual, name)
        def commit(self):
            record(self.request_id, 'commit_begin')
            self.actual.commit()
            record(self.request_id, 'commit_complete')
        def rollback(self):
            record(self.request_id, 'rollback_begin')
            self.actual.rollback()
            record(self.request_id, 'rollback_complete')
        def close(self):
            self.actual.close()
            record(self.request_id, 'connection_closed')

    original_factory = router.create_postgres_connection
    def factory(settings):
        request_id = REQUEST_ID.get()
        actual = original_factory(settings)
        actual.execute("SET statement_timeout = '20s'")
        actual.execute("SET lock_timeout = '18s'")
        isolation = actual.execute('SHOW transaction_isolation').fetchone()[0]
        proxy = Connection(actual, request_id)
        connections[request_id] = proxy
        record(request_id, 'connection_open', pid=actual.info.backend_pid, isolation=isolation)
        return proxy

    original_get = PromotionRunRepository.get_by_scope
    counts = {}
    def get_scope(self, **kwargs):
        request_id = self._db._connection.request_id
        value = original_get(self, **kwargs)
        counts[request_id] = counts.get(request_id, 0) + 1
        record(request_id, 'scope_read', found=value is not None, number=counts[request_id])
        if counts[request_id] == 1:
            if value is not None:
                raise RuntimeError('fresh scope unexpectedly exists')
            try:
                both_absent.wait()
            except threading.BrokenBarrierError:
                record(request_id, 'preparation_error', stage='both scopes absent')
                raise
        return value

    original_insert = PromotionRunRepository.insert_if_absent
    def insert(self, run):
        request_id = self._db._connection.request_id
        if request_id == 'B':
            wait(written, 'A actual binding writes')
        record(request_id, 'insert_begin', run_id=run.promotion_run_id)
        inserted = original_insert(self, run)
        record(request_id, 'insert_complete', inserted=inserted)
        return inserted

    original_load_target = AudienceSnapshotRepository._load_binding_target
    def load_target(self, **kwargs):
        row = original_load_target(self, **kwargs)
        record(self._db._connection.request_id, 'binding_target_read',
               segment_id=kwargs['binding'].segment_id, row=dict(row))
        return row

    original_bind = AudienceSnapshotRepository.bind_run_targets
    def bind(self, **kwargs):
        request_id = self._db._connection.request_id
        record(request_id, 'bind_begin')
        original_bind(self, **kwargs)
        # Read actual uncommitted writes through the same transaction.
        connection = self._db._connection
        with connection.cursor(row_factory=dict_row) as cursor:
            pending = {}
            for table in ('promotion_runs', 'ad_experiments', 'promotion_run_target_bindings',
                          'promotion_target_segments', 'segment_audience_allocation_plans',
                          'promotion_audience_exclusion_members', 'promotion_audience_exclusion_state'):
                from psycopg import sql
                cursor.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table)))
                pending[table] = cursor.fetchall()
        record(request_id, 'bind_complete', pending_rows=pending)
        if request_id == 'A':
            written.set()
            wait(release, 'observer release')
            if mode == 'rollback':
                record(request_id, 'injected_conflict_after_real_writes')
                raise RunConflictError('rcg_test_rollback_after_real_writes')

    monkeypatch.setattr(router, 'create_postgres_connection', factory)
    monkeypatch.setattr(PromotionRunRepository, 'get_by_scope', get_scope)
    monkeypatch.setattr(PromotionRunRepository, 'insert_if_absent', insert)
    monkeypatch.setattr(AudienceSnapshotRepository, 'bind_run_targets', bind)
    monkeypatch.setattr(AudienceSnapshotRepository, '_load_binding_target', load_target)

    def request(request_id):
        app = make_app(database)
        async def identified(scope, receive, send):
            token = REQUEST_ID.set(request_id)
            async def record_send(message):
                if message['type'] == 'http.response.start':
                    record(request_id, 'http_response_start', status=message['status'])
                await send(message)
            try:
                await app(scope, receive, record_send)
            except Exception as exc:
                record(request_id, 'asgi_exception', exception=type(exc).__name__,
                       message=str(exc), sqlstate=getattr(exc, 'sqlstate', None))
                raise
            finally:
                REQUEST_ID.reset(token)
        narrow = (mode == 'overlap' and request_id == 'A') or (mode == 'overlap-reverse' and request_id == 'B')
        payload = REQUEST | {'segment_ids': [SEGMENTS[0]]} if narrow else REQUEST
        try:
            with TestClient(identified, raise_server_exceptions=False) as client:
                response = client.post(PATH, json=payload)
            metadata = capture(case, request_id, payload, response)
            responses[request_id] = metadata | {'body': response.text}
            record(request_id, 'request_complete', status=response.status_code)
        except Exception as exc:
            errors[request_id] = dict(exception=type(exc).__name__, message=str(exc))

    threads = [threading.Thread(target=request, args=(name,), name='rcg-'+name, daemon=True) for name in ('A','B')]
    try:
        for thread in threads:
            thread.start()
        wait(written, 'A actual binding writes')
        deadline = time.monotonic() + WAIT_SECONDS
        with psycopg.connect(**database, autocommit=True, row_factory=dict_row,
                             options='-c statement_timeout=2000') as observer:
            while time.monotonic() < deadline:
                pids = {key: connection.info.backend_pid for key, connection in connections.items()}
                if set(pids) == {'A','B'}:
                    if pids['A'] == pids['B']:
                        raise RuntimeError('requests shared a backend')
                    rows = observer.execute('SELECT pid, state, wait_event_type, wait_event, pg_blocking_pids(pid) AS blockers FROM pg_stat_activity WHERE pid = ANY(%s)', (list(pids.values()),)).fetchall()
                    blocked = next((row for row in rows if row['pid'] == pids['B'] and pids['A'] in row['blockers']), None)
                    if blocked:
                        result['pids'] = pids
                        record('observer', 'blocking_observed', rows=rows)
                        break
                threading.Event().wait(0.01)  # Poll actual DB state; never evidence by elapsed sleep.
            else:
                raise TimeoutError('actual backend blocking was not observed')
        record('observer', 'release_A')
    finally:
        release.set()
        result['workers_alive'] = join_workers(threads, connections.values(), WAIT_SECONDS,
                                               lambda names: record('observer', 'worker_timeout', workers=names))
        result['after'] = observe(database)
        Path(os.environ['RCG_OUTPUT'], case+'.json').write_text(json.dumps(result, default=str, indent=2)+'\n')
    if errors or result['workers_alive']:
        raise RuntimeError('request workers did not finish cleanly')
    verify_contention(result)
    return result
