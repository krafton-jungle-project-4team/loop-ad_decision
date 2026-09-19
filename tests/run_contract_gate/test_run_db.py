"""Real API/SQL/commit contract scenarios; isolated synthetic PostgreSQL only."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.rows import dict_row

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.audience_snapshots import AudienceSnapshotRepository
from app.main import create_app
from .responses import capture
from .seed import ANALYSIS, PROMOTION, REQUEST, SEGMENTS, seed

pytestmark = pytest.mark.skipif(
    'RCG_SCHEMA' not in os.environ,
    reason='run scripts/run-contract-gate.sh in isolated test containers',
)


@contextmanager
def fresh_database(with_seed=True):
    # Deliberately no configurable DSN: only this container's shared Unix socket.
    name = 'rcg_' + uuid.uuid4().hex
    params = dict(host='/var/run/postgresql', user='postgres', dbname=name)
    with psycopg.connect(**(params | {'dbname': 'postgres'}), autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    try:
        with psycopg.connect(**params) as connection:
            connection.execute(Path(os.environ['RCG_SCHEMA']).read_text())
            connection.commit()
            if with_seed:
                seed(connection)
        yield params
    finally:
        with psycopg.connect(**(params | {'dbname': 'postgres'}), autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


@pytest.fixture
def database():
    with fresh_database() as params:
        yield params


def make_app(database):
    values = {key: 'synthetic-unused' for key in REQUIRED_ENV_NAMES}
    values.update(LOOPAD_ENV='test', LOOPAD_SERVICE_ID='decision-api', PORT='8080',
                  LOOPAD_AURORA_HOST=database['host'], LOOPAD_AURORA_PORT='5432',
                  LOOPAD_AURORA_DATABASE=database['dbname'], LOOPAD_AURORA_USERNAME='postgres')
    return create_app(settings=load_settings(values))


def observe(database):
    # Independent connection opened after TestClient request and dependency teardown.
    with psycopg.connect(**database, row_factory=dict_row) as connection:
        result = {}
        for table in ('promotion_runs', 'ad_experiments', 'promotion_run_target_bindings',
                      'promotion_target_segments', 'segment_audience_allocation_plans',
                      'promotion_audience_exclusion_members', 'promotion_audience_exclusion_state'):
            result[table] = sorted(connection.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table))).fetchall(),
                                   key=lambda row: json.dumps(row, default=str, sort_keys=True))
        return result


def evidence(case, value):
    Path(os.environ['RCG_OUTPUT'], case + '.json').write_text(json.dumps(value, default=str, indent=2) + '\n')


def test_create_commits_exact_scope(database):
    """RCG-01: actual API, actual commit, independent rows and lifecycle checks."""
    with TestClient(make_app(database), raise_server_exceptions=False) as client:
        response = client.post('/decision/v1/promotions/' + PROMOTION + '/runs', json=REQUEST)
    capture('RCG-01', 'create', REQUEST, response)
    stored = observe(database)
    evidence('RCG-01', {'http_status': response.status_code, 'response': response.json(), 'rows': stored})
    assert response.status_code == 200
    payload = response.json()
    scope = SEGMENTS[:2]
    assert payload['segment_ids'] == scope
    runs = stored['promotion_runs']
    assert len(runs) == 1
    run = runs[0]
    assert run['promotion_run_id'] == payload['promotion_run_id']
    assert run['segment_scope_json'] == scope
    assert run['segment_scope_fingerprint'] == hashlib.sha256(json.dumps(scope, separators=(',', ':')).encode()).hexdigest()
    experiments = stored['ad_experiments']
    assert len(experiments) == 2
    assert {row['segment_id'] for row in experiments} == set(scope)
    assert {row['ad_experiment_id'] for row in experiments} == {row['ad_experiment_id'] for row in payload['ad_experiments']}
    assert all(row['promotion_run_id'] == run['promotion_run_id'] for row in experiments)
    assert all(not row['is_fallback'] for row in payload['ad_experiments'])
    bindings = stored['promotion_run_target_bindings']
    assert len(bindings) == 2
    assert {row['segment_id'] for row in bindings} == set(scope)
    assert all(row['promotion_run_id'] == run['promotion_run_id'] and row['target_analysis_id'] == ANALYSIS for row in bindings)
    targets = {row['segment_id']: row for row in stored['promotion_target_segments']}
    assert all(row['final_snapshot_id'] == targets[row['segment_id']]['audience_snapshot_id']
               and row['allocation_plan_id'] == targets[row['segment_id']]['allocation_plan_id'] for row in bindings)
    assert {row['status'] for row in stored['segment_audience_allocation_plans']} == {'locked'}
    assert {row['segment_id']: row['audience_reservation_state'] for row in stored['promotion_target_segments']} == {
        segment: 'consumed' if segment in scope else 'reserved' for segment in SEGMENTS}
    exclusions = stored['promotion_audience_exclusion_members']
    assert len(exclusions) == 6
    assert all(row['state'] == ('consumed' if row['segment_id'] in scope else 'reserved') for row in exclusions)


def test_deferred_binding_failure_rolls_back(database, monkeypatch):
    """RCG-07: rollback alone must never turn an HTTP success into a passing case."""
    before = observe(database)
    events = []
    app = make_app(database)

    async def recording_app(scope, receive, send):
        async def record_send(message):
            if message['type'].startswith('http.response'):
                events.append({'event': message['type'], 'status': message.get('status'),
                               'more_body': message.get('more_body', False)})
            await send(message)
        try:
            await app(scope, receive, record_send)
        except Exception as exc:
            events.append({'event': 'exception', 'type': type(exc).__name__,
                           'sqlstate': getattr(exc, 'sqlstate', None), 'message': str(exc)})
            raise

    def omit_binding(*args, **kwargs):
        events.append({'event': 'binding_omitted_for_RCG_07'})

    # The only injected fault. Run and experiment SQL, commit and rollback stay real.
    monkeypatch.setattr(AudienceSnapshotRepository, 'bind_run_targets', omit_binding)
    with TestClient(recording_app, raise_server_exceptions=False) as client:
        response = client.post('/decision/v1/promotions/' + PROMOTION + '/runs', json=REQUEST)
    after = observe(database)
    evidence('RCG-07', {'http_status': response.status_code, 'response_text': response.text,
                        'asgi_events': events, 'before': before, 'after': after})
    assert after == before, 'all run writes and lifecycle changes must roll back'
    assert any(event.get('sqlstate') == '23514' and 'run binding set' in event.get('message', '') for event in events)
    assert response.status_code >= 400, 'commit failed, but the API already sent an HTTP success response'


def post(database, payload):
    with TestClient(make_app(database), raise_server_exceptions=False) as client:
        return client.post('/decision/v1/promotions/' + PROMOTION + '/runs', json=payload)


def test_retry_reuses_committed_run(database):
    first = post(database, REQUEST)
    capture('RCG-02', 'first', REQUEST, first)
    assert first.status_code == 200
    before = observe(database)
    # A new client/request models retry after losing the already committed response.
    retry = post(database, REQUEST)
    capture('RCG-02', 'retry', REQUEST, retry)
    assert retry.status_code == 200
    assert retry.json() == first.json()
    assert observe(database) == before
    evidence('RCG-02', {'rows_unchanged': True, 'rows': before})


def test_scope_order_and_duplicates(database):
    first = post(database, REQUEST)
    assert first.status_code == 200
    before = observe(database)
    reordered = post(database, REQUEST | {'segment_ids': [SEGMENTS[1], SEGMENTS[0], SEGMENTS[0]]})
    assert reordered.status_code == 200
    assert reordered.json() == first.json()
    assert reordered.json()['segment_ids'] == SEGMENTS[:2]
    assert observe(database) == before


def test_disjoint_scopes_are_distinct(database):
    first = post(database, REQUEST | {'segment_ids': [SEGMENTS[0]]})
    assert first.status_code == 200
    second = post(database, REQUEST | {'segment_ids': [SEGMENTS[1]]})
    assert second.status_code == 200
    assert first.json()['promotion_run_id'] != second.json()['promotion_run_id']
    stored = observe(database)
    assert len(stored['promotion_runs']) == 2
    assert len(stored['ad_experiments']) == len(stored['promotion_run_target_bindings']) == 2
    for response, segment in ((first, SEGMENTS[0]), (second, SEGMENTS[1])):
        run_id = response.json()['promotion_run_id']
        assert response.json()['segment_ids'] == [segment]
        runs = [row for row in stored['promotion_runs'] if row['promotion_run_id'] == run_id]
        assert runs[0]['segment_scope_json'] == [segment]
        for table in ('ad_experiments', 'promotion_run_target_bindings'):
            assert [row['segment_id'] for row in stored[table] if row['promotion_run_id'] == run_id] == [segment]


@pytest.mark.parametrize('variant, expected_status, expected_detail', [
    ('missing-analysis', 409, 'segment_audience_run_source_required'),
    ('missing-generation', 409, 'segment_audience_run_source_required'),
    ('missing-segments', 409, 'segment_audience_run_source_required'),
    ('wrong-analysis-generation', 422, 'generation run must belong to the selected promotion analysis'),
    ('outside', 409, 'segment_audience_run_source_mismatch'),
    ('empty', 422, 'segment_ids must contain at least one segment'),
    ('blank', 422, 'segment_ids must not contain blank values'),
    ('fallback', 422, 'segment_ids must not include the fallback segment'),
], ids=['missing-analysis', 'missing-generation', 'missing-segments', 'wrong-analysis-generation',
        'outside', 'empty', 'blank', 'fallback'])
def test_rejects_invalid_source_without_writes(database, variant, expected_status, expected_detail):
    from .seed import SOURCE
    payload = dict(REQUEST)
    if variant.startswith('missing-'):
        payload.pop({'missing-analysis':'analysis_id', 'missing-generation':'generation_id',
                     'missing-segments':'segment_ids'}[variant])
    elif variant == 'wrong-analysis-generation':
        payload['analysis_id'] = SOURCE
    else:
        payload['segment_ids'] = {'outside':['rcg_outside'], 'empty':[], 'blank':[' '],
                                  'fallback':['seg_existing_all']}[variant]
    before = observe(database)
    response = post(database, payload)
    assert response.status_code == expected_status, response.text
    detail = response.json()['detail']
    assert (detail['code'] if isinstance(detail, dict) else detail) == expected_detail
    assert observe(database) == before


@pytest.mark.parametrize('stage', ['experiment-insert', 'target-binding'])
def test_failure_after_real_writes_rolls_back(database, monkeypatch, stage):
    from app.decision.repositories import AdExperimentRepository
    owner, method = ((AdExperimentRepository, 'insert_many') if stage == 'experiment-insert'
                     else (AudienceSnapshotRepository, 'bind_run_targets'))
    original = getattr(owner, method)
    reached = []

    def fail_after_real_write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        reached.append(stage)
        raise RuntimeError('synthetic failure after actual ' + stage)

    before = observe(database)
    monkeypatch.setattr(owner, method, fail_after_real_write)
    response = post(database, REQUEST)
    assert reached == [stage]
    assert response.status_code == 500
    assert observe(database) == before


def test_overlapping_target_binding_is_rejected(database):
    first = post(database, REQUEST | {'segment_ids': [SEGMENTS[0]]})
    assert first.status_code == 200
    before = observe(database)
    second = post(database, REQUEST)
    assert second.status_code == 409, second.text
    assert second.json()['detail']['code'] == 'segment_audience_target_already_run_bound'
    assert observe(database) == before


@pytest.fixture
def baseline_database():
    from .baseline import read_fixture, restore_rows
    directory = os.environ.get('RCG_BASELINE', '/gate/tests/fixtures/run_contract_gate/baseline')
    rows, expected = read_fixture(directory)
    with fresh_database(with_seed=False) as database:
        with psycopg.connect(**database) as connection:
            restore_rows(connection, rows)
        yield database, expected


def test_baseline_rows_are_reused(baseline_database):
    from .baseline import export_rows
    database, expected = baseline_database
    with psycopg.connect(**database) as connection:
        before = export_rows(connection)
    response = post(database, expected['request'])
    capture('RCG-08', 'reuse', expected['request'], response)
    assert response.status_code == 200, response.text
    assert response.json() == expected['response']
    with psycopg.connect(**database) as connection:
        assert export_rows(connection) == before
    evidence('RCG-08', {'http_status': response.status_code, 'response': response.json(),
                        'baseline_rows_unchanged': True})


@pytest.fixture(params=['commit', 'rollback', 'overlap', 'overlap-reverse'])
def contention(request, database, monkeypatch):
    from .concurrency import run_contention
    return run_contention(database, monkeypatch, request.param, make_app, observe)


def test_concurrent_requests_preserve_atomic_scope(contention):
    """RCG-10/11/12: setup proves blocking; call checks actual API/DB invariants."""
    result = contention
    responses, events, rows = result['responses'], result['timeline'], result['after']
    statuses = {name: value['status'] for name, value in responses.items()}
    if result['mode'] == 'commit':
        assert statuses == {'A': 200, 'B': 200}, responses
        assert json.loads(responses['A']['body']) == json.loads(responses['B']['body'])
        assert any(e['request_id'] == 'B' and e['event'] == 'insert_complete' and not e['inserted'] for e in events)
        assert any(e['request_id'] == 'B' and e['event'] == 'scope_read' and e['number'] == 2 and e['found'] for e in events)
    else:
        assert sorted(statuses.values()) == [200, 409], responses
        loser = next(name for name, status in statuses.items() if status == 409)
        detail = json.loads(responses[loser]['body'])['detail']
        if result['mode'] == 'rollback':
            assert loser == 'A'
            assert detail == 'rcg_test_rollback_after_real_writes'
        else:
            assert detail == dict(code='segment_audience_run_binding_invalid',
                segment_id=','.join(responses[loser]['request']['segment_ids']),
                reason='segment_audience_exclusion_binding_invalid: '+SEGMENTS[0]), detail
    winner = next(name for name, status in statuses.items() if status == 200)
    payload = json.loads(responses[winner]['body'])
    scope = payload['segment_ids']
    assert len(rows['promotion_runs']) == 1
    assert rows['promotion_runs'][0]['promotion_run_id'] == payload['promotion_run_id']
    assert rows['promotion_runs'][0]['segment_scope_json'] == scope
    assert len(rows['ad_experiments']) == len(rows['promotion_run_target_bindings']) == len(scope)
    assert {row['ad_experiment_id'] for row in rows['ad_experiments']} == {row['ad_experiment_id'] for row in payload['ad_experiments']}
    for table in ('ad_experiments', 'promotion_run_target_bindings'):
        assert {row['segment_id'] for row in rows[table]} == set(scope)
        assert all(row['promotion_run_id'] == payload['promotion_run_id'] for row in rows[table])
    assert {row['segment_id']: row['audience_reservation_state'] for row in rows['promotion_target_segments']} == {
        segment: 'consumed' if segment in scope else 'reserved' for segment in SEGMENTS}
    assert all(row['state'] == ('consumed' if row['segment_id'] in scope else 'reserved') for row in rows['promotion_audience_exclusion_members'])
    assert len(rows['promotion_audience_exclusion_members']) == 6
    assert {row['status'] for row in rows['segment_audience_allocation_plans']} == {'locked'}
    before_revision = result['before']['promotion_audience_exclusion_state'][0]['revision']
    assert rows['promotion_audience_exclusion_state'][0]['revision'] == before_revision + 1
    targets = {row['segment_id']: row for row in rows['promotion_target_segments']}
    for row in rows['promotion_run_target_bindings']:
        assert row['target_analysis_id'] == ANALYSIS
        assert row['final_snapshot_id'] == targets[row['segment_id']]['audience_snapshot_id']
        assert row['allocation_plan_id'] == targets[row['segment_id']]['allocation_plan_id']
    for name, status in statuses.items():
        own = [event for event in events if event['request_id'] == name]
        end = 'commit_complete' if status == 200 else 'rollback_complete'
        assert len([event for event in own if event['event'] == end]) == 1
        assert next(event['sequence'] for event in own if event['event'] == end) < next(event['sequence'] for event in own if event['event'] == 'http_response_start')
        assert not any(event['event'] == ('rollback_complete' if status == 200 else 'commit_complete') for event in own)
    assert result['pids']['A'] != result['pids']['B']
    assert not result['workers_alive']
