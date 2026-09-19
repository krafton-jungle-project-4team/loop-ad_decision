"""Strict case/JUnit validation and fixed-versus-latest policy, without Docker access."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name('manifest.json').read_text())
REQUIRED = {name for names in MANIFEST.values() for name in names}
SHA = re.compile(r'^[0-9a-f]{40}$')


def classify(collected, reports, exit_code, required=REQUIRED):
    missing = sorted(required - set(collected))
    unexpected = sorted(set(collected) - required)
    incomplete = list(missing) + unexpected
    failures = []
    for name in sorted(required & set(collected)):
        phases = reports.get(name, {})
        if any(p.get('wasxfail') or p.get('outcome') == 'skipped' for p in phases.values()):
            incomplete.append(name)
        elif phases.get('setup', {}).get('outcome') == 'failed' or phases.get('teardown', {}).get('outcome') == 'failed':
            incomplete.append(name)
        elif phases.get('call', {}).get('outcome') == 'failed':
            failures.append(name)
        elif set(phases) != {'setup','call','teardown'} or any(p['outcome'] != 'passed' for p in phases.values()):
            incomplete.append(name)
    if len(collected) != len(set(collected)):
        incomplete.append('duplicate collection')
    if exit_code not in (0,1):
        incomplete.append('pytest exit ' + str(exit_code))
    if exit_code == 1 and not failures:
        incomplete.append('unaccounted pytest failure')
    return dict(status='INCOMPLETE' if incomplete else 'FAIL' if failures else 'PASS',
                missing=missing, incomplete=sorted(set(incomplete)), failures=failures)


def validate_lane(directory, required=REQUIRED, *, expected_sha=None, expected_ddl=None):
    directory = Path(directory)
    try:
        data = json.loads((directory/'result.json').read_text())
        assert SHA.fullmatch(data['contract_sha'])
        assert re.fullmatch(r'[0-9a-f]{64}', data['ddl_sha256'])
        if expected_sha is not None:
            assert data['contract_sha'] == expected_sha, 'lane SHA differs from resolved input'
        if expected_ddl is not None:
            assert data['ddl_sha256'] == expected_ddl, 'lane DDL hash differs from acquired input'
        judged = classify(data['collected'], data['reports'], data['pytest_exit'], required)
        assert judged['status'] == data['status'], 'claimed verdict differs from case reports'
        root = ET.parse(directory/'junit.xml').getroot()
        cases = root.findall('.//testcase')
        assert sorted(c.attrib['name'] for c in cases) == sorted(data['collected']), 'JUnit collection differs'
        for case in cases:
            name = case.attrib['name']
            phases = data['reports'][name]
            has_bad = any(p['outcome'] != 'passed' or p.get('wasxfail') for p in phases.values())
            junit_bad = any(case.find(tag) is not None for tag in ('failure','error','skipped'))
            assert has_bad == junit_bad, 'JUnit outcome differs'
        return data | judged
    except (OSError, ValueError, KeyError, TypeError, AssertionError, ET.ParseError) as exc:
        return dict(status='INCOMPLETE', stage='result-validation', reason=str(exc), collected=[])


def aggregate(fixed, latest, *, controls_ok, cleanup_ok, artifacts_ok=True):
    status = fixed['status'] if controls_ok and cleanup_ok and artifacts_ok else 'INCOMPLETE'
    latest_status = {'PASS':'PASS','FAIL':'WARN_DRIFT','INCOMPLETE':'WARN_UNVERIFIED'}[latest['status']]
    return dict(status=status, exit_code={'PASS':0,'FAIL':1,'INCOMPLETE':2}[status],
                latest_status=latest_status, controls_ok=controls_ok, cleanup_ok=cleanup_ok,
                artifacts_ok=artifacts_ok)
