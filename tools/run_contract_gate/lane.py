"""Container-only lane execution; no network, Docker socket, env files, or fake repositories."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import tomllib
import pytest
from packaging.requirements import Requirement
from .report import classify


class Reports:
    def __init__(self):
        self.collected = []
        self.reports = {}

    def pytest_collection_finish(self, session):
        self.collected = [item.name for item in session.items]

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid.split('::')[-1], {})[report.when] = dict(
            outcome=report.outcome, wasxfail=hasattr(report, 'wasxfail'))


def main():
    output = Path(os.environ['RCG_OUTPUT'])
    start = time.monotonic()
    plugin = Reports()
    data = dict(schema_version=1, contract_sha=os.environ['RCG_CONTRACT_SHA'],
                ddl_sha256=hashlib.sha256(Path(os.environ['RCG_SCHEMA']).read_bytes()).hexdigest())
    controls = os.environ.get('RCG_CONTROLS') == '1'
    try:
        project = tomllib.loads(Path('/source/pyproject.toml').read_text())['project']
        for value in project['dependencies'] + project['optional-dependencies']['dev']:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate():
                assert importlib.metadata.version(requirement.name) in requirement.specifier, value
        code = int(pytest.main(['-q', '-p', 'no:cacheprovider', '--confcutdir=/gate/tests/run_contract_gate',
            '/gate/tests/run_contract_gate/test_runner_control.py' if controls else '/gate/tests/run_contract_gate/test_run_db.py', '--junitxml='+str(output/'junit.xml')], plugins=[plugin]))
        data.update(collected=plugin.collected, reports=plugin.reports, pytest_exit=code)
        from .finalize import control_names
        data.update(classify(plugin.collected, plugin.reports, code, control_names()) if controls else classify(plugin.collected, plugin.reports, code))
    except Exception as exc:
        data.update(status='INCOMPLETE', stage='preparation', reason=str(exc), collected=[], reports={}, pytest_exit=3)
    data['duration_seconds'] = round(time.monotonic()-start, 3)
    (output/'result.json').write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[data['status']]


if __name__ == '__main__':
    raise SystemExit(main())
