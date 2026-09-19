"""Adversarial controls: actual pytest reports and the actual host cleanup/timeout functions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest
from tools.run_contract_gate.report import aggregate, classify, validate_lane


@pytest.mark.parametrize('mode', ['missing', 'skip', 'xfail', 'xpass', 'pass'])
def test_required_execution(mode, tmp_path):
    """CTRL-01: real pytest, including xpass whose raw pytest exit is zero."""
    text = 'import pytest\ndef test_a(): pass\n'
    if mode != 'missing':
        text += {'skip':'@pytest.mark.skip(reason="synthetic")\n',
                 'xfail':'@pytest.mark.xfail(reason="synthetic")\n',
                 'xpass':'@pytest.mark.xfail(reason="synthetic", strict=False)\n'}.get(mode, '')
        text += 'def test_b(): ' + ('assert False' if mode == 'xfail' else 'pass') + '\n'
    (tmp_path/'test_case.py').write_text(text)
    code = """
import json, pytest, sys
from tools.run_contract_gate.lane import Reports
from tools.run_contract_gate.report import classify
p=Reports()
rc=int(pytest.main(['-q','-p','no:cacheprovider',sys.argv[1]],plugins=[p]))
print('VERDICT='+json.dumps(classify(p.collected,p.reports,rc,{'test_a','test_b'})))
"""
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path/'test_case.py')],
                            text=True, capture_output=True, timeout=30, check=True)
    verdict = json.loads(next(line.removeprefix('VERDICT=') for line in result.stdout.splitlines() if line.startswith('VERDICT=')))
    assert verdict['status'] == ('PASS' if mode == 'pass' else 'INCOMPLETE')


@pytest.mark.parametrize('fixed', ['FAIL','INCOMPLETE'])
def test_latest_cannot_mask_fixed(fixed):
    """CTRL-02."""
    result = aggregate({'status':fixed},{'status':'PASS'},controls_ok=True,cleanup_ok=True)
    assert result['status'] == fixed
    assert result['exit_code'] == (1 if fixed == 'FAIL' else 2)


@pytest.mark.parametrize('latest', ['FAIL','INCOMPLETE'])
def test_latest_warning_preserves_fixed(latest):
    """CTRL-03."""
    result = aggregate({'status':'PASS'},{'status':latest},controls_ok=True,cleanup_ok=True)
    assert result['exit_code'] == 0 and result['status'] == 'PASS'
    assert result['latest_status'] == ('WARN_DRIFT' if latest == 'FAIL' else 'WARN_UNVERIFIED')


@pytest.mark.parametrize('stage', ['schema', 'runner', 'timeout', 'interrupt', 'foreign-owner', 'inspect-error', 'stubborn-timeout'])
def test_cleanup_and_interruption(stage, tmp_path):
    """CTRL-04: shell helper exercised with an in-memory Docker double; no Docker socket."""
    helpers = Path(__file__).parents[2]/'tools/run_contract_gate/resources.sh'
    script = r'''
set -eu
source "$HELPERS"
RUN_ID=owned; RUNNER=owned-runner; DB=owned-db; SOCKET=owned-socket; LABEL=gate
ACTIVE_PID=
docker_double() {
    if [ "$1 $2" = 'container ls' ]; then echo container-id
    elif [ "$1 $2" = 'volume ls' ]; then echo owned-socket
    elif [ "$1" = inspect ] || [ "$1 $2" = 'volume inspect' ]; then
        [ "$STAGE" != inspect-error ] || return 1
        if [ "$STAGE" = foreign-owner ] && [ "${!#}" = owned-runner ]; then echo foreign; else echo owned; fi
    elif [ "$1" = rm ]; then echo "remove-container ${!#}" >> "$LOG"
    elif [ "$1 $2" = 'volume rm' ]; then echo "remove-volume ${!#}" >> "$LOG"
    else return 99; fi
}
D=(docker_double)
finish() {
    code=$?; trap - EXIT TERM INT
    if [ -n "$ACTIVE_PID" ]; then kill "$ACTIVE_PID" 2>/dev/null || true; wait "$ACTIVE_PID" 2>/dev/null || true; fi
    cleanup_resources || code=2
    echo "$code" > "$STATUS"
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
case "$STAGE" in
  schema|runner) exit 7;;
  stubborn-timeout) run_bounded 1 "$PYTHON" -c 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)';;
  timeout) run_bounded 1 "$PYTHON" -c 'import time; time.sleep(30)';;
  interrupt) touch "$READY"; run_bounded 30 "$PYTHON" -c 'import time; time.sleep(30)';;
  *) exit 7;;
esac
'''
    path = tmp_path/'probe.sh'; path.write_text(script)
    env = os.environ | {'HELPERS':str(helpers), 'LOG':str(tmp_path/'log'), 'STATUS':str(tmp_path/'status'),
                        'READY':str(tmp_path/'ready'), 'STAGE':stage, 'PYTHON':sys.executable}
    process = subprocess.Popen(['bash',str(path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if stage == 'interrupt':
        deadline = time.monotonic()+5
        while not (tmp_path/'ready').exists() and time.monotonic()<deadline:
            time.sleep(.02)
        assert (tmp_path/'ready').exists()
        process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode != 0, (stdout,stderr)
    assert (tmp_path/'status').exists(), stderr
    removed = (tmp_path/'log').read_text().splitlines() if (tmp_path/'log').exists() else []
    if stage == 'inspect-error':
        assert removed == []
    elif stage == 'foreign-owner':
        assert removed == ['remove-container owned-db','remove-volume owned-socket']
        assert process.returncode == 2
    else:
        assert removed == ['remove-container owned-runner','remove-container owned-db','remove-volume owned-socket']
    if stage in ('timeout','stubborn-timeout'):
        assert process.returncode == 124


def valid_artifacts(directory, names=None):
    names = ['test_a','test_b'] if names is None else sorted(names)
    reports = {name:{phase:{'outcome':'passed','wasxfail':False} for phase in ('setup','call','teardown')} for name in names}
    data = dict(contract_sha='a'*40,ddl_sha256='b'*64,collected=names,reports=reports,pytest_exit=0,status='PASS')
    (directory/'result.json').write_text(json.dumps(data))
    root = ET.Element('testsuites'); suite = ET.SubElement(root,'testsuite',tests=str(len(names)))
    for name in names: ET.SubElement(suite,'testcase',name=name)
    ET.ElementTree(root).write(directory/'junit.xml')
    return data


@pytest.mark.parametrize('damage', ['result-missing','result-corrupt','junit-missing','junit-corrupt',
                                    'sha-missing','cases-missing','junit-outcome','verdict-forged','sha-disagrees','ddl-disagrees','none'])
def test_result_integrity(damage, tmp_path):
    """CTRL-05: missing/corrupt/contradictory artifacts cannot yield green."""
    data = valid_artifacts(tmp_path)
    if damage.endswith('missing') and damage.split('-')[0] in ('result','junit'):
        (tmp_path/('result.json' if damage.startswith('result') else 'junit.xml')).unlink()
    elif damage.endswith('corrupt'):
        (tmp_path/('result.json' if damage.startswith('result') else 'junit.xml')).write_text('{broken')
    elif damage == 'sha-missing':
        del data['contract_sha']
    elif damage == 'sha-disagrees':
        data['contract_sha']='c'*40
    elif damage == 'ddl-disagrees':
        data['ddl_sha256']='d'*64
    elif damage == 'cases-missing':
        data['collected'].pop()
    elif damage == 'junit-outcome':
        (tmp_path/'junit.xml').write_text('<testsuites><testsuite><testcase name="test_a"><failure/></testcase><testcase name="test_b"/></testsuite></testsuites>')
    elif damage == 'verdict-forged':
        data['reports']['test_b']['call']['outcome']='failed'
    if damage in ('sha-missing','cases-missing','verdict-forged','sha-disagrees','ddl-disagrees'):
        (tmp_path/'result.json').write_text(json.dumps(data))
    verdict = validate_lane(tmp_path, {'test_a','test_b'}, expected_sha='a'*40, expected_ddl='b'*64)
    assert verdict['status'] == ('PASS' if damage == 'none' else 'INCOMPLETE')


@pytest.mark.parametrize('fault', ['cleanup','controls','artifacts'])
def test_common_failure_overrides_pass(fault):
    options = dict(cleanup_ok=True,controls_ok=True,artifacts_ok=True)
    options[fault+'_ok']=False
    assert aggregate({'status':'PASS'},{'status':'PASS'},**options)['exit_code'] == 2


def synthetic_bundle(directory):
    """Control-only fake responses exercise integrity, never claim API coverage."""
    from tools.run_contract_gate.bundle import SAMPLES, EVIDENCE, build_bundle
    from tools.run_contract_gate.report import REQUIRED
    import hashlib
    names = sorted(REQUIRED)
    data = dict(contract_sha='a'*40, ddl_sha256='b'*64, collected=names, pytest_exit=0, status='PASS',
        reports={name:{phase:dict(outcome='passed', wasxfail=False) for phase in ('setup','call','teardown')} for name in names})
    (directory/'result.json').write_text(json.dumps(data))
    root = ET.Element('testsuites'); suite = ET.SubElement(root, 'testsuite', tests=str(len(names)))
    for name in names:
        ET.SubElement(suite, 'testcase', name=name)
    ET.ElementTree(root).write(directory/'junit.xml')
    for name in EVIDENCE:
        (directory/name).write_text('{"control_only":true}')
    (directory/'responses').mkdir()
    for stem, (case, request, _) in SAMPLES.items():
        body = b'{"control_only":true}'
        (directory/'responses'/(stem+'.body')).write_bytes(body)
        metadata = dict(case_id=case, request_id=request, method='POST',
            path='/decision/v1/promotions/rcg_synthetic_promotion/runs', request={'control_only':True},
            status=200, content_type='application/json', body_file=stem+'.body',
            body_sha256=hashlib.sha256(body).hexdigest())
        (directory/'responses'/(stem+'.json')).write_text(json.dumps(metadata))
    producer = dict(candidate_commit='c'*40, candidate_dirty=False, source_sha256='d'*64,
        baseline_producer_sha='e'*40, lock_sha256='f'*64, runner_image='synthetic-control',
        architecture='synthetic-control', postgres_image='synthetic-control')
    return build_bundle(directory, lane='fixed', run_id='rcg-control-only', producer=producer, gate_status='PASS')


@pytest.mark.parametrize('damage', ['none','body-missing','body-corrupt','manifest-missing','manifest-digest',
    'symlink','path-escape','duplicate-case','wrong-lane','wrong-producer','wrong-source','case-outcome',
    'response-missing','extra-file','selected-digest','wrong-gate-verdict'])
def test_consumer_bundle_integrity(damage, tmp_path):
    from tools.run_contract_gate.bundle import validate_bundle, write_manifest, digest
    bundle = synthetic_bundle(tmp_path)
    data = json.loads((bundle/'manifest.json').read_text())
    body = bundle/data['samples'][0]['body_file']
    selected = digest(bundle/'manifest.json')
    if damage == 'body-missing': body.unlink()
    elif damage == 'body-corrupt': body.write_bytes(b'{"wrong":true}')
    elif damage == 'manifest-missing': (bundle/'manifest.json').unlink()
    elif damage == 'manifest-digest': (bundle/'manifest.sha256').write_text('0'*64)
    elif damage == 'symlink':
        body.unlink(); body.symlink_to(tmp_path/'result.json')
    elif damage == 'path-escape': data['files']['../result.json'] = data['files'].pop('result.json')
    elif damage == 'duplicate-case': data['samples'][-1] = data['samples'][0]
    elif damage == 'wrong-lane': data['lane'] = 'latest'
    elif damage == 'wrong-producer': data['producer']['candidate_commit'] = '0'*40
    elif damage == 'wrong-source': data['producer']['source_sha256'] = '0'*64
    elif damage == 'case-outcome': data['samples'][0]['db_assertions_passed'] = False
    elif damage == 'response-missing': data['samples'].pop()
    elif damage == 'extra-file': (bundle/'unreviewed.txt').write_text('synthetic')
    elif damage == 'wrong-gate-verdict': data['gate_status'] = 'FAIL'
    if damage in ('path-escape','duplicate-case','wrong-lane','wrong-producer','wrong-source',
                  'case-outcome','response-missing','wrong-gate-verdict'):
        write_manifest(bundle, data)
        selected = digest(bundle/'manifest.json')  # Semantic checks must survive recomputed hashes.
    if damage == 'selected-digest': selected = '0'*64
    args = dict(lane='fixed', producer_sha='c'*40, source_sha256='d'*64,
                manifest_sha256=selected, run_id='rcg-control-only', gate_status='PASS')
    if damage == 'none':
        assert validate_bundle(bundle, **args)['status'] == 'VERIFIED'
    else:
        with pytest.raises((OSError, ValueError, KeyError, TypeError, AssertionError)):
            validate_bundle(bundle, **args)


@pytest.mark.parametrize('damage', ['none','missing-blocker','same-pid','worker-timeout','missing-writes'])
def test_contention_proof_classification(damage, tmp_path):
    result = dict(pids={'A':10,'B':11}, workers_alive=[], timeline=[
        dict(sequence=0,request_id='A',event='scope_read',found=False),
        dict(sequence=1,request_id='B',event='scope_read',found=False),
        dict(sequence=2,request_id='A',event='bind_complete'),
        dict(sequence=3,request_id='observer',event='blocking_observed',rows=[{'pid':11,'blockers':[10]}]),
        dict(sequence=4,request_id='observer',event='release_A')])
    if damage == 'missing-blocker': result['timeline'][3]['rows'][0]['blockers'] = []
    elif damage == 'same-pid': result['pids']['B'] = 10
    elif damage == 'worker-timeout': result['timeline'].append(dict(event='worker_timeout'))
    elif damage == 'missing-writes': result['timeline'].pop(2)
    text = ('import pytest\nfrom tools.run_contract_gate.contention import verify_contention\n'
            '@pytest.fixture\ndef proof():\n    verify_contention('+repr(result)+')\n'
            'def test_contention(proof): pass\n')
    (tmp_path/'test_proof.py').write_text(text)
    code = """
import json,pytest,sys
from tools.run_contract_gate.lane import Reports
from tools.run_contract_gate.report import classify
p=Reports()
rc=int(pytest.main(['-q','-p','no:cacheprovider',sys.argv[1]],plugins=[p]))
print('VERDICT='+json.dumps(classify(p.collected,p.reports,rc,{'test_contention'})))
"""
    process = subprocess.run([sys.executable,'-c',code,str(tmp_path/'test_proof.py')],
                              capture_output=True,text=True,timeout=30,check=True)
    verdict = json.loads(next(line.removeprefix('VERDICT=') for line in process.stdout.splitlines() if line.startswith('VERDICT=')))
    assert verdict['status'] == ('PASS' if damage == 'none' else 'INCOMPLETE')


def test_worker_timeout_cancels_and_joins():
    """Actual stalled thread; explicit connection double models cancellation unblock."""
    import threading
    from tools.run_contract_gate.contention import join_workers
    release, started = threading.Event(), threading.Event()
    records = []
    class CancelConnection:
        closed = False
        def cancel(self):
            records.append('cancel')
            release.set()
    def worker():
        started.set(); release.wait(5)
    thread = threading.Thread(target=worker, name='control-worker')
    thread.start()
    try:
        assert started.wait(1)
        remaining = join_workers([thread], [CancelConnection()], 0.02,
                                 lambda names: records.append(names))
        assert records == [['control-worker'], 'cancel']
        assert remaining == [] and not thread.is_alive()
    finally:
        release.set(); thread.join(1)


@pytest.mark.parametrize('damaged_lane', ['fixed', 'latest'])
def test_bundle_failure_is_required_incomplete(damaged_lane, tmp_path, monkeypatch):
    """Drive the actual finalizer: even latest integrity damage fails the common gate."""
    import shutil
    from tools.run_contract_gate.finalize import main, control_names
    from tools.run_contract_gate.bundle import validate_bundle
    producer = None
    for lane in ('fixed', 'latest'):
        directory = tmp_path/lane
        directory.mkdir()
        bundle = synthetic_bundle(directory)
        producer = json.loads((bundle/'manifest.json').read_text())['producer']
        shutil.rmtree(bundle)
        (tmp_path/(lane+'.completed')).touch()
        (tmp_path/(lane+'.sha')).write_text('a'*40)
        (tmp_path/(lane+'.ddl.sha256')).write_text('b'*64)
    (tmp_path/'controls').mkdir()
    valid_artifacts(tmp_path/'controls', control_names())
    (tmp_path/'controls.completed').touch()
    (tmp_path/'cleanup.ok').touch()
    (tmp_path/'inputs.json').write_text(json.dumps(producer))
    next((tmp_path/damaged_lane/'responses').glob('*.body')).unlink()
    monkeypatch.setenv('RCG_RUN_ID', 'rcg-control-only')
    monkeypatch.setenv('RCG_DURATION', '0')
    assert main(tmp_path) == 2
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['status'] == 'INCOMPLETE' and not result['artifacts_ok']
    assert result['consumer_bundles'][damaged_lane]['status'] == 'INCOMPLETE'
    if damaged_lane == 'latest':
        assert result['latest_status'] == 'WARN_UNVERIFIED'
    other = 'latest' if damaged_lane == 'fixed' else 'fixed'
    verified = validate_bundle(tmp_path/other/'consumer', lane=other, producer_sha='c'*40,
        source_sha256='d'*64, gate_status='INCOMPLETE',
        manifest_sha256=result['consumer_bundles'][other]['manifest_sha256'])
    assert verified['gate_status'] == 'INCOMPLETE'
