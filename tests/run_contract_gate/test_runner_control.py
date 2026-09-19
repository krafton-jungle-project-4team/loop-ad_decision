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


def valid_artifacts(directory):
    names = ['test_a','test_b']
    reports = {name:{phase:{'outcome':'passed','wasxfail':False} for phase in ('setup','call','teardown')} for name in names}
    data = dict(contract_sha='a'*40,ddl_sha256='b'*64,collected=names,reports=reports,pytest_exit=0,status='PASS')
    (directory/'result.json').write_text(json.dumps(data))
    root = ET.Element('testsuites'); suite = ET.SubElement(root,'testsuite',tests='2')
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
