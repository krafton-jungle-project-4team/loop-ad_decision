"""Versioned, byte-preserving synthetic run responses for a pinned consumer."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from .report import MANIFEST, SHA, validate_lane

SCHEMA = 'rcg-run-consumer.v1'
REPOSITORY = 'krafton-jungle-project-4team/loop-ad_decision'
HEX256 = re.compile(r'^[0-9a-f]{64}$')
PRODUCER_KEYS = ('candidate_commit', 'candidate_dirty', 'source_sha256', 'baseline_producer_sha',
                 'lock_sha256', 'runner_image', 'architecture', 'postgres_image')
SAMPLES = {
    'RCG-01-create': ('RCG-01', 'create', MANIFEST['RCG-01'][0]),
    'RCG-02-first': ('RCG-02', 'first', MANIFEST['RCG-02'][0]),
    'RCG-02-retry': ('RCG-02', 'retry', MANIFEST['RCG-02'][0]),
    'RCG-08-reuse': ('RCG-08', 'reuse', MANIFEST['RCG-08'][0]),
}
for case, test in [('RCG-10', MANIFEST['RCG-10'][0]), ('RCG-11', MANIFEST['RCG-11'][0]),
                   ('RCG-12', MANIFEST['RCG-12'][0]), ('RCG-12-reverse', MANIFEST['RCG-12'][1])]:
    for request in ('A', 'B'):
        SAMPLES[case+'-'+request] = (case, request, test)
EVIDENCE = sorted({case+'.json' for case, _, _ in SAMPLES.values()})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_file(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name or not path.parts:
        raise ValueError('unsafe bundle path: '+str(name))
    current = Path(root)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('bundle symlink refused: '+name)
    if not current.is_file():
        raise ValueError('missing bundle file: '+name)
    return current


def write_manifest(directory, manifest):
    path = directory/'manifest.json'
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    (directory/'manifest.sha256').write_text(digest(path)+'\n')


def build_bundle(directory, *, lane, run_id, producer, gate_status):
    """Copy only reviewed artifact kinds; raw logs, DSNs and environment are excluded."""
    directory = Path(directory)
    target = directory/'consumer'
    target.mkdir(exist_ok=False)
    lane_data = json.loads(safe_file(directory, 'result.json').read_text())
    samples = []
    files = ['result.json', 'junit.xml'] + EVIDENCE
    for stem, (case, request, test) in SAMPLES.items():
        metadata = json.loads(safe_file(directory, 'responses/'+stem+'.json').read_text())
        assert metadata['case_id'] == case and metadata['request_id'] == request
        assert metadata['body_file'] == stem+'.body'
        assert digest(safe_file(directory, 'responses/'+stem+'.body')) == metadata['body_sha256']
        passed = lane_data['reports'][test].get('call', {}).get('outcome') == 'passed'
        samples.append(metadata | dict(body_file='responses/'+stem+'.body', test_name=test,
            case_outcome=lane_data['reports'][test], db_assertions_passed=passed, evidence_file=case+'.json'))
        files.append('responses/'+stem+'.body')
    hashes = {}
    for name in files:
        source = safe_file(directory, name)
        destination = target/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[name] = dict(sha256=digest(destination), bytes=destination.stat().st_size)
    manifest = dict(schema_version=SCHEMA, lane=lane, run_id=run_id,
                    producer={'repository': REPOSITORY} | {key: producer[key] for key in PRODUCER_KEYS},
                    contract_sha=lane_data['contract_sha'], ddl_sha256=lane_data['ddl_sha256'],
                    gate_status=gate_status, lane_status=lane_data['status'],
                    samples=samples, files=hashes)
    write_manifest(target, manifest)
    return target


def validate_bundle(directory, *, lane, producer_sha, source_sha256, manifest_sha256=None,
                    run_id=None, gate_status=None):
    """Raise on missing, forged or mixed evidence; caller maps errors to INCOMPLETE."""
    root = Path(directory)
    assert not root.is_symlink(), 'bundle root symlink refused'
    manifest_path = safe_file(root, 'manifest.json')
    actual_hash = digest(manifest_path)
    assert safe_file(root, 'manifest.sha256').read_text().strip() == actual_hash, 'manifest digest mismatch'
    if manifest_sha256 is not None:
        assert actual_hash == manifest_sha256, 'selected manifest digest differs'
    data = json.loads(manifest_path.read_text())
    assert data['schema_version'] == SCHEMA and data['lane'] == lane, 'bundle schema or lane differs'
    assert lane in ('fixed', 'latest')
    assert isinstance(data['run_id'], str) and data['run_id'].startswith('rcg-')
    if run_id is not None:
        assert data['run_id'] == run_id, 'Gate run differs'
    producer = data['producer']
    assert producer['repository'] == REPOSITORY
    assert SHA.fullmatch(producer_sha) and producer['candidate_commit'] == producer_sha, 'selected producer SHA differs'
    assert HEX256.fullmatch(source_sha256) and producer['source_sha256'] == source_sha256, 'selected source digest differs'
    assert type(producer['candidate_dirty']) is bool
    assert SHA.fullmatch(producer['baseline_producer_sha'])
    assert HEX256.fullmatch(producer['lock_sha256'])
    assert all(isinstance(producer[key], str) and producer[key] for key in ('runner_image','architecture','postgres_image'))
    assert data['gate_status'] in ('PASS','FAIL','INCOMPLETE')
    if gate_status is not None:
        assert data['gate_status'] == gate_status, 'Gate verdict differs'
    assert SHA.fullmatch(data['contract_sha']) and HEX256.fullmatch(data['ddl_sha256'])
    expected_files = {'result.json', 'junit.xml', *EVIDENCE} | {'responses/'+stem+'.body' for stem in SAMPLES}
    assert set(data['files']) == expected_files, 'bundle file inventory differs'
    actual_files = set()
    for path in root.rglob('*'):
        assert not path.is_symlink(), 'bundle contains symlink'
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    assert actual_files == expected_files | {'manifest.json','manifest.sha256'}, 'unexpected or missing file'
    for name, info in data['files'].items():
        path = safe_file(root, name)
        assert path.stat().st_size == info['bytes'] and digest(path) == info['sha256'], 'file digest mismatch: '+name
    result = validate_lane(root, expected_sha=data['contract_sha'], expected_ddl=data['ddl_sha256'])
    assert result.get('stage') != 'result-validation', result.get('reason')
    assert result['status'] == data['lane_status']
    seen = set()
    for sample in data['samples']:
        stem = sample['case_id']+'-'+sample['request_id']
        assert stem in SAMPLES and stem not in seen, 'duplicate or unexpected response'
        seen.add(stem)
        case, request, test = SAMPLES[stem]
        assert sample['test_name'] == test
        assert sample['method'] == 'POST' and sample['path'] == '/decision/v1/promotions/rcg_synthetic_promotion/runs'
        assert isinstance(sample['request'], dict)
        assert type(sample['status']) is int and 100 <= sample['status'] <= 599
        assert isinstance(sample['content_type'], str) and sample['content_type']
        assert sample['body_file'] == 'responses/'+stem+'.body'
        assert sample['body_sha256'] == data['files'][sample['body_file']]['sha256']
        assert sample['evidence_file'] == case+'.json'
        assert sample['case_outcome'] == result['reports'][test]
        passed = result['reports'][test].get('call', {}).get('outcome') == 'passed'
        assert sample['db_assertions_passed'] is passed
    assert seen == set(SAMPLES), 'required response missing'
    return dict(status='VERIFIED', manifest_sha256=actual_hash, lane=lane, responses=len(seen),
                gate_status=data['gate_status'], lane_status=result['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--lane', required=True, choices=('fixed','latest'))
    parser.add_argument('--producer-sha', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = vars(parser.parse_args())
    try:
        print(json.dumps(validate_bundle(**args), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print(json.dumps(dict(status='INCOMPLETE', reason=str(exc))))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
