"""Container-side source fingerprinting; only explicitly staged candidate files exist here."""
import hashlib
import json
import os
from pathlib import Path


def main():
    files = {}
    for directory, prefix in ((Path('/source'),''), (Path('/gate'),'gate/')):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                files[prefix+str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = json.loads(Path('/gate/tests/fixtures/run_contract_gate/baseline/provenance.json').read_text())
    value = dict(candidate_commit=os.environ['RCG_HEAD'], candidate_dirty=os.environ['RCG_DIRTY']=='true',
                 source_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(), files=files,
                 baseline_producer_sha=provenance['producer_sha'], lock_sha256=hashlib.sha256(Path('/requirements.lock').read_bytes()).hexdigest(),
                 runner_image=os.environ['RCG_IMAGE'], architecture=os.environ['RCG_ARCH'], postgres_image=os.environ['RCG_PG_IMAGE'])
    Path('/output/inputs.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
