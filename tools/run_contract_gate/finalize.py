"""Write the final machine-readable summary after host-owned resources are cleaned."""
import hashlib
import json
import os
from pathlib import Path
import sys
from .report import aggregate, validate_lane, SHA


def main():
    output = Path('/output')
    lanes = {}
    artifacts_ok = True
    for lane in ('fixed','latest'):
        if (output/(lane+'.completed')).exists():
            lanes[lane] = validate_lane(output/lane, expected_sha=read_input(output, lane+'.sha'),
                                        expected_ddl=read_input(output, lane+'.ddl.sha256'))
            if lanes[lane].get('stage') == 'result-validation':
                artifacts_ok = False
        else:
            reason = (output/(lane+'.error')).read_text() if (output/(lane+'.error')).exists() else 'lane not completed'
            lanes[lane] = dict(status='INCOMPLETE', stage='preparation-or-execution', reason=reason, collected=[])
        if (output/(lane+'.sha')).exists():
            lanes[lane]['resolved_sha'] = (output/(lane+'.sha')).read_text().strip()
    controls = validate_lane(output/'controls', required=control_names(), expected_sha=read_input(output, 'fixed.sha'),
                             expected_ddl=read_input(output, 'fixed.ddl.sha256')) if (output/'controls.completed').exists() else {'status':'INCOMPLETE'}
    cleanup_ok = (output/'cleanup.ok').exists()
    summary = aggregate(lanes['fixed'], lanes['latest'], controls_ok=controls['status']=='PASS',
                        cleanup_ok=cleanup_ok, artifacts_ok=artifacts_ok)
    provenance = json.loads((output/'inputs.json').read_text())
    if not SHA.fullmatch(provenance.get('candidate_commit','')):
        summary.update(status='INCOMPLETE', exit_code=2)
    if (output/'interrupted').exists() or (output/'host.error').exists():
        summary.update(status='INCOMPLETE', exit_code=2)
    summary.update(schema_version=1, run_id=os.environ['RCG_RUN_ID'], inputs=provenance,
                   lanes=lanes, controls=controls, duration_seconds=int(os.environ['RCG_DURATION']),
                   artifacts={'fixed_junit':'fixed/junit.xml','latest_junit':'latest/junit.xml','controls_junit':'controls/junit.xml'},
                   cleanup_log='cleanup.log', resources={'containers':[os.environ['RCG_RUN_ID']+'-db', os.environ['RCG_RUN_ID']+'-runner'], 'volumes':[os.environ['RCG_RUN_ID']+'-socket']})
    (output/'result.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    (output/'finalizer.exit').write_text(str(summary['exit_code'])+'\n')
    print(f"Run Contract Gate: {summary['status']} (exit {summary['exit_code']})")
    print(f"fixed: {lanes['fixed']['status']}; latest: {summary['latest_status']}; controls: {controls['status']}; cleanup: {cleanup_ok}")
    for name, data in lanes.items():
        print(f"{name}: SHA={data.get('resolved_sha', 'unresolved')} cases={len(data.get('collected', []))} reason={data.get('reason', '')}")
    return summary['exit_code']


def read_input(output, name):
    path = output/name
    return path.read_text().strip() if path.exists() else ''


def control_names():
    return set(json.loads(Path(__file__).with_name('control-manifest.json').read_text()))


if __name__ == '__main__':
    sys.exit(main())
