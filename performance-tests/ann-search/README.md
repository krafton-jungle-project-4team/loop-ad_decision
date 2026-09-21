# ANN audience search performance experiments

This directory is the repository checkpoint for the ANN audience-search
experiments. It follows the same boundary used by the team's infrastructure
performance tests:

- `evidence/` contains small, reviewable conclusions and an experiment index.
- `tools/` contains deterministic validation for the committed evidence.
- `artifacts/` documents the boundary for large local outputs.
- measurement code remains in `offline_evaluation/` and `scripts/`.

The checkpoint is deliberately narrower than a product rollout decision.
It proves a query-specific candidate-retrieval efficiency region at the
measured scale. It does **not** claim that ANN is a valid full-membership
policy or a deployable common runtime setting.

## Validate

Install development dependencies before running the experiment tests:

```bash
python -m pip install -e '.[dev]'
```

Validate the committed checkpoint:

```bash
python performance-tests/ann-search/tools/validate_evidence.py
```

Also verify the ignored local source artifacts when they are present:

```bash
python performance-tests/ann-search/tools/validate_evidence.py \
  --verify-local-artifacts
```

Run the focused experiment tests:

```bash
pytest -q \
  tests/test_ann_search_experiment.py \
  tests/test_ann_search_scale_series.py \
  tests/test_ann_search_goal2.py \
  tests/test_ann_search_goal3.py \
  tests/test_ann_search_checkpoint_evidence.py
```

The source-archive test skips when no local source archive is installed. A
partial archive or a hash mismatch fails. Goal 2 planning tests use synthetic
fixtures and do not require the original measurements.

See [merge readiness](merge-readiness.md) for the dev synchronization checks.

## Current decision

- Candidate retrieval: passed for four query-specific regions at 1M users.
- Full membership: not established.
- Common runtime setting: not established.
- Production policy: keep the Exact fallback.
- Status: follow-up experiments required.
