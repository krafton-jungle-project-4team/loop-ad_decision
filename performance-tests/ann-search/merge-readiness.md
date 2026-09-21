# ANN evidence merge readiness — 2026-09-21

The experiment branch incorporates `origin/dev` at
`e1de8b29b902b54df3a58f21f1daa27c1171fe80`. Git merged it without textual
conflicts. The published measurements and their claim boundaries are unchanged.

## Verdict

`merge-safe` for the evidence checkpoint and the default-preserving repository
options, based on local verification. This is not approval to deploy ANN as a
membership policy. No new performance measurements were collected.

## Affected Dashboard journey

Analysis -> audience snapshot -> allocation -> run binding -> assignment ->
Dashboard DB reads. The default predicate batch size and Exact fallback remain
unchanged. The bounded-cohort count helper is used by the offline benchmark.
The last locally verified database action is allocation/run binding from older
snapshot selection methods (`transition`, `exact_fallback`); deployed Dashboard
actions were not exercised.

## Contract impact

The only application change relative to dev is in
`app/analysis/audience_search_repository.py`: an optional positive batch size
and an offline bounded-cohort count helper. No public API, DTO, persistent ID,
status, table, or event contract changes are introduced.

Analysis still produces the existing audience snapshot/member rows; allocation
and run binding consume those rows; downstream assignment and Dashboard DB
read contracts retain their dev implementation. No Dashboard changes or new
Data Contract migration are required.

## Existing-data compatibility

Loaded upstream Data Contract `main` at
`0ec2cef0290f4659ad21ccc1dd2a20df2801ff50` into a disposable local
`pgvector/pgvector:0.8.5-pg16` database. The integration test creates snapshots
with both older selection methods and exercises allocation and run binding
through the new checkout. Both cases passed.

## Verification and fixes

- `.venv/bin/python -m pytest -q`: **1302 passed, 8 skipped**. Seven existing
  opt-in database tests and one optional local source-archive test were skipped.
- `tests/test_lean_audience_contract_integration.py` with the disposable local
  PostgreSQL DSN: **2 passed**, separately covering two of those database skips.
- `python performance-tests/ann-search/tools/validate_evidence.py`: passed.
- The same validator with `verify_local_artifacts=True` and its source root set
  to the original experiment worktree: all six recorded source hashes passed.
- Batch sizes 1, 2, and 10000 produce the same bounded-cohort hard-match count
  and visit every supplied user exactly once in the regression fixture.
- The dev-relative diff passes whitespace checks. Graphify AST update completed.
- One existing FastAPI/Starlette test-client deprecation warning remains.

Merge preparation fixed missing `matplotlib` development dependency, removed
private-file dependencies from Goal 2 planning tests using explicitly synthetic
fixtures, and made source-archive validation optional only when the entire
archive is absent. Partial or changed archives still fail validation. An
existing dev test expecting email offer-cards v4 was updated to the implemented
v5 contract, consistent with the dedicated email-variant tests.

## Merge order

This checkpoint can merge independently after branch review. No cross-service
deployment, flag change, or backfill is needed. Runtime ANN adoption remains
outside this change; Exact fallback remains in effect.

## Rollback boundary

Revert the evidence branch delta relative to dev if needed. Do not revert the
upstream dev commits incorporated by the synchronization merge. There are no
new persistent rows, migrations, or changed ID meanings to undo.

## Excluded local context

`AGENTS.md`, `agent/`, graphify outputs, untracked render scripts, and raw local
experiment artifacts are excluded from staging, commits, and the PR.

## Optional portfolio follow-up

A low-cost next step is local resampling and sensitivity analysis of the
existing per-run measurements, reporting how stable p95 ratios and the 750K/1M
crossover conclusion are under different resamples. Label this as reanalysis
of existing data. It cannot establish a larger-scale crossover, an independent
query result, or a deployable common ANN setting without new measurements.
