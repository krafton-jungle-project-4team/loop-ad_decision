# ANN efficiency recovery v1

Status: `follow_up_required`

## Question

Can HNSW outperform Exact vector search for large audience candidate retrieval
without violating Recall@K, and is that sufficient to adopt ANN as the audience
membership policy?

## Setup

- Expedia `hotel_behavior.v2` behavior vectors
- 1,198,786 source users; maximum measured cohort of 1,000,000
- 64-dimensional vectors
- observation window: 2013-01-01 through 2015-01-01 UTC
- 300 warm runs and 30 database-cold runs for final confirmation
- Recall@K and Wilson lower bound both required to be at least 0.95
- ANN p95 / Exact p95 required to be at most 0.80
- actual HNSW index use required; spill and OOM forbidden

## Result

Query-specific candidate retrieval passed in four regions at the 1M cohort:

| Candidate type | K | HNSW setting | Exact p95 | ANN p95 | Recall@K | DB-cold p95 improvement |
|---|---:|---|---:|---:|---:|---:|
| `funnel_recovery` | 500 | ef=200, relaxed, scan=20K | 113.20 ms | 6.94 ms | 0.9880 | 78.5% |
| `intent_matched` | 1,000 | ef=400, relaxed, scan=50K | 115.79 ms | 9.21 ms | 0.9860 | 68.6% |
| `target_destination_affinity` | 500 | ef=100, relaxed, scan=50K | 114.95 ms | 4.43 ms | 0.9820 | 89.5% |
| `target_destination_affinity` | 5,000 | ef=200, relaxed, scan=20K | 127.55 ms | 35.03 ms | 0.9896 | 23.7% |

The representative `funnel_recovery` cell was approximately 16.3 times
faster at warm p95 while maintaining Recall@500 of 0.988.

The corrected full-membership stage tested 46 candidate settings. None passed
pre-confirmation, so no final full-membership confirmation was run. Therefore:

- candidate retrieval passed;
- corrected full membership did not pass;
- no query-independent common setting was established;
- ANN is not a product-policy candidate yet;
- the runtime must retain the Exact fallback.

## Scale and integrity

The valid final dataset contains 5,687 phase-level cells, 73,110 stored raw
rows, and 105,970 logical query or plan invocations. Measurement ran for about
11 hours. An earlier screening dataset with an RSS measurement defect was
invalidated and excluded from the totals and conclusion.

The local finalization report recorded `1065 passed, 7 skipped`, zero missing
required artifacts, zero JSONL parse errors, and successful raw/result
reconciliation.

## Claim boundary

This checkpoint supports a portfolio claim about measured candidate-retrieval
performance and the engineering process used to validate it. It does not
support claiming that ANN was deployed, that all audience queries improve, or
that full audience membership can switch from Exact search.

## Follow-up

1. Establish a query-independent runtime setting or schedule.
2. Measure cohorts larger than 1M.
3. Add an independent `general_destination_explorer` confirmation query.
4. Rework membership recovery, then repeat corrected full-membership gates.
