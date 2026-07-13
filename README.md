# Loop-Ad Decision API

Loop-Ad Decision API is a lifecycle write API service for analysis, generation,
promotion run creation, segment assignment, evaluation, and next-loop
orchestration.

The service is not a Dashboard API, ChatKit API, or advertisement-serving
Decision hot path. Dashboard-owned systems handle segment query preview,
ChatKit flows, banner resolve, redirect handling, dispatch, public read APIs,
and any public recommendation-style API surface.

## Serving Boundary

Dashboard and ad execution must not synchronously call Decision for per-request
serving. They should read the contract database directly. When available, they
should read the Data Source Contract owned `active_ad_serving_assignments` view.

Decision does not provide active_ad_serving_assignments and does not own that
view. The data-source contract owns database schemas and serving views.

## Promotion Run Scope

Promotion runs are idempotent by project, promotion, analysis, generation,
normalized non-fallback `segment_ids`, and loop count. A retry of the same scope
returns the stored run; a different segment scope creates an independent run.
Run responses expose the normalized `segment_ids`, and each ad experiment marks
fallback membership with `is_fallback`.

`segment_scope_fingerprint` is SHA-256 over only the sorted, unique segment ID
array serialized as compact JSON. The remaining identity fields are enforced by
the composite database constraint and a short digest in `promotion_run_id`.

`LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED` is a strict boolean and defaults to
`false`. While disabled, explicit run scopes and failed-only automatic next-loop
requests return 409 before any lifecycle writes. Enable it only after the Data
Source Contract expand/backfill/finalize rollout and the Dashboard exact
scope/lineage reader are deployed.

Dashboard integration requirements and the versioned response fixture are in
[`docs/dashboard-segment-experiment-integration-fix-spec.md`](docs/dashboard-segment-experiment-integration-fix-spec.md).

Automatic next-loop analysis and generation IDs include a bounded digest of the
source promotion run. Different source scopes can therefore advance to the same
loop count without colliding in their upstream lifecycle rows or generated
content IDs.

## Next Loop Integration Note

B6 next-loop currently defines the decision-side orchestration and the
analysis/generation call boundary. The real analysis and generation adapters are
left for a follow-up integration PR after the analysis and generation flows are
ready to honor failed segment focus inputs end to end.

## Logging Work Rule

Before adding or changing application logs, read
[docs/reference_logging.md](docs/reference_logging.md). Decision logs must stay
JSON structured, use context propagation, keep stable snake_case `event` names,
and follow the shared Loop-Ad logging standard from the Dashboard API reference.

## Local Validation Tools

- [Expedia 세그먼트 추천 백테스트](docs/expedia_segment_backtest.md): 과거 행동과
  미래 예약 라벨을 시간 분리해 AI 추천 후보, Rank, 예상 전환율을 검증한다.
- [외부 데이터셋 세그먼트 추천 검증](docs/external_segment_backtest.md): Airbnb,
  Booking.com, Synerise의 서로 다른 결과 계약으로 추천의 외부 일반화를 검증한다.
