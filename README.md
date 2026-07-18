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

Explicit run scopes and failed-only automatic next-loop requests are always
enabled. Deploy this Decision version only after the Data Source Contract
expand/backfill/finalize rollout is complete and the Dashboard exact
scope/lineage reader is deployed.

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
  미래 예약 라벨을 시간 분리해 AI 추천 후보 묶음과 예상 전환율을 검증한다.
- [외부 데이터셋 세그먼트 추천 검증](docs/external_segment_backtest.md): Airbnb,
  Booking.com, Synerise의 서로 다른 결과 계약으로 추천의 외부 일반화를 검증한다.

## Full-source Sealed Evaluation

2026-07-15 평가는 Kaggle Expedia Hotel Recommendations의 `train.csv` 원본 전체를
로컬 서비스 ClickHouse `expedia_hotel_events` 테이블에 적재한 뒤 실행했다.

- 원본 행동 로그: 37,670,293건
- 사용자: 1,198,786명
- 예약 행: 3,000,693건
- 관찰 기간: 2013-01-07 ~ 2014-12-31
- 사용자 추가 표본 추출: 없음 (`user_sample_modulo=1`)
- 원본 fingerprint: `6e779cce23d70b54e9733784688f2a9224803b76d0eb98e6c02cd7af15ba5f75`

모델은 2013년 행동으로 만든 96개 후보 성과 사례와 5,554명 후보 사용자 관측을
학습했다. 2014년 데이터는 개발 검증에 사용했고, 최종 평가는 개발 목적지와 겹치지
않는 2014년 7~12월의 18개 목적지 시나리오를 manifest에 먼저 봉인한 뒤 한 번
실행했다. 따라서 “3,767만 행을 그대로 모델 입력으로 사용했다”기보다, 전체 행동
로그로 시점별 사용자 행동 특성과 미래 30일 목적지 일치 예약 결과를 산출해
학습·검증했다고 해석해야 한다.

### Expedia 최종 평가 결과

- manifest: `0b02550d60ee50ba54eb25aba6f1fb83cf654df11f2bb0047553b9e8063fc692`
- 실행 코드: `8f798ab1f40fa9322970a4365a73515ddae391c0`
- 모델 SHA-256: `30312e413c5520e28aa0c2c08c89350224df9cf8a7e2cf22c3b88aad9cee6aab`
- 판정: **실패 (`failed`)**
- 결과가 관측된 시나리오: 15개
- 평가 후보 결과: 43개
- 전체 기준률보다 성과가 높은 후보 비율: 90.70% (기준 60% 이상, 통과)
- 하나 이상의 유효 후보를 찾은 시나리오 비율: 100% (기준 70% 이상, 통과)
- 모든 후보가 기준률을 넘은 시나리오 비율: 80% (기준 50% 이상, 통과)
- 후보 평균 향상: +7.63%p (기준 0%p 이상, 통과)
- 가장 성과가 낮은 후보의 평균 향상: +4.95%p (기준 0%p 이상, 통과)
- 예상 전환율 편향: +0.71%p (절댓값 기준 1.5%p 이하, 통과)
- 예상 전환율 평균 절대오차: 4.63%p (기준 3.5%p 이하, **실패**)
- Brier skill score: 0.0063 (기준 0 초과, 통과)

이 결과는 추천 로직이 전체 사용자보다 미래 예약 가능성이 높은 고객군을 찾는
능력은 확인했지만, 후보별 예상 예약 전환율을 사전 기준 이내로 정밀하게 맞히지는
못했다는 뜻이다. 최종 데이터에 맞춰 기준이나 모델을 사후 변경하지 않는다.

과거 371,836행의 결정적 1% 사용자 표본으로 실행한 결과는 전체 원본 평가로 볼 수
없어 공식 결론에서 제외했다. 해당 실행 기록은 삭제하거나 덮어쓰지 않고 감사
목적으로만 보존한다.

### 외부 데이터 재평가

전체 Expedia 원본으로 학습한 동일 모델을 고정한 뒤 Airbnb, Booking.com,
Synerise 원본 해시를 각각 새 manifest에 봉인해 실행했다.

| 데이터셋 | 판정 | 관측 시나리오 | 후보 평균 향상 | 해석 |
| --- | --- | ---: | ---: | --- |
| Airbnb | 통과 | 1 | +1.04%p | 첫 예약 사용자 농축 여부만 검증 가능 |
| Booking.com | 통과 | 3 | +8.73%p | 3개 중 2개 시나리오에서 기준률 초과 |
| Synerise | 판단 유보 | 2 | +3.29%p | 품질 지표는 양수지만 최소 3개 관측 기준 미달 |

외부 데이터의 결과 정의는 Expedia의 “향후 30일 목적지 일치 예약”과 다르므로
예상 예약 전환율의 오차를 직접 비교하지 않는다. 외부 평가는 추천 후보가 각
데이터셋의 결과 사용자를 평균보다 많이 포함하는지 확인하는 보조 근거이며,
숙박 예약 전환율 모델의 일반화가 최종 증명됐다는 의미는 아니다.
