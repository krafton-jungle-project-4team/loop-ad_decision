# 대시보드 세그먼트 단위 실험 연동 수정 명세

작성 기준: 2026-07-13
대시보드 대상 브랜치: `feat/experiment-create-seperation` (`11ee2e6`)
Decision 연동 기준: PR #230 scope 멱등성과 `dev` PR #232 preparation lineage 통합본

## 1. 결론

대시보드의 run 생성 요청 JSON은 현재 정상이다. `segment_ids: [segmentId]`를 포함한 객체를 `JSON.stringify`하고 `Content-Type: application/json`으로 전송하고 있으므로, PostgreSQL의 `Token "("` JSONB 500을 대시보드 직렬화 코드로 고치면 안 된다.

> **운영 확인:** 새 Dashboard가 `segment_ids`를 보내는데 Decision의
> `LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED`가 `false`이면 Decision은
> lifecycle write 전에 의도적으로 `409`를 반환한다. 이는 JSON 직렬화 오류가
> 아니다. Dashboard와 Decision의 활성화 시점을 반드시 아래 운영 계약에 맞춘다.

대시보드에 실제로 필요한 수정은 Decision이 함께 생성하는 **fallback 광고 실험을 응답 파싱 과정에서 보존하고, fallback 배정이 발생했을 때 선택 세그먼트 실험과 함께 실행 상태로 전환하는 것**이다.

현재 대시보드는 다음 문제가 있다.

1. Decision 응답의 최상위 `segment_ids`와 각 실험의 `is_fallback`을 Zod 파싱 과정에서 제거한다.
2. 신규 run과 다음 루프에서 선택 세그먼트 실험만 시작하고 fallback 실험은 `planned`로 남긴다.
3. 기존 run 재사용 시 먼저 선택 세그먼트로 목록을 좁힌 뒤 같은 run을 구성하므로, 같은 run의 fallback 실험을 실행 대상으로 복원하지 못한다.
4. fallback 배정이 있어도 fallback 실험이 `planned`이면 `active_ad_serving_assignments` 뷰에서 해당 배정이 제외된다. 그 결과 onsite 노출 또는 email/SMS 발송 대상이 누락될 수 있다.

Dashboard가 계약 테스트에 사용할 단일 JSON 기준은
[`docs/contracts/decision-promotion-run-response.v1.json`](./contracts/decision-promotion-run-response.v1.json)이다.
문서의 축약 예시보다 이 versioned fixture를 우선한다.

## 2. 합의할 API 의미

### 2.1 run 생성 요청

대시보드는 현재 규칙을 유지한다.

```json
{
  "analysis_id": "analysis-1",
  "generation_id": "generation-1",
  "segment_ids": ["segment-1"],
  "loop_count": 1
}
```

- 일반 Dashboard 신규 실행의 `segment_ids`는 필수이며 정확히 1개다.
- Decision 도메인과 next-loop 요청·응답은 여러 실패 세그먼트를 허용한다. Dashboard의 단일 선택 정책을 Decision 제약으로 전파하지 않는다.
- fallback 세그먼트인 `seg_existing_all`을 요청에 직접 넣지 않는다.
- 같은 범위의 run이 이미 있으면 Decision의 멱등 응답 `200`과 기존 `promotion_run_id`를 정상 재사용한다. 중복 요청을 `409`로 바꾸지 않는다.

### 2.2 run 생성 응답

Decision 응답에서 최상위 `segment_ids`는 **사용자가 요청한 비-fallback 세그먼트 범위**다. `ad_experiments`에는 그 범위의 실험과 시스템 fallback 실험이 함께 들어올 수 있다.

```json
{
  "promotion_run_id": "run-1",
  "segment_ids": ["segment-1"],
  "ad_experiments": [
    {
      "ad_experiment_id": "experiment-selected",
      "segment_id": "segment-1",
      "is_fallback": false,
      "status": "planned"
    },
    {
      "ad_experiment_id": "experiment-fallback",
      "segment_id": "seg_existing_all",
      "is_fallback": true,
      "status": "planned"
    }
  ]
}
```

fallback 실험은 사용자가 선택한 두 번째 세그먼트가 아니다. 매칭 점수 미달, 후보 없음, 사용자 벡터 오류 등을 처리하는 동일 run의 시스템 안전망이다.

Dashboard는 모든 run 응답에서 다음 불변조건을 검증해야 한다.

- `segment_ids` 집합과 `is_fallback === false` 실험의 `segment_id` 집합이 정확히 일치한다.
- 각 비-fallback 세그먼트의 실험은 정확히 1개다.
- `is_fallback === true`이면서 `segment_id === "seg_existing_all"`인 실험은 정확히 1개다.
- 불일치하면 assignment 생성, 실험 시작, dispatch를 모두 중단한다.

### 2.3 배정 생성 응답

`POST /decision/v1/promotion-runs/{promotion_run_id}/segment-assignments/build` 응답의 다음 필드를 실행 판단에 사용한다.

- `batch_has_fallback`
- `fallback_count`
- `below_threshold_fallback_count`
- `no_candidate_fallback_count`
- `invalid_user_vector_fallback_count`

`batch_has_fallback === true` 또는 `fallback_count > 0`이면 해당 run의 fallback 실험도 serving 가능한 상태여야 한다.

## 3. 필수 변경사항

### 3.1 Decision 응답 필드를 보존한다

대상:

- `packages/shared/src/dashboard/promotion-run.ts`
- `apps/api-server/src/features/dashboard/provider/dashboard-decision-client.ts`

변경:

1. `DashboardPromotionRunAdExperimentSchema`와 Decision client 내부 실험 스키마에 `is_fallback: z.boolean()`을 추가한다.
2. `DashboardCreatePromotionRunResultSchema`와 Decision client 내부 run 스키마에 `segment_ids: z.array(z.string().min(1))`를 추가한다.
3. `decisionNextLoopResponseSchema`도 다중 `segment_ids`와 `next_ad_experiments[].is_fallback`을 필수로 보존한다.
4. Decision 계약상 두 필드는 필수이므로 임의 기본값으로 누락을 숨기지 말고, 누락 시 계약 오류가 드러나게 한다.

### 3.2 DB에서 다시 읽은 실험에도 fallback 여부를 제공한다

대상:

- `DashboardAdExperimentSchema`
- `apps/api-server/src/features/dashboard/repository/dashboard-campaign-mappers.ts`
- 필요 시 `apps/api-server/src/features/dashboard/database/dashboard.sql` 및 pgtyped 생성물

요구사항:

- 기존 run 재사용 경로에서도 각 실험의 fallback 여부를 알 수 있어야 한다.
- `ad_experiments` 테이블에 별도 `is_fallback` 컬럼이 없으므로, DB 조회 결과는 계약 상수 `seg_existing_all`과 `segment_id`를 비교해 `is_fallback`을 파생할 수 있다.
- fallback ID 문자열은 한 곳의 상수로 관리하고 여러 컴포넌트에 매직 문자열로 반복하지 않는다.
- API 응답을 받은 직후에는 Decision이 제공한 `is_fallback` 값을 그대로 사용한다.

### 3.3 실행 흐름이 선택 실험과 필요한 fallback 실험을 함께 처리하게 한다

대상:

- `apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/promotionExperimentFlow.ts`
- `apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/usePromotionWorkspaceController.ts`
- `apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/experiment/ExperimentComponent.tsx`

`ExperimentLaunchTarget`에 `isFallback: boolean`을 추가하고 신규 run, 기존 run, 다음 루프의 모든 매핑에서 값을 전달한다.

실행 순서는 다음과 같다.

1. 선택 세그먼트의 실험으로 재사용할 `promotion_run_id`를 찾는다.
2. 기존 run이면 선택 세그먼트 목록만 유지하지 말고, 전체 입력에서 같은 `promotion_run_id`의 모든 실험을 다시 모은다.
3. 기존 run이 없으면 run을 생성한다.
4. 선택 세그먼트의 비-fallback 실험이 정확히 1개인지 확인한다.
5. segment assignments를 생성하고 결과를 보존한다. `buildAssignments` 반환 타입을 `unknown`으로 버리지 않는다.
6. 기본 실행 대상은 선택 세그먼트 실험이다.
7. 배정 결과에 fallback이 있으면 `isFallback === true`인 실험도 필수 실행 대상에 추가한다.
8. 선택하지 않은 다른 비-fallback 세그먼트 실험은 시작하지 않는다.
9. 필수 실행 대상이 `planned` 또는 `approved`이면 start API를 호출한다. 이미 `running`이면 실행 준비가 된 것으로 간주한다.
10. fallback 배정이 있는데 fallback 실험이 없거나, 필수 대상 시작에 실패하면 email/SMS dispatch를 실행하지 않고 오류를 노출한다.
11. email/SMS dispatch는 필요한 선택 실험과 fallback 실험이 모두 serving 가능한 상태가 된 뒤 dispatch 작업 상태를 확인해 실행한다.

권장 판단식:

```ts
const fallbackRequired =
  assignmentResult.batch_has_fallback || assignmentResult.fallback_count > 0;

const requiredExperiments = fallbackRequired
  ? [selectedExperiment, fallbackExperiment]
  : [selectedExperiment];
```

`is_fallback`을 보존하는 목적은 “모든 실험을 시작”하기 위함이 아니다. **선택한 비-fallback 실험 + 실제 배정이 발생한 fallback 실험만 시작**하기 위함이다.

### 3.4 결과와 오류를 정확히 표시한다

- `startedExperimentIds`에는 이번 요청에서 실제 start가 성공한 실험 ID를 담는다.
- 이미 `running`인 필수 실험은 중복 start 없이 준비 완료로 판단한다.
- `failedExperimentIds`에는 선택 실험뿐 아니라 fallback 실험 시작 실패도 포함한다.
- fallback이 필요한데 응답에 fallback 실험이 없으면 조용히 계속하지 말고 명시적인 계약 오류를 발생시킨다.
- 일부 필수 실험만 시작된 상태에서는 `dispatched: false`를 반환한다.

## 4. 수정하지 말아야 할 사항

- 대시보드의 `JSON.stringify` 또는 `Content-Type: application/json` 처리를 우회하지 않는다.
- PostgreSQL JSONB 변환을 대시보드에서 수행하지 않는다. `segment_scope_json` 저장과 `Jsonb(...)` 변환은 Decision repository 책임이다.
- run 중복 생성 규칙을 `409`로 변경하지 않는다. 현재 UI는 기존 run 재사용을 전제로 한다.
- fallback 실험을 “요청 범위 밖의 잘못 생성된 일반 실험”으로 제거하지 않는다.
- 선택하지 않은 다른 일반 세그먼트 실험까지 일괄 시작하지 않는다.
- `segment_ids` 정확히 1개라는 대시보드 요청 검증을 완화하지 않는다.
- Decision에 dispatch, 광고 실행 또는 Dashboard용 조회 API를 추가하지 않는다.

## 5. 필수 테스트

### 5.1 API 계약 테스트

`apps/api-server/tests/dashboard-decision-client-contract.test.ts`

1. run 생성 응답에 선택 실험과 fallback 실험을 함께 넣는다.
2. 파싱 결과에 최상위 `segment_ids`가 그대로 남는지 확인한다.
3. 각 실험의 `is_fallback`이 그대로 남는지 확인한다.
4. 다음 루프 응답의 `next_ad_experiments[].is_fallback`도 보존되는지 확인한다.
5. `segment_ids` 또는 `is_fallback` 누락 응답이 계약 오류로 처리되는지 확인한다.

### 5.2 실행 흐름 단위 테스트

`apps/web-client/tests/promotion-experiment-flow.test.ts`

최소 다음 시나리오를 추가한다.

1. **fallback 배정 없음**: 선택 실험만 시작하고 fallback 실험은 시작하지 않는다.
2. **fallback 배정 있음**: 선택 실험과 fallback 실험을 모두 시작한 뒤 한 번 dispatch한다.
3. **다른 일반 세그먼트 포함**: 선택 실험과 필요한 fallback만 시작하고 다른 일반 실험은 시작하지 않는다.
4. **fallback 시작 실패**: `failedExperimentIds`에 fallback ID가 포함되고 dispatch하지 않는다.
5. **fallback 실험 누락**: fallback 배정이 있으면 명시적 오류로 실패하고 dispatch하지 않는다.
6. **기존 run 재사용**: run을 새로 만들지 않고 같은 run의 fallback 실험까지 복원해 동일 규칙으로 실행한다.
7. **이미 running**: 중복 start 없이 실행 준비 완료로 보고 dispatch 조건을 만족한다.
8. **다음 루프**: `next_ad_experiments`의 fallback 플래그가 실행 흐름까지 전달된다.

호출 순서 예시:

```text
create -> build -> start:selected -> start:fallback -> dispatch
```

fallback이 없는 배정의 호출 순서:

```text
create -> build -> start:selected -> dispatch
```

### 5.3 dispatch는 작업 상태와 멱등 키로 재시도한다

dispatch 여부를 “이번 요청에서 새로 시작한 실험이 있는가”로 판단하면 안 된다. 첫 시도에서 실험 시작 후 dispatch가 실패하면, 재시도에서는 실험이 이미 `running`이므로 발송이 영구 누락될 수 있다.

다음 조건일 때 dispatch를 호출한다.

```text
필요한 모든 실험이 running 또는 발송 준비 완료
AND 같은 dispatch 작업이 성공·접수된 기록이 없음
```

멱등 키는 재시도마다 새로 만들지 않고 다음 요소로 고정한다.

```text
promotion_run_id + channel + 실행 범위/dispatch 목적
```

- 타임아웃이나 실패 작업만 같은 키로 재시도한다.
- 이미 접수되었거나 완료된 작업은 다시 발송하지 않는다.
- 실험이 이번 시도에서 새로 시작됐는지는 dispatch 필요 여부와 무관하다.

### 5.4 회귀 검증

```bash
npm test
npm run typecheck
```

두 명령이 모두 통과해야 한다.

## 6. Feature flag 운영 계약

`LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED`의 기본값은 `false`다. 환경 변수가 없거나 `false`인 상태를 부분 scope 기능이 활성화된 것으로 간주하면 안 된다.

### 6.1 OFF 상태의 동작

| 요청 | Decision 동작 | 운영 판정 |
| --- | --- | --- |
| `segment_ids`를 포함한 신규 run | lifecycle write 전에 `409` | 예상된 feature gate 차단이며 JSON 오류가 아님 |
| failed-only 자동 next-loop | analysis/generation/run write 전에 `409` | 예상된 feature gate 차단 |
| 수동 preparation 생성·활성화 | preparation/run write 전에 `409` | 예상된 feature gate 차단 |
| `segment_ids`를 생략한 run 요청 | 전체 generation scope로 처리 | 하위 호환 동작이며 부분 scope 실행이 아님 |

따라서 새 Dashboard를 먼저 배포해 모든 신규 요청에 `segment_ids`가 포함됐는데 Decision flag가 OFF이면, JSON 형식이 정상이어도 실행은 계속 `409`로 실패한다. 이 상태를 해결하기 위해 Dashboard의 JSON 직렬화를 다시 변경하지 않는다.

flag 차단 응답은 주로 `explicit segment scope is disabled until Dashboard scope lineage is ready` 또는 `partial segment scope is disabled until Dashboard scope lineage is ready` 메시지를 가진다. 운영 로그와 응답에서 이 문구를 먼저 확인한다.

### 6.2 ON 전 필수 게이트

다음 조건이 모두 충족되기 전에는 dev와 production 어디에서도 flag를 켜지 않는다.

1. Data Contract `expand → backfill → finalize`가 순서대로 완료됐다.
2. 기존 `uq_promotion_runs_loop`가 제거되고 전체 composite unique가 적용됐다.
3. 모든 기존 run의 canonical scope·fingerprint와 target/fallback 실험 구성이 검증됐다.
4. scope 멱등성과 preparation/lineage가 통합된 동일 Decision image가 대상 환경의 모든 ECS task에 배포됐다.
5. Dashboard가 일반 신규 요청에 `segment_ids` 정확히 1개를 항상 보내고, 응답의 `segment_ids`와 `is_fallback`을 필수 파싱한다.
6. Dashboard가 next-loop의 복수 scope와 동일 scope `200` 재사용을 처리한다.
7. Dashboard/Advertisement 계약 테스트와 dispatch 멱등 재시도 테스트가 통과했다.

`finalize` 전에 flag를 켜면 기존 `(promotion_id, loop_count)` unique 때문에 같은 loop의 다른 scope가 충돌할 수 있다. Dashboard가 `segment_ids`를 실수로 생략하면 Decision은 하위 호환 규칙에 따라 전체 generation scope run을 만들 수 있으므로, Dashboard 요청 스키마에서 누락을 차단해야 한다.

### 6.3 ECS task 일관성

같은 ECS service의 모든 task는 다음 두 값이 동일해야 한다.

- 배포된 Decision image/revision
- `LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED`

일부 task만 ON이면 같은 요청이 라우팅된 task에 따라 `200` 또는 `409`가 되는 간헐적 장애가 발생한다. rolling deployment 중에도 구·신 task의 flag를 다르게 두지 않으며, flag 변경은 service 전체에 새 task definition을 배포해 일괄 반영한다. 새 task가 모두 healthy가 되고 이전 task가 모두 종료됐는지 확인한 뒤 smoke test를 시작한다.

### 6.4 활성화 순서

아래 순서를 바꾸지 않는다.

```text
Data Contract expand
→ Decision dual-write 배포, 모든 task flag OFF
→ 기존 run backfill 및 무결성 검증
→ finalize 및 composite unique 적용
→ Dashboard의 segment_ids·is_fallback 파싱 반영
→ 모든 dev task에서 flag ON
→ dev smoke test
→ Dashboard/Advertisement 연동 및 dispatch 재시도 확인
→ 운영 전체 task에서 flag 일괄 ON
```

dev smoke test가 실패하면 일부 task만 OFF로 되돌리지 않는다. dev service 전체를 동일한 flag 값으로 복구하고, 실패한 gate를 수정한 뒤 처음부터 task 일관성을 다시 확인한다. 운영 ON은 dev 증빙과 Dashboard 담당자의 반영 확인이 모두 남은 뒤에만 진행한다.

### 6.5 409·500 장애 분류

| 증상 | 우선 확인할 원인 | 조치 |
| --- | --- | --- |
| 신규 `segment_ids` 요청이 항상 `409` | Decision flag OFF | JSON을 수정하지 말고 배포 게이트와 전체 task flag 확인 |
| 같은 요청이 `200`과 `409`를 오감 | ECS task별 flag/image 불일치 | service 전체 task definition과 running task 교체 상태 확인 |
| ON 이후 같은 loop의 다른 scope가 충돌 | finalize 미적용 또는 기존 unique 잔존 | DB constraint metadata 확인, 운영 실행 중단 |
| 기존 run 재사용만 `409` | scope·fingerprint·target/fallback 또는 preparation lineage 손상 | 해당 run 무결성 조사, 신규 run으로 조용히 우회하지 않음 |
| `Token "("` JSONB `500` | 구버전 Decision image 또는 repository JSONB 변환 문제 | 8장의 런타임/코드 확인 절차 수행 |
| 의도보다 많은 target 실험 생성 | Dashboard의 `segment_ids` 누락 | 요청 스키마와 실제 payload 로그 확인, 전체 scope run 실행 중단 |

## 7. 로컬 통합 테스트 수용 기준

Decision 서버는 다음 조건으로 새로 실행해야 한다.

- PR #230과 최신 `dev`의 PR #232가 통합된 전달 커밋인지 확인한다.
- Data Contract finalize와 Dashboard 계약 반영이 끝난 dev 환경인지 확인한다.
- 모든 dev ECS task에 동일한 Decision image와 `LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED=true`가 적용됐는지 확인한다.
- 기존 8081 프로세스 또는 컨테이너를 종료하고 이미지를 재빌드한 뒤 재시작한다.

수용 기준:

1. 선택 세그먼트 1개로 run 생성 시 `200`을 받고, 응답의 `segment_ids`는 선택 세그먼트 1개다.
2. 응답에는 선택 실험 1개와 `is_fallback: true`인 fallback 실험 1개가 있다.
3. 동일 요청 반복 시 `409`가 아니라 동일 scope의 기존 run을 `200`으로 재사용한다.
4. fallback 배정이 발생한 실행 후 선택 실험과 fallback 실험이 모두 serving 가능한 상태다.
5. `active_ad_serving_assignments`에서 fallback 배정도 조회된다.
6. email/SMS인 경우 필요한 실험 시작이 모두 성공한 뒤 dispatch가 한 번 실행된다.
7. 선택하지 않은 다른 비-fallback 세그먼트 실험은 생성·시작되지 않는다.
8. failed-only 자동 next-loop가 복수 실패 세그먼트로 생성된다.
9. 수동 preparation 생성·활성화가 `409`로 차단되지 않는다.
10. ECS 배포 상태에서 모든 running task가 동일한 task definition revision을 사용하고, 해당 revision의 flag가 `true`임을 확인한다.

## 8. JSON 500을 분리해서 판정하는 방법

오류가 다음 형태라면 대시보드 요청 JSON 문제가 아니다.

```text
invalid input syntax for type json
Token "("
```

이 오류는 PostgreSQL에 JSON 배열이 아니라 Python tuple 문자열 표현이 전달됐다는 뜻이다. Decision 기준 커밋의 repository에는 다음 변환이 있어야 한다.

```py
Jsonb(list(run.segment_scope_json))
```

따라서 동일 오류가 계속되면 대시보드 코드를 수정하기 전에 실제 8081 프로세스가 최신 Decision 코드를 실행하는지 확인한다.

```bash
git fetch origin
git branch --show-current
git rev-parse --short HEAD
git rev-parse --short origin/fix/seg-ad-exp
git status --short
rg -n 'Jsonb\(list\(run.segment_scope_json\)\)' app/decision/repositories.py
lsof -nP -iTCP:8081 -sTCP:LISTEN
docker ps --filter publish=8081 --format 'table {{.ID}}\t{{.Image}}\t{{.Ports}}\t{{.Names}}'
```

기준은 전달받은 통합 브랜치와 커밋, clean worktree, `Jsonb(list(...))` 코드 존재다. 저장소가 최신이어도 실행 중인 컨테이너 내부 코드가 다르면 반드시 재빌드·재시작한다.

## 9. 완료 정의

다음 조건을 모두 만족하면 대시보드 수정이 완료된 것으로 본다.

- Decision의 `segment_ids`와 `is_fallback`이 API server에서 web client까지 손실 없이 전달된다.
- 신규 run, 기존 run 재사용, 다음 루프 모두 동일한 fallback 실행 규칙을 사용한다.
- fallback 배정이 있는 경우 fallback 실험이 `planned`에 남지 않는다.
- fallback 실행 실패 시 dispatch하지 않고 UI가 실패를 알 수 있다.
- 다른 일반 세그먼트는 시작하지 않는다.
- API 계약 테스트, 실행 흐름 테스트, 전체 테스트, typecheck가 통과한다.
- 최신 Decision 런타임을 사용한 로컬 통합 테스트에서 JSON 500이 재현되지 않는다.
- dispatch 실패 후 같은 멱등 키 재시도와 접수·완료 작업의 미재발송이 검증된다.
- 모든 dev ECS task의 Decision image와 flag 값이 동일하다.
- `409`가 JSON 오류, feature gate 차단, 기존 run 무결성 실패 중 무엇인지 응답·로그 기준으로 분류할 수 있다.
- 위 증빙 전에는 운영 `LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED`를 켜지 않는다.
- 운영 ON은 전체 task에 일괄 적용하고, 일부 task만 다른 flag 값으로 남기지 않는다.
