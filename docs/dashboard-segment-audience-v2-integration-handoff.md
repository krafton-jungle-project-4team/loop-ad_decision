# Dashboard Segment Audience V2 연동 Handoff

## 1. 문서 목적

Decision PR [#288](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/288)의 Segment Audience V2 결과를 Dashboard의 세그먼트 추천 카드, 확정, 콘텐츠 생성, 카드별 실험 실행 흐름에 연결하기 위한 구현 계약입니다.

이 변경에서 Dashboard는 사용자군을 계산하지 않습니다.

```text
Decision
→ 후보별 source snapshot 계산
→ 선택 조합 allocation
→ final snapshot 생성
→ run/assignment에서 final snapshot 재사용

Dashboard
→ 저장된 source/final 수치 조회
→ 현재 선택 조합 preview 표시
→ 확정 요청에 선택 segment_id 전달
→ 카드 실행 시 정확한 analysis/generation/segment 전달
```

기준:

- Decision: PR #288, `feat/new64d`
- Data Contract: PR [#25](https://github.com/krafton-jungle-project-4team/loop-ad_data-source_contract/pull/25)가 먼저 머지됐다고 가정
- Dashboard 확인 기준: `origin/main@cb0fc25`
- Dashboard와 Data Contract에 환경변수나 기능 스위치를 추가하지 않음

## 2. 최종 사용자 여정

```text
세그먼트 후보 추천받기
→ Decision이 후보별 source snapshot 생성
→ Dashboard가 source snapshot 수치로 카드 표시
→ 운영자가 후보 1~3개 선택
→ Dashboard가 현재 선택 조합 preview 표시
→ 운영자가 확정
→ Decision /analyses가 final allocation snapshot 생성 및 사용자 예약
→ Dashboard가 확정 결과와 suggestion 상태 저장
→ 확정 analysis 기준 콘텐츠 생성·승인
→ 각 확정 카드에서 광고 소재·실험 실행
→ 현재 카드 하나의 final snapshot으로 run 생성
→ assignment는 snapshot 회원만 재사용
```

핵심 단위:

```text
source snapshot
= 후보 하나를 단독 선택했을 때의 독립 사용자군
= 후보끼리 사용자 중복 허용

allocation plan
= 한 번의 확정에서 선택한 1~3개 후보 사이의 사용자 배분 단위

final snapshot
= allocation plan에서 중복 해소 후 해당 세그먼트에 고정된 사용자군

promotion run
= 확정 카드 하나를 실제 실행하는 단위
```

A+B를 함께 확정해도 A와 B를 한 run에 자동으로 합치지 않습니다.

```text
A+B 확정
→ A/B final snapshot은 서로 겹치지 않음

A 카드 실행
→ A analysis/generation/final snapshot만 사용

B 카드 실행
→ B analysis/generation/final snapshot으로 별도 run 생성
```

## 3. 추천 카드 조회 계약

### 4.1 기존 카드 수치는 Decision 값을 그대로 사용

Decision은 기존 `promotion_segment_suggestions.metadata_json.display_copy.audience` 필드에 source snapshot 수치를 투영합니다.

```json
{
  "display_copy": {
    "audience": {
      "total_eligible_user_count": 613,
      "matching_user_count": 230,
      "selected_user_count": 180,
      "matching_user_ratio": 0.375204,
      "selected_user_ratio": 0.293638,
      "selection_ratio_within_matching": 0.782609,
      "selection_limited": true,
      "selection_basis": "hard_predicate_and_exact_cosine",
      "selected_user_role": "final_experiment_audience"
    }
  }
}
```

Dashboard는 다음 값을 다시 계산하면 안 됩니다.

| 카드 표시 | 읽을 값 | 의미 |
|---|---|---|
| 분석 가능 사용자 | `total_eligible_user_count` | exclusion 적용 후 active vector generation의 vector-valid 모집단 |
| 행동 조건 부합 | `matching_user_count` | 같은 모집단에서 hard predicate를 통과한 사용자 수 |
| 단독 대상 사용자 | `selected_user_count` | 후보를 단독 선택했을 때 hard predicate와 exact cosine threshold를 통과한 source snapshot 회원 수 |

기존 `대표 표본` 라벨은 제거합니다. allocation preview까지 함께 노출하는 최종 화면에서는 source 값의 의미가 분명하도록 `단독 대상 사용자`를 권장합니다.

화면에서 `audience_summary` 문자열을 파싱해 숫자를 만들지 말고 구조화된 `display_copy.audience`를 사용합니다.

### 4.2 suggestion read model에 source snapshot ID 추가

`ListDashboardPromotionSegmentSuggestions`에 아래 필드를 추가합니다.

```sql
pss.audience_snapshot_id AS "audienceSnapshotId"
```

공유 Zod schema와 mapper에도 nullable 필드를 추가합니다.

```ts
audience_snapshot_id: z.string().nullable()
```

Legacy suggestion은 `NULL`을 허용해야 합니다.

### 4.3 audience schema의 additive 필드 보존

현재 Dashboard normalizer는 일부 필드만 보존합니다. 아래 필드를 optional로 추가하는 것을 권장합니다.

```ts
matching_user_ratio?: number
selection_ratio_within_matching?: number
selection_basis?: "hard_predicate_and_exact_cosine"
selected_user_role?: "final_experiment_audience"
```

Legacy metadata에는 이 값들이 없을 수 있으므로 optional이어야 합니다.

## 4. 현재 선택 조합 allocation preview

Decision은 추천 analysis의 다음 위치에 canonical preview를 저장합니다.

```text
promotion_analyses.output_json.audience_allocation_preview_context
```

형식:

```json
{
  "preview_version": "audience_allocation_preview.v1",
  "candidate_batch_analysis_id": "analysis_recommendation_001",
  "candidate_segment_ids": ["seg_a", "seg_b", "seg_c"],
  "exclusion_revision": 12,
  "allocation_policy_version": "hotel_segment_allocation.v1",
  "allocation_policy_hash": "sha256:...",
  "allocation_previews": [
    {
      "selected_segment_ids": ["seg_a", "seg_b"],
      "candidate_batch_analysis_id": "analysis_recommendation_001",
      "exclusion_revision": 12,
      "preview_version": "audience_allocation_preview.v1",
      "allocation_policy_version": "hotel_segment_allocation.v1",
      "allocation_policy_hash": "sha256:...",
      "total_allocated_user_count": 200,
      "per_segment": [
        {
          "segment_id": "seg_a",
          "allocated_user_count": 95,
          "targetable": true,
          "meets_min_sample_size": true,
          "audience_status": "targetable"
        },
        {
          "segment_id": "seg_b",
          "allocated_user_count": 105,
          "targetable": true,
          "meets_min_sample_size": false,
          "audience_status": "insufficient_sample"
        }
      ]
    }
  ]
}
```

Dashboard 구현 규칙:

1. 현재 체크된 `segment_id`를 중복 제거 후 오름차순 정렬합니다.
2. `allocation_previews[].selected_segment_ids`가 정확히 같은 항목 하나를 찾습니다.
3. 선택된 카드에만 `현재 선택 기준 최종 배정 사용자`를 표시합니다.
4. 미선택 카드는 source snapshot의 `단독 대상 사용자`만 표시합니다.
5. 모든 조합을 표나 카드에 나열하지 않습니다.
6. `candidate_batch_analysis_id`, `preview_version`, `exclusion_revision`이 현재 context와 일치하지 않으면 preview를 사용하지 않고 refetch합니다.
7. Dashboard가 selection fingerprint나 allocation 결과를 직접 계산하지 않습니다.

예시:

```text
현재 선택: A+B

A 카드 — 선택됨
- 단독 대상 사용자: 100명
- 현재 선택 기준 최종 배정 사용자: 85명

B 카드 — 선택됨
- 단독 대상 사용자: 120명
- 현재 선택 기준 최종 배정 사용자: 105명

C 카드 — 미선택
- 단독 대상 사용자: 80명
- allocation 수치는 표시하지 않음
```

추천 카드가 3개면 Decision이 7개 조합을 저장하지만, Dashboard는 현재 선택 조합 하나만 읽습니다.

### Preview 조회 구현 권장안

`ListDashboardPromotionSegmentSuggestions`가 사용하는 recommendation `analysis_id`로 `promotion_analyses.output_json`을 함께 읽고, suggestion list 응답에 top-level nullable context로 노출합니다.

```ts
{
  suggestions: [...],
  audience_allocation_preview_context: PreviewContext | null
}
```

preview 전체를 suggestion마다 반복해서 복제하지 않는 것을 권장합니다.

## 5. 세그먼트 확정 흐름 변경

### 6.1 현재 direct confirm SQL을 V2에 그대로 사용하면 안 됨

현재 `ConfirmDashboardPromotionSegmentSuggestions` SQL은:

- promotion 전체의 `accepted` suggestion을 조회하고
- active manual segment를 자동 합산하고
- Dashboard가 만든 manual analysis에 target/vector를 직접 INSERT합니다.

V2에서는 이 방식이 final snapshot과 reservation을 만들지 않으므로 사용할 수 없습니다.

### 6.2 V2 권장 순서

Dashboard 공개 confirm 요청 DTO는 유지할 수 있습니다. API server가 현재 recommendation analysis의 `accepted` V2 suggestion을 조회해 segment ID 목록을 만듭니다.

```text
1. 현재 화면의 recommendation analysis_id 확정
2. 해당 analysis에 속한 accepted V2 suggestion 1~3개 조회
3. 다른 analysis, legacy, manual/ChatKit 혼합 여부 검증
4. Decision POST /decision/v1/promotions/{promotion_id}/analyses 호출
5. Decision이 confirmation analysis + allocation plan + final snapshot + reservation 생성
6. 반환된 confirmation analysis_id의 target row에 suggestion_id/confirmed_by/confirmed_at 보강
7. 선택 suggestion만 confirmed로 변경
8. suggestion, analysis, target, preview 관련 query invalidate/refetch
```

Decision 요청:

```http
POST /decision/v1/promotions/{promotion_id}/analyses
Content-Type: application/json
```

```json
{
  "project_id": "demo_project",
  "campaign_id": "campaign_001",
  "promotion_id": "promotion_001",
  "segment_ids": ["seg_a", "seg_b"],
  "operator_instruction": null
}
```

응답에서 보존할 값:

```json
{
  "analysis_id": "analysis_confirmation_001",
  "promotion_id": "promotion_001",
  "status": "completed",
  "target_segments": [
    {
      "segment_id": "seg_a",
      "audience_snapshot_id": "snapshot_final_a",
      "final_audience_count": 85,
      "meets_min_sample_size": true,
      "targetable": true,
      "audience_status": "targetable"
    }
  ]
}
```

Dashboard Decision client의 analysis response Zod schema가 현재 `analysis_id`, `promotion_id`, `status`만 보존하므로 `target_segments`를 additive하게 파싱해야 합니다.

### 6.3 확정 SQL 변경 원칙

V2에서 Dashboard SQL은 target/vector를 새로 만들지 않습니다. Decision이 만든 confirmation target을 보강하고 suggestion 상태를 확정하는 역할만 합니다.

반드시 지킬 조건:

- update 대상 target의 `analysis_id`는 Decision이 반환한 confirmation analysis ID
- source suggestion은 현재 recommendation analysis의 선택 항목만 허용
- `audience_snapshot_id`, `allocation_plan_id`, `audience_reservation_state`를 Dashboard 값으로 덮어쓰지 않음
- `rule_json`, `segment_vector_id`, `estimated_size`를 이전 suggestion 값으로 되돌리지 않음
- active manual/ChatKit segment를 V2 확정에 자동 합산하지 않음
- 모든 historical accepted suggestion을 promotion 단위로 한꺼번에 확정하지 않음

V2 AI와 manual/ChatKit을 같은 확정 요청에 섞지 않습니다. manual/ChatKit은 기존 legacy 흐름을 별도로 유지합니다.

### 6.4 재시도

Decision의 confirmation analysis/allocation은 동일 입력에 대해 멱등적입니다. Dashboard DB update가 실패하면 동일 선택으로 Decision 요청부터 재시도한 뒤 미완료 Dashboard update를 완료할 수 있어야 합니다.

다음 오류를 사용자에게 일반적인 “추천 실패”로 숨기지 않습니다.

| HTTP/code | Dashboard 처리 |
|---|---|
| `409 segment_audience_source_batch_mismatch` | 다른 추천 회차 후보가 섞였음을 표시하고 최신 후보 refetch |
| `409 segment_audience_source_already_confirmed` | 이미 다른 확정에 사용된 후보임을 표시하고 refetch |
| `409 segment_audience_segment_already_confirmed` | 같은 promotion의 활성 확정 세그먼트임을 표시 |
| `409 segment_audience_allocation_empty` | 현재 조합에서 특정 세그먼트의 최종 배정 사용자가 0명임을 표시하고 확정 차단 |
| `409 segment_audience_exclusion_conflict` | 동시 확정 충돌. 최신 추천/preview refetch 후 재시도 안내 |
| `409 segment_audience_exclusion_projection_not_ready` | exclusion projection 동기화 대기 후 재시도 안내 |
| `422 segment_audience_exclusion_contract_missing` | Data Contract/배포 순서 오류. 운영 오류로 표시 |
| 기타 구조화된 422 | `detail.code`, `detail.segment_id`, `detail.reason` 표시 |

구조화된 오류 형식:

```json
{
  "detail": {
    "code": "segment_audience_source_batch_mismatch",
    "promotion_id": "promotion_001",
    "segment_id": "seg_a",
    "reason": "..."
  }
}
```

## 6. 확정 세그먼트 read model

`ListDashboardCampaignSegments`, `ListDashboardPromotionSegments`, `GetDashboardPromotionSegment`에 V2 binding을 additive하게 노출합니다.

필수 target 필드:

```text
pts.analysis_id
pts.audience_snapshot_id
pts.allocation_plan_id
pts.audience_reservation_state
```

final snapshot join 권장 필드:

```text
snapshot.snapshot_kind
snapshot.final_user_count
snapshot.min_sample_size
snapshot.meets_min_sample_size
snapshot.audience_status
snapshot.status
snapshot.source_snapshot_id
```

V2 확정 카드에서 표시할 실제 사용자 수는 final snapshot의 `final_user_count`입니다. `segment_definitions.sample_size`나 source suggestion의 `selected_user_count`로 되돌아가면 안 됩니다.

Legacy target은 snapshot 관련 필드가 모두 `NULL`일 수 있으므로 read schema는 nullable이어야 합니다.

## 7. 콘텐츠 생성 계약

확정 성공 후 콘텐츠 생성은 Decision이 반환한 confirmation `analysis_id`를 사용합니다.

```json
{
  "analysis_id": "analysis_confirmation_001",
  "content_option_count": 3,
  "operator_instruction": null
}
```

A+B를 같이 확정했다면 같은 confirmation analysis에서 A/B 콘텐츠를 생성할 수 있습니다.

사용자군 allocation이 바뀌었다는 이유만으로 기존 승인 콘텐츠를 자동 폐기하지 않습니다. 다만 run 생성 시 현재 카드 segment에 대해 요청 `generation_id` 안에 approved/active 콘텐츠가 정확히 하나 있어야 합니다.

## 8. 카드별 run 요청

Dashboard의 현재 `segment_ids.length(1)` 계약은 유지합니다.

A 카드의 `광고 소재·실험` 실행 요청:

```http
POST /dashboard/v1/promotions/{promotion_id}/runs
```

```json
{
  "analysis_id": "analysis_confirmation_001",
  "generation_id": "generation_confirmation_001",
  "segment_ids": ["seg_a"],
  "loop_count": 1
}
```

중요:

- `analysis_id`: 현재 카드가 속한 confirmation target의 값
- `generation_id`: 해당 confirmation analysis와 카드 콘텐츠가 속한 값
- `segment_ids`: 현재 클릭한 카드 하나만 전달
- promotion의 latest analysis/generation으로 대체하지 않음
- 같은 allocation plan의 다른 카드 B를 자동 포함하지 않음
- 다른 analysis/generation의 target을 한 run에 섞지 않음

V2 요청에서 세 출처 필드가 없거나 맞지 않으면 Decision이 다음 오류를 반환합니다.

```text
409 segment_audience_run_source_required
409 segment_audience_run_source_mismatch
409 segment_audience_target_already_run_bound
409 segment_audience_snapshot_binding_missing
409 segment_audience_snapshot_invalid
409 segment_audience_snapshot_semantic_mismatch
409 segment_audience_not_targetable
```

기존 Dashboard 카드별 run 생성 형태는 이미 방향이 맞습니다. 핵심은 값의 출처를 latest promotion 상태가 아니라 현재 카드의 confirmation target/content로 고정하는 것입니다.

## 9. Assignment 응답 처리

V2 assignment는 새 검색을 하지 않고 final snapshot 회원을 재사용합니다.

```json
{
  "matching_mode": "analysis_snapshot_reuse",
  "ann_candidate_limit": 0,
  "ann_candidate_count": 0,
  "exact_reranked_pair_count": 0,
  "ann_applied": false,
  "ann_not_applied_reason": "analysis_snapshot_reuse",
  "assignment_count": 85,
  "status": "completed"
}
```

Dashboard는 ANN 진단 값이 0이라는 이유로 실패로 처리하면 안 됩니다. V2에서는 `matching_mode == "analysis_snapshot_reuse"`와 `assignment_count`가 정상 결과입니다.

assignment 단계에서 Dashboard가 source snapshot 회원을 다시 합치거나 overlap을 계산하면 안 됩니다.

## 10. 삭제·release UI 경계

현재 Dashboard의 `StopDashboardPromotionTargetSegment`는 관련 row를 물리 삭제하는 legacy SQL입니다. V2 target에 그대로 적용하면 안 됩니다.

현재 범위 권장 정책:

- V2 target이고 run binding이 있으면 삭제/release 차단
- A+B 같은 다중 확정 plan에서 A만 부분 release 차단
- V2 release 전용 Decision public API가 제공되기 전까지 V2 삭제 버튼 비활성화
- manual/legacy 삭제 흐름은 기존대로 유지

향후 release API가 연결되면:

```text
단일 target, run 전
→ 전체 reservation release 가능

다중 target plan, run 전
→ plan 전체 release만 가능

하나라도 run binding/consumed
→ release 불가
```

## 11. 추천 재실행과 freshness

확정된 final 사용자는 promotion exclusion ledger에 `reserved`로 기록됩니다. 같은 promotion에서 추천을 다시 실행하면 해당 사용자는 새 source snapshot의 모든 수치와 회원에서 제외됩니다.

Dashboard는 이전 추천 analysis의 preview를 새 추천에 재사용하면 안 됩니다.

확정 후 같은 추천 화면이 남아 있다면 Decision이 갱신한 현재 recommendation analysis의 preview를 refetch합니다.

```text
확정 전 revision 12
→ A 확정
→ revision 13
→ 남은 B/C/B+C preview 갱신
→ Dashboard는 revision 13만 사용
```

## 12. 변경 예상 파일

Dashboard `origin/main` 기준 예상 범위입니다.

### Shared schema

- `packages/shared/src/dashboard/segment.ts`
- `packages/shared/src/dashboard/promotion-run.ts` — 기존 카드별 1개 segment run 계약 유지 확인

### API server

- `apps/api-server/src/features/dashboard/provider/dashboard-decision-client.ts`
- `apps/api-server/src/features/dashboard/service/dashboard-query.service.ts`
- `apps/api-server/src/features/dashboard/repository/dashboard-campaign-reader.ts`
- `apps/api-server/src/features/dashboard/repository/dashboard-campaign-mappers.ts`
- `apps/api-server/src/features/dashboard/database/dashboard.sql`
- pgtyped generated query — SQL 수정 후 재생성

### Web client

- `apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/components/PromotionSegmentSuggestions.tsx`
- `apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/usePromotionWorkspaceController.ts`
- 확정 세그먼트 카드와 run 실행에 analysis/generation을 전달하는 관련 component/model

## 13. 필수 테스트

### API/contract

- suggestion nullable `audience_snapshot_id` parsing
- legacy suggestion의 기존 metadata parsing
- preview context version/revision/list parsing
- Decision `/analyses` response의 `target_segments` parsing
- V2 confirm이 현재 analysis의 accepted suggestion만 전달
- V2 confirm이 manual/ChatKit segment를 자동 합산하지 않음
- Decision 성공 후 Dashboard target 보강과 suggestion confirmed 처리
- 동일 confirm 재시도 멱등성
- 구조화된 409/422 오류 전달

### UI

- source 카드의 세 숫자가 Decision metadata와 일치
- `대표 표본`이 노출되지 않음
- A+B 선택 시 A/B에만 현재 선택 기준 final count 표시
- C 미선택 시 C allocation count 미표시
- C 선택 시 A+B+C preview 하나로 세 카드가 함께 갱신
- stale revision preview 미사용 및 refetch
- `targetable=false` 조합 확정 차단
- `insufficient_sample`은 선택/실험 가능하되 평가 불충분 안내

### 전체 여정

```text
추천
→ 카드 source 수치 확인
→ A+B 선택 preview 확인
→ 확정
→ A/B final snapshot count 확인
→ 콘텐츠 생성·승인
→ A 카드 run/assignment
→ B 카드 별도 run/assignment
```

검증 사항:

- A/B final snapshot 회원이 서로 겹치지 않음
- A run이 B를 자동 포함하지 않음
- assignment count가 각 final snapshot member count와 일치
- 새 추천이 A/B reserved/consumed 사용자를 포함하지 않음
- legacy 추천/확정/run 흐름 회귀 없음

## 14. 머지·배포 순서

```text
1. Data Contract PR #25 머지·배포
2. Decision PR #288 머지·배포
3. vector generation / semantic artifact / exclusion projection 준비
4. Dashboard additive reader + V2 confirm orchestration + UI 라벨/preview 배포
5. dev 단일 promotion 전체 여정 smoke test
```

Dashboard가 먼저 배포되더라도 legacy nullable 값을 읽을 수 있어야 합니다. Decision V2를 Data Contract보다 먼저 활성화하면 안 됩니다.

## 15. 완료 기준

- 카드의 세 숫자가 source snapshot과 일치
- 현재 선택 조합의 final allocation count만 선택 카드에 표시
- confirm이 Decision `/analyses`를 거쳐 final snapshot과 reservation 생성
- confirmed target read model이 final snapshot 수치를 사용
- 콘텐츠 생성과 run이 confirmation analysis/generation을 사용
- 카드 run은 현재 segment 하나만 포함
- assignment가 `analysis_snapshot_reuse`로 완료
- 같은 promotion의 다음 추천에서 이전 reserved/consumed 사용자 제외
- legacy 데이터와 manual/ChatKit 흐름 유지

## 16. 변경하지 않는 것

- Dashboard에서 ANN, cosine, raw predicate 계산
- Dashboard에서 snapshot member를 직접 재배분
- 후보 ontology·순위·segment ID 생성
- 64차원 schema
- ClickHouse HNSW
- 신규 환경변수·기능 스위치·별도 요청 모드
- promotion 전체 확정 target을 한 run에 자동 합산하는 동작
