# Decision 개인정보 데이터 경계 PoC

## 문서 상태

이 문서는 Private Connector를 도입할 때 Decision이 지켜야 할 데이터 경계와 현재
코드의 차이를 정리한 설계안이다. 이 브랜치에서 Decision 저장소의 실행 코드는
변경하지 않으며, 개인정보 보호 구조가 운영에 적용됐다고 주장하지 않는다.

## 목표 구조

```text
고객사 네트워크
  원본 고객 ID
  고객 DB
  동의 원장
  HMAC 비밀키
       |
       | 허용 속성 + subject_id + 동의 증적
       v
LoopAd 데이터 평면
  가명 이벤트
  행동 벡터
  audience snapshot
  treatment/control unit
  Uplift dataset
       |
       | 프로모션 문맥 + 집계 근거
       v
외부 AI provider
  사용자별 이벤트, 식별자, 벡터를 받지 않음
```

고객사 원본 고객 ID와 HMAC 비밀키는 LoopAd 경계를 넘지 않는다. 고객사
Connector가 `privacy-event.v2`의 `subject_id`를 만든 뒤 허용된 행동 속성만
전송한다.

## 현재 코드와의 차이

현재 Decision과 Data Contract의 분석 키 이름은 대부분 `user_id`다. Uplift
`UpliftTrainingExample`도 `user_id`, 배정 전 64차원 특징 `features`, treatment,
outcome을 내부 학습 데이터로 보유한다.

Private Connector를 실제 연결하려면 다음 중간 단계가 더 필요하다.

1. Collector 이후 ingestion이 `subject_id`를 분석용 주체 키로 저장한다.
2. ClickHouse raw event, 행동 벡터, audience snapshot, assignment가 같은 가명 키를
   사용한다.
3. 기존 `user_id` 컬럼을 곧바로 제거하지 않고, 값의 의미가 가명 주체임을 계약과
   migration으로 명시한다.
4. 원본 ID와 가명 ID를 역매핑하는 테이블은 LoopAd에 만들지 않는다.
5. 키 회전 시 namespace와 key version별 재연결 또는 단절 정책을 정한다.

이 연결 작업은 현재 PoC 범위에 포함되지 않는다.

## Decision 내부 처리

가명 주체 단위 처리가 필요한 영역:

```text
raw event audience membership
user_behavior_vector_search_generations
segment_audience_snapshot_members
ad_experiment_units
user_segment_assignments
Uplift X/T/Y dataset
outcome matching
```

이 영역은 고객별 계산을 위해 가명 키를 사용할 수 있지만, 로그나 API 응답에 전체
키를 노출하지 않는다. 운영 추적에는 snapshot ID, generation ID, experiment ID,
건수와 계약 hash를 사용한다.

## 외부 AI 호출 경계

현재 광고 생성 prompt는 프로모션 문맥과 다음 세그먼트 집계 정보를 사용한다.

```text
segment_id와 표시 이름
예상 인원
프로모션 목표
검증된 행동 신호 요약
조건 및 SQL 요약
광고 문구 작성 지침
```

다음 데이터는 외부 AI provider에 전달하지 않는 계약으로 고정해야 한다.

```text
subject_id 또는 기존 user_id 목록
개별 raw event
개별 행동 벡터
개별 treatment/control arm
개별 예약 outcome
원본 SQL 결과 행
고객사 DB metadata와 credential
동의 원문
```

실제 운영 전에는 provider 호출 직전 payload validator를 추가해 식별자 배열,
64차원 벡터, 이벤트 원문을 거절해야 한다. 작은 집계가 개인을 쉽게 드러낼 수 있는
경우를 위해 최소 집계 인원과 속성 allowlist도 별도 정책 버전으로 관리해야 한다.

LLM은 프로모션 문장을 구조화하고 광고 설명을 생성할 수 있지만, 고객 membership과
인원은 서버가 실행 가능한 조건으로 계산한다. LLM이 사용자별 행이나 자유 SQL에
접근하는 구조로 확장하지 않는다.

## Uplift 학습 경계

Uplift 학습은 Decision 내부 또는 고객사 전용 계산 환경에서 다음 값으로 수행한다.

```text
X = 배정 이전 가명 사용자의 행동 특징
T = treatment/control ITT arm
Y = immutable outcome spec과 일치한 결과
```

Model artifact에는 학습 row, subject ID, experiment unit ID를 포함하지 않는다.
Dataset manifest에는 계약 hash, experiment 집합 fingerprint, 표본 수와 cutoff만
남긴다. 학습 중간 파일이 필요하면 project별 암호화 저장소, 짧은 보존 기간,
접근 감사 로그가 필요하다.

현재 LoopAd Uplift 모델은 운영 데이터로 검증되지 않았고 serving에 활성화되지
않는다. Criteo 외부 검증 artifact도 LoopAd 사용자를 대상으로 serving할 수 없다.

## 삭제와 동의 철회

고객사 Connector는 원본 ID로 같은 `subject_id`를 다시 계산해 삭제 요청을 만든다.
Decision은 다음 계보를 따라 삭제 또는 사용 제한 상태를 기록해야 한다.

```text
subject_id
→ raw events
→ behavior vectors
→ audience snapshot members
→ assignments와 experiment units
→ 미완료 training dataset
```

완료된 실험과 모델 artifact는 감사 또는 법적 보존 의무와 삭제권이 충돌할 수 있다.
따라서 무조건 물리 삭제하지 않고 `delete`, `restrict`, `retain` 결정을 근거와 함께
기록하는 정책 계층이 필요하다. 이 판단은 고객사와 법률 검토 없이 코드가 임의로
결정하지 않는다.

## 로깅

허용되는 추적 필드:

```text
requestId
projectId
promotionId
promotionRunId
segmentId
audienceSnapshotId
vectorGenerationId
adExperimentId
contractVersion
policyVersion
memberCount
durationMs
failureCode
```

로그 금지 항목:

```text
원본 고객 ID
subject_id 전체 값
개별 feature vector
개별 event properties
학습 row
provider prompt에 없는 내부 원문
```

## 발표에서 말할 수 있는 범위

- 고객사 내부에서 원본 식별자를 가명처리하는 Connector 경계를 코드로
  검증했다.
- SDK와 Collector가 선택적인 개인정보 보호 이벤트 계약을 처리하는 실험
  브랜치를 만들었다.
- Decision에서 외부 AI 호출과 사용자별 분석 데이터를 분리할 데이터 경계를
  설계했다.

말하면 안 되는 내용:

- 모든 고객사 DB adapter를 지원한다.
- 현재 운영 데이터가 이미 `subject_id`만 사용한다.
- 개인정보보호법 또는 다른 법률 준수를 보장한다.
- 삭제, 키 회전, tenant 격리와 접근 통제가 운영 검증을 마쳤다.

