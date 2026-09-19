# Run Contract Gate 검증 명세

| 항목 | 내용 |
|---|---|
| 대상 독자 | 테스트 구현자, 리뷰어 |
| 상태 | PR 0 개인 통합 병합 완료 · PR 1 로컬 구현·검증 완료 · PR 2 CI 미구현 |
| 기준 revision | 후보 기반 09442f29e8da514df1d1a5f2a52b03646c92e170 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |

## 실행 경계와 공통 fixture

실제 [router dependency](../../app/decision/router.py)의 connection 생성·commit·rollback·close를 사용한다. 정상 검사에서는 service·repository·DB connection을 fake로 대체하지 않는다. settings는 [load_settings](../../app/config.py)에 명시적인 합성 mapping을 전달하고 create_app(settings=...)를 사용한다. .env를 읽지 않으며 LOOPAD_ENV=test로 외부 worker 시작을 막는다.

기본 seed는 프로젝트·캠페인·프로모션, 완료한 analysis/generation, 승인 콘텐츠, V2 정의, 활성 vector generation, source/final snapshot과 synthetic member, allocation plan, 예약 상태 target/exclusion 관계를 준비한다. 현재 컴파일된 rule hash·threshold·member count·semantic metadata가 실제 source와 맞아야 한다. 구 integration fixture의 빈/간소화 rule을 검증 없이 복사하지 않는다.

각 case는 canonical DDL을 적용한 **자기 전용 DB**와 commit된 seed로 시작한다. 테스트 전체를 바깥 transaction으로 감싸 끝에 rollback하는 방식은 사용하지 않는다. 요청 transaction의 commit을 관찰해야 하기 때문이다. assertion은 요청 처리가 완전히 끝난 후 독립적인 connection에서 수행한다.

요청은 명시적 analysis_id·generation_id·segment_ids, loop_count=1, next_loop_preparation_id 없음으로 고정한다. 기본 데이터는 legacy가 아닌 현재 V2 계약이다. segment_audience.v1은 이 경로의 계약 식별자이고 hotel_behavior.v2는 별도의 스키마 버전이다.

## 필수 시나리오

RCG-01~09는 [test_run_db.py](../../tests/run_contract_gate/test_run_db.py)에 구현했다. 9개 논리 ID가 17개 실제 pytest case에 대응한다. CTRL-01~05는 30개 제어 case로 검증한다. 논리 ID는 이름이 바뀌어도 보존한다.

필수 목록: [DB manifest](../../tools/run_contract_gate/manifest.json), [제어 manifest](../../tools/run_contract_gate/control-manifest.json). 수집된 이름·각 setup/call/teardown 결과·JUnit을 대조한다.

| ID / 테스트 | 막으려는 문제 | 준비·요청·실패 주입 | 응답·DB 기대 결과 |
|---|---|---|---|
| RCG-01 / test_create_commits_exact_scope | HTTP 성공인데 데이터가 commit되지 않거나 선택 밖 실험이 생성됨 | A/B/C가 준비된 seed에서 A/B 요청 | HTTP 200, scope A/B, 각 1개 experiment, C·fallback 없음. 새 연결에서 run 1개 및 정확한 binding·locked/consumed 상태 확인 |
| RCG-02 / test_retry_reuses_committed_run | 응답을 잃은 후 재요청으로 중복 생성 | RCG-01과 같은 요청을 완료 후 새 요청으로 재전송 | 같은 run/experiment ID 집합. row/binding/exclusion 수 증가 없음. timestamp 전체 동등성은 요구하지 않음 |
| RCG-03 / test_scope_order_and_duplicates | 배열 순서·중복 때문에 identity가 바뀜 | A/B 생성 후 B/A/A 요청 | 같은 canonical scope·fingerprint·run/experiment. 응답 배열의 계약상 순서도 확인 |
| RCG-04 / test_disjoint_scopes_are_distinct | 서로 다른 정상 scope를 잘못 재사용 | 같은 analysis/generation에서 A 요청 후 B 요청. 서로 다른 target/member 사용 | 두 run ID가 다름. 각 scope·experiment·binding 정확. A/B를 동시에 요청한 결과와 혼동하지 않음 |
| RCG-05 / test_rejects_invalid_source_without_writes | 잘못된 source를 다른 카드/분석으로 보정 | 필수 source 누락, 타 analysis의 generation, 범위 밖/빈/공백/fallback segment를 각각 요청 | 해당 schema/service가 정의한 4xx와 오류 형태, 신규 run/experiment/binding 없음, seed 상태 보존 |
| RCG-06 / test_failure_after_real_writes_rolls_back | 일부 INSERT·소비 상태 변경 후 예외에서 부분 데이터 잔존 | 두 하위 case: 실제 experiment insert_many 직후, 실제 bind_run_targets 직후에 각각 테스트 wrapper가 예외 발생 | 오류 응답 또는 서버 예외, 신규 run/experiment/binding 0, target/plan/exclusion seed 상태 보존 |
| RCG-07 / test_deferred_binding_failure_rolls_back | statement 성공을 commit 성공으로 오인 | 테스트에서 binding 작업만 의도적으로 생략; 실제 run/experiment INSERT와 실제 dependency commit은 실행 | canonical deferred binding 제약으로 commit 실패. 새 연결에서 부분 row 없음. HTTP 성공이 관찰되면 PASS 금지 |
| RCG-08 / test_baseline_rows_are_reused | 후보 reader가 과거 writer의 row를 읽거나 재사용하지 못함 | 아래 provenance가 있는 고정 baseline fixture 복원 후 같은 요청 | 고정 expected run·experiment·scope·binding과 일치. 현재 writer로 fixture/expected를 재생성하지 않음 |
| RCG-09 / test_overlapping_target_binding_is_rejected | 이미 연결된 target을 다른 run에 다시 연결 | A로 run 생성 후 같은 analysis에서 A/B 요청 | 충돌 거절, 첫 run 불변, 두 번째 run/experiment 없음. UNIQUE(target_analysis_id,segment_id)의 의미 확인 |

RCG-01의 ID 검증은 production helper로 기대 ID를 다시 계산하는 것에만 의존하지 않는다. 응답·DB의 참조 관계와 scope cardinality를 직접 확인하고, 고정 baseline expected ID는 별도로 보존한다. scope fingerprint는 canonical 정렬·중복 제거 배열의 SHA-256과 비교한다.

RCG-05는 [기존 API 오류 테스트](../../tests/test_decision_run_api.py)와 실제 exception handler의 상태 코드를 기준으로 작성한다. 모든 invalid input을 무조건 같은 400/409로 묶지 않는다.

RCG-06/07의 실패 주입은 테스트에만 존재한다. production flag나 endpoint를 추가하지 않는다. 실제 쓰기를 fake 반환값으로 대체하지 않는다. RCG-07에서 FastAPI dependency teardown 예외가 client에 어떻게 나타나는지는 실제 실행으로 확인하고 오류 transport 형태를 기록한다. 어떠한 경우에도 성공 응답과 commit 실패 조합을 정상으로 승인하지 않는다.

각 case의 합격 조건은 응답과 독립 connection의 DB assertion이 모두 충족되는 것이다. 실패 원인 분류와 합격 여부는 별개다.

## 기존 row fixture

1. baseline SHA의 코드와 고정 DDL을 별도 임시 작업 공간·전용 DB에서 준비한다. 후보 코드를 baseline이라고 표기하지 않는다.
2. 합성 seed를 만들고 **baseline의 실제 run API**를 실행해 commit한다.
3. 해당 scope에 필요한 row 집합을 export한다. 운영 dump·실제 사용자 데이터는 사용하지 않는다.
4. provenance에는 producer SHA, Contract SHA, dependency lock hash, 이미지 digest, seed/생성 절차, 요청, expected identity, 각 fixture 파일 hash, 생성 시점, 비개인 합성 데이터 여부를 기록한다.
5. export를 새 빈 DB에 복원하고 baseline reader/API가 재사용하는지 확인한 뒤 fixture를 고정한다.
6. 평소 Gate에서는 candidate reader/API만 실행한다. 후보 writer로 expected artifact를 덮어쓰지 않는다.
7. 최신 lane에서도 같은 baseline fixture를 최신 fresh DDL에 복원한다. 구조 불일치는 최신 drift 경고이며 baseline fixture 자체를 자동 변경하지 않는다.

보존 위치: [rows.json](../../tests/fixtures/run_contract_gate/baseline/rows.json), [expected.json](../../tests/fixtures/run_contract_gate/baseline/expected.json), [provenance.json](../../tests/fixtures/run_contract_gate/baseline/provenance.json). canonical DDL은 복제하지 않는다.

INSERT lifecycle 제약 때문에 생성 전/후 행을 함께 보존한다. [복원 코드](../../tests/run_contract_gate/baseline.py)는 finalized/reserved 행을 먼저 복원·commit한 뒤, 저장된 baseline 변경분만 SQL로 적용한다. 트리거·제약을 비활성화하지 않으며 복원 결과가 committed 행과 전부 같은지 확인한다. DDL 자체의 `seg_existing_all` 기본 행은 export 대상에서 제외한다. 현재 writer를 호출하는 절차가 아니다.

이 검사는 **보존한 한 baseline의 생성 직후 run 상태**에 대한 호환성을 보장한다. 전체 legacy 데이터·모든 lifecycle 상태·DB migration의 안전성을 보장하지 않는다.

## Gate 자체의 제어 검증

구현: [test_runner_control.py](../../tests/run_contract_gate/test_runner_control.py). 아래 case는 실제 서비스 테스트와 별도로 runner의 잘못된 녹색 판정을 막는다.

| ID | 검증 |
|---|---|
| CTRL-01 | 필수 논리 case 하나가 수집되지 않거나 skip/xfail/예상하지 않은 xpass가 되면 fixed PASS 금지 |
| CTRL-02 | fixed FAIL 또는 INCOMPLETE와 latest PASS를 합쳐도 전체 exit 0이 되지 않음 |
| CTRL-03 | fixed PASS + latest 불일치/검증 불가에서 fixed는 PASS 유지, 전체 exit 0, latest 경고 및 원인 보존 |
| CTRL-04 | schema 준비/runner 실행 실패·timeout·중단에서 소유한 자원만 cleanup. 실패 중에도 가능한 결과 기록 |
| CTRL-05 | result.json/JUnit 누락·파싱 오류·lane SHA 누락·필수 결과 누락을 PASS로 처리하지 않음 |

필수 목록은 manifest에 명시하고 실제 수집·실행 결과와 대조한다. pytest 종료 코드 0이나 tests=0만으로 성공을 판정하지 않는다. 매개변수화된 RCG-05·RCG-06의 모든 하위 case도 필수다.

## 결과 판정

| lane | 상태 | 의미 | 전체 종료·CI |
|---|---|---|---|
| fixed | PASS | 필수 case 전부 실행·통과, 입력·결과·정리 기록 유효 | 다른 필수 제어 검증도 통과하면 exit 0 |
| fixed | FAIL | assertion 또는 계약 불일치 확인 | exit 1, 필수 check 실패 |
| fixed | INCOMPLETE | 환경·취득·fixture 준비·skip·누락·timeout·결과 손상으로 결론 불가 | exit 2, 필수 check 실패 |
| latest | PASS | 해석한 최신 SHA에 대해 전부 실행·통과 | fixed 결과 유지 |
| latest | WARN_DRIFT | 유효한 입력으로 DDL/row/응답의 계약 불일치 확인 | 경고, fixed 결과 유지 |
| latest | WARN_UNVERIFIED | 취득·환경·준비 실패로 최신 계약을 판단하지 못함 | 경고, fixed 결과 유지 |

증거만으로 drift와 환경 실패를 구분할 수 없으면 WARN_UNVERIFIED로 기록한다. GitHub job의 continue-on-error만 설정하고 결과를 버리는 구현은 허용하지 않는다.

cleanup 누락과 결과 기록 손상은 runner의 필수 제어 실패로 다룬다. latest 자체의 검증 실패 때문에 fixed verdict를 바꾸지는 않지만, 공통 runner가 결과를 신뢰할 수 없게 만들면 전체 실행은 INCOMPLETE다. 의도적 중단은 성공으로 기록하지 않는다.

### 결과 artifact의 최소 정보

result.json을 사람이 읽는 요약의 기준으로 삼는다. schema_version, 실행 식별자, 후보 commit 및 allowlist 소스 digest/dirty 여부, baseline producer SHA, fixed/latest resolved SHA·DDL hash, runner lock·이미지 digest·architecture, 각 case 결과와 누락/skip, 실패 단계·분류, 소요 시간, JUnit 위치, cleanup 결과를 포함한다.

일반 로컬 검사는 allowlist에 포함된 미커밋 변경도 검사할 수 있으나 commit SHA만으로 식별하지 않는다. 최종 제출 근거는 clean한 최종 commit으로 다시 실행한다. DSN·비밀번호·토큰·실제 사용자 데이터는 출력하지 않는다.

JUnit은 fixed/latest를 구분해 저장하고 latest assertion을 전체 필수 실패로 다시 집계하지 않는다. JSON은 상세 pytest traceback을 복제하지 않으며 traceback은 JUnit/콘솔의 역할이다.

## 제외한 보장과 다음 단계

실제 concurrent request, 실제 network server, Dashboard 변환·실행, assignment, 운영 DB, 전체 migration, 모든 dependency/architecture 조합은 이번 성공의 의미에 포함하지 않는다.

**필수 후속 동시성 검증:** 최소 두 실제 DB connection과 동시 요청으로 같은 scope를 경합시켜 승자·패자 응답의 identity, row 수, lock·commit·rollback을 확인한다. 현재 fake race 분기 테스트나 RCG-02로 대체하지 않는다.

## 2026-09-19 실제 확인 상태

초기 고정 baseline 실행은 RCG-01 PASS·RCG-07 FAIL이었다. binding 생략 시 실제 commit은 SQLSTATE 23514로 실패했지만 HTTP 200이 이미 나갔다. [E-09](evidence-log.md#e-09-pr-1-초기-실행과-중단)의 실패는 보존했다.

별도 PR 0을 개인 통합에 병합하고 PR 1에 기반으로 반영한 뒤 고정·최신 lane 각각 17개, 제어 30개가 통과했다. RCG-07은 실제 오류 응답과 전체 rollback을 확인한다. baseline fixture를 복원한 RCG-08도 통과했다. 의도적 assertion·timeout·최신 취득 실패의 결과는 [E-12](evidence-log.md#e-12-pr-1-로컬-gate-검증)에 구분해 기록한다. E-12의 결과는 커밋 전 후보 기준이다. 제출 commit 재검증 결과는 PR 본문에 별도로 기록하며 CI는 PR 2 범위다.
