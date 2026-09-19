# Run Contract Gate 검증 명세

| 항목 | 내용 |
|---|---|
| 대상 독자 | 테스트 구현자, 리뷰어 |
| 상태 | PR 0·1·2 개인 통합 병합 완료 · PR 2 CI PASS · PR 3A/3B 계획 확정, 미구현 |
| 기준 revision | 문서 기반 14e54cda5c4e92cf835a6ddffc5c20bac0d4ea1e / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
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

PR 1·2 성공에는 실제 concurrent request와 Dashboard 변환·launch 검증이 포함되지 않는다. 이를 PR 3A·3B로 계획했다. 실제 Decision/Dashboard 서버를 함께 띄운 live HTTP 여정·브라우저 전체 실행·assignment 처리·운영 DB·전체 migration·모든 dependency/architecture 조합은 PR 3에서도 보장하지 않는다.

**필수 후속 동시성 검증:** 최소 두 실제 DB connection과 동시 요청으로 같은 scope를 경합시켜 승자·패자 응답의 identity, row 수, lock·commit·rollback을 확인한다. PR 3A로 수행할 계획이며 현재 fake race 분기 테스트나 RCG-02로 대체하지 않는다.

## 2026-09-19 실제 확인 상태

초기 고정 baseline 실행은 RCG-01 PASS·RCG-07 FAIL이었다. binding 생략 시 실제 commit은 SQLSTATE 23514로 실패했지만 HTTP 200이 이미 나갔다. [E-09](evidence-log.md#e-09-pr-1-초기-실행과-중단)의 실패는 보존했다.

별도 PR 0을 개인 통합에 병합하고 PR 1에 기반으로 반영한 뒤 고정·최신 lane 각각 17개, 제어 30개가 통과했다. RCG-07은 실제 오류 응답과 전체 rollback을 확인한다. baseline fixture를 복원한 RCG-08도 통과했다. 의도적 assertion·timeout·최신 취득 실패의 결과는 [E-12](evidence-log.md#e-12-pr-1-로컬-gate-검증)에 구분해 기록한다. E-12의 결과는 커밋 전 후보 기준이다. 제출 commit 재검증 결과는 PR 본문에 별도로 기록하며 CI는 PR 2 범위다.

PR 2는 위 판정 규칙을 바꾸지 않고 Gate 종료 코드를 필수 check에 전달한다. latest 경고 표시와 JSON/JUnit 보관은 [개발자 안내 8절](developer-guide.md#8-pr-2-ci와-최종-제출), 정적·로컬 근거는 [E-14](evidence-log.md#e-14-pr-2-ci-구현과-로컬-검증)를 따른다.

## PR 3 검증 경계 — 계획, 아직 실행하지 않음

PR 3A와 PR 3B의 소유·순서는 [개발 계획 21절](implementation-plan.md#21-pr-3-milestone-실행-계약)이 기준이다. 아래 ID는 새 논리 case의 예약이며 구현된 테스트명·통과 개수가 아니다. 기존 RCG-01~09와 CTRL-01~05의 의미를 바꾸지 않는다. 실제 매개변수 case는 구현 시 각 저장소의 manifest에 모두 열거한다.

### PR 3A 실제 DB 동시성 검증 계획

각 case의 두 요청은 **같은 전용 DB·합성 seed**에 접근하고, 서로 다른 실제 PostgreSQL connection/transaction을 사용한다. 실제 API dependency·service·repository·commit/rollback을 통과한다. `TestClient`를 사용할 수 있지만 요청별 실행 context와 connection을 분리하고 요청 중첩을 관찰한다. TCP·별도 서버 process·운영 connection pool의 검증으로 확대하지 않는다.

| 논리 ID | 실행 순서 | 합격 조건 |
|---|---|---|
| RCG-10: 동일 scope, 선행 commit | 두 요청이 아직 없는 동일 scope를 읽도록 제어. A의 실제 INSERT 후 commit을 잠시 보류하고 B의 실제 DB 대기를 확인한 뒤 A를 완료 | 두 요청 성공, 같은 run/experiment ID 집합. run 1개·scope 내 고객군마다 experiment/binding 1개. B의 insert 패배·재조회 경로와 commit 이후 관측 확인 |
| RCG-11: 동일 scope, 선행 rollback | A가 실제 쓰기를 수행한 뒤 B가 대기하도록 제어. A에 테스트 전용 예외를 주입해 실제 rollback. B를 완료 | A는 성공 응답 없음, B는 성공해 완전한 run 1개 생성. A의 부분 experiment/binding·상태 변경이 잔존하지 않으며 B의 최종 row·소비 상태 정확 |
| RCG-12: 겹치는 scope 경합 | 동일 analysis에서 A와 A/B scope를 동시 요청. 실제 target/binding 경합이 생기도록 순서 제어 | 중복 target binding 없이 한 요청만 유효한 결과를 남김. 다른 요청은 해당 API의 정의된 충돌 응답. 성공 row 보존, 실패 요청의 추가 row·부분 소비 없음. 승자가 누구인지는 고정하지 않음 |

경합 발생 여부를 `sleep`이나 성공 횟수로 추정하지 않는다. 테스트 전용 barrier/event 또는 실제 메서드를 호출하는 wrapper로 순서를 제어한다. wrapper가 fake row·insert 결과를 반환하거나 commit을 대신 수행하면 안 된다. 한 요청이 DB lock을 잡은 상태에서 두 요청 모두 도달할 수 없는 barrier를 기다리게 만들지 않는다.

최소 증거는 요청별 식별자, 서로 다른 backend PID, 단계별 순서, 자체 요청 PID 사이의 blocker/wait 관찰, commit/rollback 완료, HTTP status/body, 종료 후 독립 connection의 DB 조회다. 관찰한 isolation level과 lock/statement/wait 제한도 기록한다. `pg_blocking_pids` 등으로 자기 실행의 PID만 조회하며 특정 PostgreSQL 내부 lock 이름 하나에 기대값을 고정하지 않는다.

경합 준비·대기 관찰 실패나 timeout은 필수 검증 불가다. 검증된 경합에서 응답·row 불변식이 깨지면 불일치다. 이를 setup/실행 결과에 구분해 기존 INCOMPLETE/FAIL 규칙에 연결한다. hang이 끝나지 않으면 성공으로 건너뛰지 않고 모든 worker를 회수·종료한 뒤 소유 DB/컨테이너를 정리한다. 다른 case를 같은 DB에서 병렬 실행하지 않는다.

서비스가 기대 불변식을 만족하지 못하면 실제 실패를 보존한다. 관찰 결과에 맞춰 성공 기준을 낮추거나 production retry/lock을 검증 PR에 추가하지 않는다. 현재 순차 RCG-09는 HTTP 409와 `segment_audience_target_already_run_bound`를 확인하며, router는 unique 위반에도 409를 반환한다. RCG-12는 충돌 409를 요구하되, 제어한 경합이 어느 실제 handler 경로에 도달하는지 확인해 정확한 오류 형태를 고정한다. 임의 500이나 timeout을 정상 충돌 응답으로 인정하지 않는다.

### PR 3A 결과·응답 bundle 계약 계획

제안 schema 식별자는 `rcg-run-consumer.v1`이다. 이 표는 구현할 계약이며 파일/옵션이 이미 존재한다는 뜻이 아니다. 기존 `result.json`·JSON/JUnit을 보존하고 별도 consumer bundle을 추가한다.

| 필드 묶음 | 필수 내용 |
|---|---|
| 식별 | schema_version, Gate run ID, case ID, request ID, fixed/latest lane |
| producer | Decision repository·실제 checkout SHA·candidate dirty·allowlist source digest; PR head/base는 별도 값 |
| 계약·환경 | fixed/latest resolved Contract SHA·DDL hash, baseline producer SHA, lock/image digest·architecture |
| 실제 입력·출력 | HTTP method/path, 합성 request body, 실제 response status·content type·body 파일; 배열·ID·status 보존 |
| 결과 연결 | Gate verdict·case outcome·DB assertion 결과, 해당 JSON/JUnit·경합 증거 파일 참조 |
| 무결성 | 포함 파일의 상대 경로·SHA-256, bundle manifest digest; 소비 실행이 선택한 producer SHA·source digest와 대조 |

정상 생성(RCG-01), 순차 재사용(RCG-02), 기존 baseline 재사용(RCG-08), 새 동시 요청(RCG-10/11/12)의 실제 응답을 식별해 내보낸다. 오류 응답은 실패 증거로 분리하며 성공 샘플로 바꾸지 않는다. body를 Dashboard 타입에 맞춰 보정하거나 필드를 채우지 않는다. 고정 baseline의 `expected.json`은 계속 독립 기준이다.

필수 bundle의 파일 누락·손상·SHA 불일치·case/lane/producer 혼동·경로 이탈은 INCOMPLETE로 처리한다. 해시는 provenance와 byte 일치를 검사하는 수단이며 발행자의 신원을 보증하는 서명은 아니다. 선택한 source revision과 생성 명령·CI 실행 링크를 함께 검토한다. 같은 폴더의 self-declared PASS만 신뢰하지 않는다.

고정 baseline·정상 응답·경합 응답은 합성 데이터만 포함한다. credential·DSN·전체 환경변수·운영 row·raw server log를 bundle에 넣지 않는다. 공통 metadata가 fixed와 latest를 혼동시키지 않도록 파일 경로와 manifest를 분리한다. 두 lane의 JSON/JUnit/bundle을 실패 시에도 가능한 범위에서 보관한다. 준비 실패로 생성되지 않은 응답이나 JUnit은 만들어내지 않는다.

### PR 3B Dashboard consumer 검증 계획

Dashboard의 실제 client가 로컬 replay 서버에서 3A의 원본 status/body를 읽고 실제 schema를 통과한다. 이 값을 실제 hook과 공유하는 변환 함수에 전달하고 실제 `launchPromotionExperiment`를 호출한다. 이 연결이 하나의 검사 안에 있어야 한다. 변환 함수만 별도 fixture로 테스트하거나 source 문자열 존재만 검사하는 것으로 대체하지 않는다. 기존 hook이 추출된 함수를 호출하는 wiring·typecheck·기존 회귀도 확인한다.

| 논리 ID | 입력·검사 | 합격 조건 |
|---|---|---|
| RCC-01: 정상 소비 | 3A 정상 생성 응답 → client → 공유 변환 → launch | request에 원래 analysis/generation/segment가 전달됨. scope·fallback·status 보존. build에 원래 run ID, start에 선택 experiment ID, dispatch 대상 채널에 원래 run ID가 전달됨 |
| RCC-02: 재사용 소비 | 순차 retry·동일 scope 동시 성공 응답을 각각 소비 | run/experiment identity와 scope가 유지되고 다른 ID를 만들어내지 않음. downstream 대역 호출 검증을 실제 assignment/발송 멱등성으로 주장하지 않음 |
| RCC-03: 기존 row 소비 | RCG-08의 고정 row → 후보 reader 실제 응답을 소비 | 보존된 expected identity와 같고 client·변환·launch가 다음 operation에 정확한 ID를 전달 |
| RCC-04: 계약 오류 차단 | 원본에서 명시적으로 파생한 malformed DTO, 요청과 다른 scope, experiment 누락/중복·잘못된 fallback | client 또는 launch의 실제 검증이 거절. build/start/dispatch 모두 호출되지 않음. 어느 층이 거절했는지 기록 |
| RCC-05: 실제 오류 응답 | 3A 실패 요청의 non-2xx status/body를 replay | 실제 client 오류 처리, 성공 run으로 변환하지 않음, downstream 호출 없음 |
| RCC-06: launch 분기 보존 | 실제 run 입력에 합성 assignment 결과를 조합: scheduled, start 실패, 채널별 dispatch, 이미 running 상태 등 | 해당 실제 flow의 호출·중단 순서와 반환 결과 유지. 상태/assignment 값을 변형한 입력은 파생 시나리오로 표시 |

원본 3A 응답과 RCC-04/06용 파생 입력을 다른 파일·case로 관리한다. 파생 입력은 원본 hash·변경 필드·목적을 기록하며 producer가 실제 반환한 결과로 표시하지 않는다. downstream operation 대역은 호출 인자·횟수·순서를 관찰하고 의도한 결과/예외를 돌려준다. 실제 assignment DB write·start transaction·외부 발송은 수행하지 않는다.

### PR 3 제어 검사와 판정 연결

추가 제어 검사는 3A의 경합 미성립·worker timeout·응답 bundle 누락/손상, 3B의 producer SHA/response hash 불일치·실제 consumer case 누락·수집 0·skip, 그리고 fixed 실패와 latest 성공의 잘못된 합산을 다룬다. 실제 제어 case 수는 구현 후 manifest·JUnit에서 집계하며 지금 숫자를 정하지 않는다.

3A/3B 모두 fixed가 필수이며 latest는 별도 경고다. latest bundle이 없으면 해당 lane을 WARN_UNVERIFIED로 기록하고 빈 consumer 실행을 PASS로 표시하지 않는다. 유효한 최신 입력의 계약 불일치는 WARN_DRIFT다. 공통 준비·결과 무결성·cleanup 실패는 필수 INCOMPLETE다. 각 PR의 결과와 milestone의 revision 연결을 별도로 확인하며 JUnit 개수를 두 번 합산하지 않는다.

완료 근거는 [E-16 대장](evidence-log.md#e-16-pr-3-milestone-결정과-증거-대장)에 모은다. PR 3 검증으로도 전체 브라우저·배포·네트워크·부하/성능·실제 발송 안전성을 보장하지 않는다.
