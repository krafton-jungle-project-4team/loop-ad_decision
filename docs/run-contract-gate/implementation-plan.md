# Run Contract Gate 개발 계획

| 항목 | 내용 |
|---|---|
| 대상 독자 | 구현 개발자·에이전트, 리뷰어 |
| 상태 | PR 0·1·2·milestone 문서 통합 완료 · PR 3A 로컬 PASS · PR 3A CI 제출 전 · PR 3B pending |
| 기준 revision | PR 3A base 6de82a36ddc1cf51c88a431d65e84f873ea8f472 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |
| Merge-safety verdict | **incomplete evidence** — PR 2 로컬·CI PASS; PR 3 동시성·Dashboard 소비 검증 및 배포 여정은 미검증 |

## 1. 결정 요약

개인 Decision 변경과 AI 생성 변경을 검증할 로컬·PR CI 도구를 만든다. 실제 FastAPI TestClient 요청이 실제 service·repository·PostgreSQL commit을 통과해야 한다. 고정 Contract 검사는 필수이며 최신 Contract 검사는 별도 경고다. 테스트 실행기와 DB를 별도 컨테이너로 실행한다.

PR 0 #394는 응답 전 commit을 보장하는 서비스 수정, PR 1 #395는 실제 DB 로컬 Gate, PR 2 #396은 같은 명령의 CI 연결로 개인 통합에 병합됐다. PR 2 제출 SHA의 로컬·CI 결과와 병합 revision은 [E-15](evidence-log.md#e-15-pr-2-ci-성공과-개인-통합-병합)에 구분해 기록한다.

2026-09-19 사용자는 후속을 **PR 3A — Decision 동시성·artifact**, **PR 3B — Dashboard 실제 consumer 검증**으로 나누고 두 저장소의 revision·결과를 **PR 3 milestone**으로 묶기로 결정했다. 문서 PR #397이 `6de82a3`으로 통합된 뒤 `feat/run-contract-gate-concurrency`에서 PR 3A를 구현했다. PR 3A의 commit·push·PR 생성은 승인됐고 merge는 제외한다. 3B는 이번 구현 범위 밖이다. 구현자가 따라야 할 범위·선행 관계는 이 문서 21절, 합격 기준은 검증 명세의 PR 3 절이 기준이다.

## 2. 해결하려는 실제 문제와 코드 근거

run의 안전성은 HTTP 응답뿐 아니라 저장된 scope·experiment·snapshot binding 및 transaction 결과에 달려 있다. 기존 테스트는 이 요소를 여러 계층에서 검사하지만 실제 요청부터 commit 후 재조회까지 한 경로로 묶지는 않는다.

- [create_run](../../app/decision/service.py), 기준 130행: source 선택 → scope 계산 → 기존 run 조회 → run/experiment 삽입 → V2 binding.
- [get_promotion_run_service](../../app/decision/router.py), 기준 180행: 실제 repository 구성 후 dependency 종료 시 commit, 예외 시 rollback, finally close.
- [PromotionRunRepository.insert_if_absent](../../app/decision/repositories.py), 기준 1264행: INSERT … ON CONFLICT DO NOTHING.
- [canonical promotion_runs DDL](https://github.com/krafton-jungle-project-4team/loop-ad_data-source_contract/blob/0ec2cef0290f4659ad21ccc1dd2a20df2801ff50/postgres/schema.sql#L1416): scope CHECK와 composite UNIQUE.
- [V2 binding 제약](https://github.com/krafton-jungle-project-4team/loop-ad_data-source_contract/blob/0ec2cef0290f4659ad21ccc1dd2a20df2801ff50/postgres/schema.sql#L2825): binding 집합·identity·lifecycle을 검사하는 지연 제약.

새 도구의 가치는 알려진 운영 장애를 해결했다는 데 있지 않다. 코드 변경 때 이 경계를 반복 검증하는 절차를 재현 가능하게 만드는 데 있다.

## 3. 사용자와 사용 시점

첫 사용자는 Decision을 수정하는 본인이다. scope·ID·SQL·DTO·transaction을 변경한 뒤, AI가 만든 변경을 검토할 때, 개인 통합 브랜치와 dev 제출 전에 실행한다. 조직 공통 플랫폼이나 운영 백오피스를 첫 목표로 삼지 않는다.

## 4. 기존 테스트의 보장과 공백

| 기존 근거 | 보장 | 이번에 보충할 부분 |
|---|---|---|
| [test_decision_run_service.py](../../tests/test_decision_run_service.py), 458·475·516행 | 동일 scope 재사용, scope 분리, 순서 정규화 | make_service는 2430행의 FakeRepositoryBundle 사용; 실제 SQL·DDL·commit 필요 |
| 같은 파일 556행 | concurrent insert 패배 분기의 처리 | 실제 동시 요청이나 DB lock 검증이 아님 |
| [test_decision_run_api.py](../../tests/test_decision_run_api.py), 220행 | repository 연결, commit 호출 | RecordingConnection이 실제 PostgreSQL을 대체 |
| [test_lean_audience_contract_integration.py](../../tests/test_lean_audience_contract_integration.py), 37행 | 실제 DDL 기반 allocation·binding lifecycle | run을 직접 INSERT하고 마지막에 rollback; 실제 run API commit 경로 추가 필요 |
| [Contract 검증 스크립트](https://github.com/krafton-jungle-project-4team/loop-ad_data-source_contract/blob/0ec2cef0290f4659ad21ccc1dd2a20df2801ff50/scripts/verify_postgres_contract.sh) | fresh DDL·migration·SQL contract 검증 자산 | 해당 전체 suite 복제 대신 Decision writer/reader 경계 검증 |

기존 테스트를 폐기하지 않는다. unit 테스트의 빠른 분기 검증과 실제 DB 검증의 역할을 나눈다.

## 5. 대안과 선택 이유

| 검토한 대안 | 결정과 이유 |
|---|---|
| fake 테스트만 추가 | 실제 SQL·제약·transaction 경계를 보충하지 못하므로 단독 해법에서 제외 |
| Contract SQL suite만 재실행 | application의 source 선택·retry·응답까지 보장하지 못함 |
| 외부 서비스와 Dashboard까지 즉시 E2E | PR 1·2에서는 제외. PR 3B는 artifact를 통한 실제 consumer 경계에 한정 |
| 테스트마다 구버전/신버전 Python 실행 | 비용이 커서 제외; 기준 코드가 생성한 fixture를 고정 |
| legacy와 V2 모두 검증 | 첫 범위는 현재 V2 run 경로에 집중 |
| Python 가상환경 + DB 컨테이너 | 사용자 선택에 따라 runner도 컨테이너화 |
| 최신 DDL을 필수 기준으로 사용 | 무관한 외부 변경이 판정을 흔들므로 고정 필수·최신 경고 분리 |
| 초기 로컬·CI 우선 분리 | PR 1·2로 로컬과 CI를 먼저 완성. 후속은 저장소별 PR 3A·3B로 분리, 배포 연결 제외 |

## 6. 단계별 범위와 비범위

PR 1·2 완료 범위: 명시적 analysis_id·generation_id·segment_ids의 일반 run 생성, 순차 재시도, scope/ID/row 일치, V2 binding, 실패 rollback, 고정 기존 row 재사용, 로컬 실행, PR CI, 결과 보관.

PR 3A 구현 범위: 같은 전용 DB의 복수 실제 connection에서 요청을 경합시키고 commit·rollback·재조회를 확인한다. Dashboard가 소비할 실제 요청/응답과 DB 근거를 provenance가 있는 artifact로 내보낸다.

PR 3B 계획 범위: 실제 Dashboard client·응답 변환·launch flow에 PR 3A 응답을 연결하고 다음 operation에 전달되는 run/experiment ID, scope, 호출 순서와 오류 차단을 검사한다. 응답 변환 추출이 필요하면 Dashboard 내 동작 보존 refactor까지 포함한다.

공통 비범위: 브라우저 전체 E2E, 배포 서버·ALB 검증, 실제 assignment 처리·start 상태 전이·발송, next-loop activation, ClickHouse·외부 AI, 분석/콘텐츠 생성 전체 파이프라인, 전체 migration·legacy 호환, 운영 데이터, 배포/branch protection 변경. PR 3B의 operation 호출 기록은 downstream 처리 성공의 증거가 아니다.

동시성과 producer bundle은 **3A 로컬 검증 완료**, consumer는 **3B pending**이다. 기존 순차 테스트나 PR 2 CI 성공으로 PR 3 전체를 완료 처리하지 않는다.

## 7. Producer → row → consumer → Dashboard action

    Dashboard의 선택된 source/고객군
      → Decision POST /decision/v1/promotions/{promotion_id}/runs
      → 실제 service / repository
      → promotion_runs + ad_experiments + promotion_run_target_bindings
        및 allocation/target/exclusion 상태
      → 실제 commit 후 새 연결로 조회
      → RunCreateResponse
      → Dashboard API client의 응답 검증
      → 화면의 snake_case → camelCase 변환
      → launchPromotionExperiment의 scope/experiment 검증
      → assignment build / start / dispatch

PR 1·2와 PR 3A는 실제 commit 후 row·응답을 검사한다. PR 3B는 artifact를 실제 Dashboard 소비 코드에 연결하고 assignment/start/dispatch operation의 호출 인자를 관찰한다. 위 전체 여정의 실제 downstream 처리나 화면 클릭까지 실행했다고 주장하지 않는다. API 준비 전의 분석·generation·snapshot 데이터는 canonical DDL을 만족하는 합성 fixture로 준비한다.

## 8. Run·scope·ID·transaction 불변식

검증의 상세 기준은 [검증 명세](verification-spec.md)다.

- run identity는 project·promotion·analysis·generation·loop·정규화된 segment scope로 결정된다.
- 같은 identity의 재시도는 같은 run과 experiment 집합을 돌려주며 row·binding을 늘리지 않는다.
- scope 내 고객군마다 experiment 하나, scope 밖 experiment는 생성하지 않는다.
- V2 run binding 집합은 scope와 일치하고 allocation·snapshot·target identity 및 lifecycle이 맞아야 한다.
- commit 이후 새 연결에서 기대 row를 읽을 수 있어야 한다.
- 실패하면 새 run·experiment·binding과 관련 상태 변경이 함께 rollback되어야 한다.
- 서로 다른 scope의 정상 생성은 **겹치지 않는 target 집합**으로 검증한다. 같은 analysis+segment의 중복 binding은 별도의 금지 조건이다.

[audience_snapshots.py](../../app/decision/audience_snapshots.py)의 bind_run_targets와 canonical UNIQUE(target_analysis_id, segment_id)를 무시하고 {A} 생성 후 {A,B}도 반드시 성공해야 한다고 명세하지 않는다.

## 9. 기존 row → 새 reader 호환

최초 baseline producer는 위 Decision SHA, DDL은 위 Contract SHA로 고정한다. baseline 코드가 합성 입력을 받아 실제 run API로 commit한 결과와 필요한 연관 row를 fixture로 내보낸다. producer SHA·DDL SHA·생성 절차·요청·기대 응답·파일 hash를 provenance로 기록한다.

후보 코드는 그 fixture를 읽고 같은 요청을 재사용한다. 정상 테스트의 expected row를 후보 writer로 매번 재생성하지 않는다. 최초에는 후보와 baseline이 같을 수 있지만, 이후 코드 변경에도 fixture와 기대값을 보존하므로 시간에 따른 호환성 기준이 된다.

fresh DDL에 fixture를 복원하는 검사이며, 구 DB 전체의 migration을 실행했다는 뜻은 아니다. 모든 historical status와 legacy row를 지원한다고 확대하지 않는다. 자세한 baseline 생성·복원 조건은 [명세](verification-spec.md#기존-row-fixture) 참조.

## 10. Dashboard 소비 코드 연결 방식

PR 3B는 Dashboard 저장소가 소유한다. Decision 저장소에 Zod schema·변환·launch 로직을 복사한 별도 TS 검증기를 만들지 않는다. 확인한 로컬 Dashboard revision은 `40af537b0e48a26f738f9cbf4dfc5fbcf2055d62`이며 현재 배포/원격 최신 revision으로 간주하지 않는다. 구현 착수 시 대상 branch·SHA·lock을 다시 확인한다.

- [실제 API client](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/blob/40af537b0e48a26f738f9cbf4dfc5fbcf2055d62/apps/api-server/src/features/dashboard/provider/dashboard-decision-client.ts): 실제 request 생성·fetch·응답 schema 검증을 사용한다.
- [화면의 응답 변환](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/blob/40af537b0e48a26f738f9cbf4dfc5fbcf2055d62/apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/usePromotionWorkspaceController.ts): 현재 hook 안의 run 변환을 동작 보존 순수 함수로 추출하고 실제 hook도 그 함수를 호출하게 한다. 테스트 전용 복사본은 금지한다.
- [실행 흐름](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/blob/40af537b0e48a26f738f9cbf4dfc5fbcf2055d62/apps/web-client/src/features/dashboard/ui/pages/campaign/promotion/promotionExperimentFlow.ts): 실제 `launchPromotionExperiment`의 scope 검증·호출 순서를 사용한다.

PR 3A bundle의 HTTP status/body를 격리된 로컬 replay 서버가 제공하고 실제 client가 읽는다. upstream 응답만 재생하며 client·schema·변환·launch를 mock하지 않는다. replay는 Decision이 같은 프로세스 흐름에서 live 호출됐다는 뜻이 아니다. `buildAssignments`·`startExperiment`·`dispatch`는 인자를 기록하는 테스트 대역으로 제한하고, 그 반환값도 합성 downstream 입력이라고 표시한다.

이 경계의 공식 명칭은 **run-consumer 통합 검증**이다. 전체 Dashboard E2E·브라우저 E2E·실제 발송 성공이라고 표현하지 않는다. client부터 launch까지의 연결 검사와 각 층의 오류 검사를 모두 갖춰야 한다. 자세한 case는 [검증 명세](verification-spec.md#pr-3b-dashboard-consumer-검증-계획)에서 관리한다.

## 11. DDL·revision·실행 환경 고정

canonical 원본은 Contract 저장소의 postgres/schema.sql이다. Decision에 production schema.sql을 복제하지 않는다.

- 필수 lane: 고정 SHA의 DDL을 임시 다운로드·검증하여 빈 전용 DB에 적용.
- 최신 lane: 실행당 Contract main SHA를 한 번 해석하고 그 SHA만 사용. 별도의 빈 DB에서 같은 시나리오 실행.
- 최신 SHA가 고정 SHA와 같아도 결과를 두 lane으로 구분한다.
- 고정 SHA를 가져오지 못하면 필수 검증 불가. 최신 SHA를 가져오지 못하면 최신 검증 불가 경고. 다른 revision으로 조용히 대체하지 않는다.
- canonical DDL bytes의 hash, resolved SHA, runner lock hash, 이미지 digest·실행 architecture를 결과에 남긴다.
- Gate 전용 직접·간접 의존성과 Python/pgvector 이미지를 고정한다. 실제 digest·lock 값은 이미지 해석·설치 검증 이후에만 기입한다.
- pyproject.toml과 lock의 불일치는 검증 준비 실패다. 운영 이미지와 같은 의존성을 사용한다고 추정하지 않는다.

runner는 운영 Dockerfile과 분리하고 필요한 소스·테스트·설정만 build context에 넣는다. .env·Git metadata·개인 파일·credential을 image에 복사하지 않는다. 전용 네트워크에서 테스트하고 외부 서비스 연결은 허용하지 않는다. 필요한 다운로드는 준비 단계로 분리한다.

기존 [conftest.py](../../tests/conftest.py)는 localhost/Unix socket만 허용한다. Gate는 공유 Unix socket을 이용하거나 좁게 분리된 전용 연결 설정을 구현해야 하며, 기존 테스트의 운영 DB 방지 제한을 광범위하게 풀지 않는다. **기본 구현은 두 컨테이너에 전용 Unix socket volume을 공유하는 방식**으로 정한다.

고정 기준 갱신은 별도 PR에서 DDL diff·fixture 호환·전후 결과를 검토한다. 최신 경고성 PASS가 자동 갱신을 의미하지 않는다.

## 12. PR별 책임과 독립 merge 가능 여부

| PR | 포함 | 제외 | 완료·선행 관계 |
|---|---|---|---|
| PR 0 → 개인 통합 | POST /runs 응답 전 commit, 최소 FastAPI 지원 버전, 회귀 테스트 | Gate 구현·fixture·CI | 선행 서비스 수정. PR #394 → 개인 통합에 09442f2로 병합 완료 |
| PR 1 → 개인 통합 | 실제 DB 시나리오, baseline fixture/provenance, runner·DB 준비/정리, lock, JSON/JUnit, 문서 | CI workflow·Dashboard·서비스 동작 수정 | 로컬 단일 명령으로 완결. 단독 사용 가능 |
| PR 2 → 개인 통합 | 같은 명령의 CI 호출, 고정/최신 결과 분리, artifact 보관, 문서 갱신 | 배포 workflow·branch protection | PR 1 반영 후 진행. PR 1 없이 독립 실행 가능한 변경이 아님 |
| PR 3A → Decision 개인 통합 | 실제 동시성, Gate manifest·결과/응답 bundle·제어 검사·CI artifact 확장 | Dashboard runtime·서비스 결함 수정 | 문서 #397 통합 후 구현·로컬 PASS; 3B 없이 독립 실행·검토 가능 |
| PR 3B → Dashboard 합의한 통합 branch | 실제 client·공유 변환·launch 소비 검사, 전용 실행·CI·결과 | Decision runtime·실제 assignment/발송·브라우저 E2E | 3A의 고정 producer SHA·bundle 계약 필요; Dashboard base는 착수 시 확인 |
| PR 3 milestone | 3A·3B revision 조합 및 각 결과·artifact hash 연결 | 별도 구현 PR 번호나 GitHub Milestone 자동 생성 | 두 저장소의 검증 근거를 같은 조합으로 연결해야 완료 |
| 통합 → dev Draft | 저장소별 통합 결과·최종 검증 근거 | 검토되지 않은 추가 기능 | 기존 PR 1·2 제출과 PR 3 milestone 완료는 별도 상태; 실제 제출 시 포함 범위를 명시 |

PR 1 생성 후보: scripts/run-contract-gate.sh, tools/run_contract_gate/, tests/run_contract_gate/, tests/fixtures/run_contract_gate/, Dockerfile.run-contract-gate 및 전용 ignore, Gate lock/설정 파일, 이 문서 묶음. 현재 로컬 구현 범위다. 초기 중단 근거는 E-09, PR 0 병합과 전체 검증은 E-11/E-12에서 확인한다.

PR 2 변경: .github/workflows/run-contract-gate.yml. PR 3은 21절의 소유 경계를 따른다. 이 문서화에서 자동 Markdown 보고서 생성기나 별도 원격 PR은 만들지 않는다.

현재 .gitignore는 *.md를 제외하므로 이번 문서 7개만 정확한 경로로 예외 처리했다. 이 작은 추적 설정 변경도 PR 1에 포함한다. 다른 기존 로컬 문서의 제외 정책은 유지한다.

## 13. Branch와 worktree 전략

- 시작 기준: 원격 조회로 확인한 origin/dev = e1de8b29b902b54df3a58f21f1daa27c1171fe80.
- 개인 통합: integration/run-contract-gate.
- PR 0: fix/run-commit-before-response, 별도 /Users/ran/loop-ad/loop-ad_decision-run-commit-fix worktree.
- PR 1: feat/run-contract-gate-db. PR 0 반영 후 기반 정렬; 서비스 patch를 PR 1 diff에 중복 포함하지 않음.
- PR 2: feat/run-contract-gate-ci, PR 1이 통합된 개인 브랜치에서 분기.
- PR 3A 후보 branch: feat/run-contract-gate-concurrency (구현 착수 시 생성).
- PR 3B 후보 branch: test/run-contract-consumer (Dashboard에서 base 확정 후 생성).
- PR 3 문서 branch: docs/run-contract-gate-pr3-milestone, PR 2 통합 revision에서 분기한 로컬 작업.
- 현재 문서 worktree: /Users/ran/loop-ad/loop-ad_decision-run-contract-gate.
- 최종 Draft의 base: dev. main은 대상이 아니다.

PR 0·1·2와 milestone 문서 #397은 개인 통합에 병합됐다. PR 3A branch는 `6de82a36ddc1cf51c88a431d65e84f873ea8f472`에서 분기했으며 Dashboard 파일은 변경하지 않는다. PR 3A를 구현·검증해 개인 통합 대상 PR로 제출하고 merge는 수행하지 않는다. PR 3B는 후속 작업이다. 원래 main 작업 공간의 변경·untracked 파일은 옮기지 않는다.

실제 원격 쓰기 전 repository·branch·commit 범위·diff/stat·정확한 파일을 검토하고 해당 작업 승인을 받는다. AGENTS.md, agent/, .codex/, .env, 출력물·캐시·개인 기록은 포함하지 않는다. 광범위한 git add를 사용하지 않는다.

## 14. Unit·contract·integration·CI 구분

- Unit: scope 계산과 service 분기, Gate의 결과 집계·exit code·필수 case 누락 판정.
- Contract: canonical DDL과 snapshot/binding·scope 제약, 고정 기존 row와 후보 reader.
- Integration: TestClient → 실제 dependency/service/repository → PostgreSQL → commit/rollback 후 별도 연결 확인.
- CI: 위 검증을 재현하고 결과를 보관하는 실행 환경. CI 자체가 새 의미의 서비스 검증은 아니다.

일반 unit suite의 DB opt-in 정책은 유지한다. Gate 필수 case의 skip을 성공 처리하지 않는다. 상세 case ID와 테스트명은 [검증 명세](verification-spec.md)에 둔다.

## 15. 실패·skip·환경 부족 판정

[검증 명세의 판정표](verification-spec.md#결과-판정)가 기준이다. 고정 lane은 통과·불일치·검증 불가를 구분하며 뒤의 두 상태는 필수 check 실패다. 최신 lane의 불일치·검증 불가는 별도 경고이고 고정 verdict를 바꾸지 않는다.

필수 case 누락, skip, xfail, timeout, fixture 복원 실패, 결과 파일 유실은 PASS가 아니다. DB assertion 실패를 서비스 결함으로 곧바로 단정하지 않고 fixture·환경·후보 코드·DDL의 원인을 구분한다.

## 16. 위험·rollback 경계·후속 과제

| 위험 | 대응 |
|---|---|
| fixture가 실제 계약보다 단순함 | 실제 baseline writer로 생성하고 commit 성공·읽기 경로를 검증 |
| 같은 분석·고객군 중복 binding을 scope 분리로 착각 | 겹치지 않는 정상 scope와 중복 binding 거절을 분리 |
| 의존성·이미지 변경으로 결과가 흔들림 | Gate 전용 lock·digest와 입력 증거 보관 |
| cross-repo 접근·다운로드 장애 | 필수 검증 불가와 최신 경고를 분리 |
| cleanup이 다른 DB·컨테이너를 삭제 | 이번 실행이 만든 ID/label/volume만 정리 |
| 자동화가 배포를 잘못 막음 | 이번에는 PR check만 추가, deploy 수정 없음 |

이번 rollback은 Gate workflow·도구·문서 변경을 되돌리는 범위다. 운영 DB migration이나 운영 데이터 rollback을 수행하지 않는다.

구현한 PR 3A: 실제 동시 요청·복수 connection에서 같은 scope 생성 경합, insert 패배 경로, lock·commit·rollback 결과 검증. PR 3B: Dashboard 실제 소비 코드 연결. 두 PR은 하나의 PR 3 milestone으로 추적한다. legacy 전체 호환·배포 차단·전체 migration suite는 필요를 재평가한 뒤 결정한다.

## 17. 지표와 완료 조건

수치 목표를 꾸며 넣지 않는다. 측정할 항목은 필수 case 실행/통과/누락 수, 고정/최신 각각의 결과·소요 시간, cold/warm 실행 시간, 환경 실패 원인, 실제 탐지한 회귀 종류다. CI 안정성은 실행 횟수와 기간을 함께 기록한다.

PR 1·2 완료 조건(이 목록 통과만으로 PR 3 milestone이 완료되지는 않음):
1. 필수 case 전부 실행·통과, skip/누락 없음.
2. 로컬에서 단일 명령으로 준비·실행·정리·결과 기록.
3. 의도적 실패에 대해 올바른 비통과 exit/check와 cleanup을 확인.
4. baseline fixture 생성 provenance와 후보 reader 결과 확보.
5. PR CI가 개인 통합 브랜치를 대상으로 실제 실행됨.
6. 통합 최종 commit에 대해 로컬·CI 및 두 lane 결과 확보.
7. 최신 경고가 있으면 내용과 영향 확인을 기록; 필수 PASS로 은폐하지 않음.
8. 문서의 명령·테스트 링크·기여 표현을 실제 결과에 맞춰 갱신.

## 18. 구현 순서와 중단 기준

| 순서 | 작업 | 중단 기준 |
|---|---|---|
| 0 | 현재 문서 작성·상호 링크 검증 | 문서가 구현·성과를 사실처럼 표현하면 수정 |
| 1 | 구현 착수 시 base·DDL·접근·image 확인 | 조사 기준과 의미 있는 차이, 원본 조회/lock 재현 불가 |
| 2 | synthetic seed와 baseline producer fixture 생성 | 실제 writer가 commit하지 못하거나 provenance를 보장하지 못함 |
| 3 | 정상·retry·scope·ID 시나리오 | canonical DDL에 맞추려면 임의 완화나 runtime 수정이 필요 |
| 4 | rollback·deferred constraint·기존 row 검증 | 실제 transaction을 fake로 대체해야만 통과 |
| 5 | runner·판정·cleanup·artifact 검증 | skip/누락이 녹색이 되거나 다른 자원 접근 가능 |
| 6 | PR 1 검토·통합 후 PR 2 CI | CI에서 외부 저장소 접근 또는 같은 명령 실행 불가 |
| 7 | 기존 PR 1·2 통합 제출 준비 | 실제 포함 revision의 필수 lane 실패/검증 불가, 근거 불일치 |
| 8 | PR 3A 동시성·bundle | 실제 경합 증거 없음, 서비스 수정 필요, 원본 응답·provenance 연결 실패 |
| 9 | PR 3B 실제 consumer 연결 | 실제 코드를 복제해야만 통과, 의미 변경 필요, producer 입력 불일치 |
| 10 | PR 3 revision 조합 확정·제출 준비 | 어느 필수 결과라도 실패/미실행, 두 저장소와 bundle hash 연결 불가 |

서비스 결함은 재현과 원인 분석까지 수행하고 수정 범위를 별도로 결정한다. 통과시키기 위해 fixture 기대값·DDL·필수 case를 임의로 바꾸지 않는다.

## 19. 미확인 사항과 해소 방법

- CI runner의 익명 Contract 취득: PR 2 실제 Actions에서 확인. 실행 revision·결과는 E-15. 다른 revision·runner의 성공을 자동 보장하지 않는다.
- canonical 전체 DDL과 baseline API를 만족하는 최소 seed: RCG-01 실제 commit으로 확인. 고정 row export·복원과 baseline/candidate reader 재사용을 확인했다.
- 고정 이미지 digest·transitive lock: Linux/arm64 실제 설치·실행 확인. E-09 기록; 다른 architecture는 미검증.
- Gate 실행 시간: E-12에 단일 환경 측정 기록. 장기간 환경 실패 빈도와 cold build 시간은 미측정.
- Dashboard 원격 최신·배포 상태와 실제 운영 사용: 이번 과제에서는 미확인, 성과 주장 근거로 사용하지 않음.

이 항목은 조용히 구현자에게 정책 선택을 넘기는 항목이 아니다. 지정된 확인 단계에서 사실을 검증하고, 실패하면 위 중단 기준을 적용한다.

## 20. 구현 중 이해해야 할 핵심 개념

[학습 안내](learning-guide.md)의 1~6단원은 기존 Gate, 7~9단원은 PR 3A 구현·3B 계획을 따른다. 설명할 수 있어야 할 차이는 fake/실DB, 멱등성/동시성, statement/commit, scope/ID/binding, baseline fixture/현재 writer, 필수 판정/최신 경고, PR check/배포 차단이다.

이 설계는 운영 경험을 새로 만드는 기능이 아니다. 실제 코드·DB 경계를 검증하고 재현 가능한 증거를 쌓는 개발 안전성 과제다.


## 21. PR 3 milestone 실행 계약

### 명칭과 책임

PR 3A/3B는 문서상 논리 이름이다. 서로 다른 저장소의 GitHub PR 번호와 혼동하지 않는다. 번호를 PR 3/4로 바꾸더라도 milestone 이름과 두 역할은 유지한다. GitHub Milestone 생성은 이번 문서 작업에 포함하지 않는다.

| 구분 | PR 3A: Decision | PR 3B: Dashboard |
|---|---|---|
| 소유자 관점 | writer·transaction·실제 응답 생산 | client·변환·launch의 실제 소비 |
| 변경 위치 / 3B 예정 위치 | tests/run_contract_gate/, tools/run_contract_gate/, scripts/run-contract-gate.sh, 전용 lock/image가 필요하면 해당 파일, Gate workflow·문서 | api-server client contract 테스트, web-client 변환 함수·hook·launch 테스트, 전용 bundle 검증/실행 도구·CI·문서 |
| production 변경 | service/repository/API 의미 변경 제외 | hook의 기존 변환을 공유 함수로 추출하는 동작 보존 변경만 허용 |
| 필수 결과 | 경합 증거·최종 DB 상태·응답 bundle·JSON/JUnit | 동일 bundle을 소비한 client/변환/launch 결과·JSON/JUnit |
| 독립 완료 의미 | Decision 동시성·producer 검증 완료 | 특정 Decision revision과 Dashboard revision 조합의 소비 검증 완료 |

### 구현 순서와 중단 조건

1. PR 2 통합 SHA·Contract 고정 SHA를 확인하고 3A를 시작한다. 기존 RCG-01~09·CTRL 판정은 유지하며 새 필수 manifest와 누락 검사를 확장한다.
2. 동일 scope의 commit 경합 → 선행 rollback → 겹치는 scope 충돌 순으로 구현한다. 실제 경합 증거가 없으면 실행 횟수가 많아도 성공으로 간주하지 않는다.
3. 기존 정상·retry·baseline 재사용과 새 경합에서 실제 요청/응답을 bundle로 만든다. 검증 명세의 schema·hash·provenance 조건을 검토하고 3A clean SHA의 로컬·CI 결과를 고정한다.
4. 3B 착수 시 Dashboard의 작업 branch·base·실제 client/변환 위치·Node lock을 다시 확인한다. 로컬 조사 SHA를 원격 최신이라고 사용하지 않는다.
5. 실제 변환을 공유 함수로 추출하고 hook과 테스트가 함께 사용하게 한다. upstream replay → client → 변환 → launch를 하나의 연결 검사로 실행한다. 호출 대역은 downstream operation에만 둔다.
6. 3B CI는 준비 단계에서 **명시한 3A Decision SHA**를 취득하여 같은 Gate 명령으로 bundle을 재생성하고 소비한다. 로컬은 동일 provenance의 보관 bundle을 전달할 수도 있다. mutable branch의 최신 artifact나 무관한 last-successful artifact로 대체하지 않는다. 저장된 baseline expected를 새 writer로 갱신하지 않는다.
7. 두 저장소의 최종 검증 revision 조합을 [근거 기록](evidence-log.md#e-16-pr-3-milestone-결정과-증거-대장)에 연결한다. 이후 producer·DDL·consumer·lock이 바뀌면 영향받는 생산/소비 검증을 새 조합으로 다시 수행한다.

서비스 결함, 계약·ID·status 의미 변경, 동작이 달라지는 Dashboard 수정, 추가 credential/운영 환경 의존성이 필요하면 해당 증거를 보존하고 별도 수정 범위를 제시한다. 검증을 통과시키려고 구현 PR 안에 조용히 섞지 않는다. 3B의 공유 변환 추출 외 refactor는 필요성과 영향부터 검토한다.

### 고정 기준과 CI 비용

Contract의 fixed 필수·latest 경고 규칙과 exit 0/1/2 의미를 유지한다. 3A 동시성은 두 lane에 포함한다. 3B도 같은 Decision·Dashboard revision을 유지한 채 각 Contract lane의 bundle을 별도로 소비한다. fixed bundle 취득/검증 불가 또는 fixed consumer 실패는 필수 실패이며, latest 취득/생산/소비 결과는 경고로 구분한다. 공통 runner·결과 무결성·cleanup 실패를 latest 경고로 숨기지 않는다.

Dashboard 최신 branch까지 추가하는 조합 검사는 PR 3 범위 밖이다. CI에서 Decision Gate 재실행 비용과 Node 준비 시간을 따로 측정한다. 다운로드는 준비 단계에서만 허용하고 replay·소비 검사는 외부 연결을 차단한다. 접근 실패 시 credential이나 대체 revision을 임의로 추가하지 않는다.

### Milestone 완료 기준

- [x] 3A 기존·새 필수 case가 모두 실행되고, 실제 DB 경합·commit/rollback·정리 증거가 있다. 로컬 E-17 기준.
- [x] 3A artifact가 실제 응답과 DB 검증을 같은 실행으로 연결하며 누락·손상·hash 불일치 탐지 제어 검사가 있다. 로컬 E-17 기준.
- [ ] 3B의 실제 client → 공유 변환 → 실제 launch 연결 검사가 정상·retry·baseline·오류 입력을 다룬다.
- [ ] 각각의 clean revision·CI 결과·Contract SHA·lock/image·bundle hash·소비 결과가 같은 조합으로 연결된다.
- [ ] latest 경고가 있으면 원인과 영향이 기록돼 있고 필수 통과로 합산되지 않는다.
- [ ] 각 PR의 merge 상태와 실제 검증 SHA를 구분한다. 통합 후 SHA가 바뀌면 실제 tree·source digest를 대조하고 통합 결과를 재검증한다.
- [ ] 개발자 실행 안내·근거 대장·학습/포트폴리오의 완료 표현이 실제 결과와 일치한다.

3A만 통과하면 milestone은 `3A verified / 3B pending`이다. 3B가 무관한 fixture로 통과하거나 revision 연결이 없으면 `incomplete evidence`다. 두 PR의 fixed 검증과 연결 근거가 갖춰져야 `PR 3 verified`로 기록한다. merge 여부는 별도 필드이며 verified를 merge 승인으로 해석하지 않는다.

### Cross-repository 안전성 판정

| 항목 | 계획 단계의 결론 |
|---|---|
| Verdict | 3A local verified / 3B pending — 전체 milestone은 incomplete evidence |
| Affected Dashboard journey | run 생성 응답 수신 → 변환 → launch의 다음 operation 인자 확인; 실제 downstream 실행은 제외 |
| Contract impact | 기존 run DTO·identity·DDL은 유지. 새 계약은 테스트용 producer bundle뿐이며 v1과 hash로 관리 |
| Existing-data compatibility | 고정 baseline row → 후보 Decision 재사용 응답 → 후보 Dashboard client/변환/launch |
| Required verification | 기존 unit·client contract, 3A DB integration, 3B consumer integration, 두 CI; 배포 smoke는 별도 범위 |
| Merge order | PR 2 → 3A → 고정 3A SHA를 사용하는 3B. 공유 변환은 기존 UI 동작 유지 후 독립 배포 가능해야 함 |
| Rollback boundary | 테스트·runner·공유 변환 변경의 revert. 운영 row/schema 변경 없음; 기존 baseline과 다른 팀 변경 보존 |
| Excluded local context | AGENTS.md, agent/, .codex/, .env, credential·운영 데이터 |
