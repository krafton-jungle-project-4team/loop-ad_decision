# Run Contract Gate 근거 기록

| 항목 | 내용 |
|---|---|
| 대상 독자 | 구현자, 리뷰어, 포트폴리오 작성자 |
| 상태 | PR 3A merged (`9ace3b6`) · PR 3B stacked local/CI/artifact verified · Dashboard #246/#247 OPEN (미병합) |
| 기준 revision | Decision producer `9ace3b6` → Dashboard fix `b77b901` → 3B head `58a2133` / 실제 CI checkout `5535ffb`; 전체 SHA·hash는 E-17 마지막 실행 조합 참조 |
| 마지막 확인 | 2026-09-19 KST |

## 기록 규칙

이 기록은 운영 로그나 자동 테스트 보고서가 아니다. 사람이 중요한 결정과 실제 증거를 연결한다. 실행 전에는 미실행, 측정 전에는 미측정으로 남긴다. 0회 실패·100% 통과처럼 실행이 있었던 것으로 오인될 숫자를 기입하지 않는다.

각 새 실행은 날짜, 목적, 후보 commit/source digest, baseline·Contract SHA, 환경 digest, 명령, case/lane 결과, artifact 위치, 관찰, 한계를 기록한다. raw log·DB dump·credential·실사용자 자료를 이 문서에 복사하지 않는다.

## 현재 조사 기록

아래 E-01~E-08은 최초 조사 시점 기록이다. 이후 구현·병합·실행은 E-09~E-12에 순서대로 남겼다.

| ID | 확인 내용 | 방법·근거 | 한계 |
|---|---|---|---|
| E-01 | 원격 Decision dev = e1de8b29b902b54df3a58f21f1daa27c1171fe80 | credential helper를 끈 익명 git ls-remote --heads origin dev 성공. 기존 로컬 origin/dev와 일치 | 조회 시점의 ref 확인. 배포 상태 증거 아님 |
| E-02 | 원격 Contract main = 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 | 같은 방식의 익명 원격 ref 조회 성공. 로컬 HEAD와 일치 | CI 취득·전체 DDL 실행·image 준비는 미검증 |
| E-03 | Dashboard 조사 HEAD = 40af537b0e48a26f738f9cbf4dfc5fbcf2055d62 | 로컬 Git 및 실제 client/controller/flow 코드 확인. cached origin/main은 df9daf13b57324d52a0eacb15b33c845a399c79f | 원격 최신·배포 미확인 |
| E-04 | 실제 transaction dependency 존재 | [router.py](../../app/decision/router.py), get_promotion_run_service | 새 Gate에서 실행한 결과는 아직 없음 |
| E-05 | 현재 service/API 테스트의 fake 경계 확인 | [service fixture](../../tests/test_decision_run_service.py), [API wiring](../../tests/test_decision_run_api.py) | fake 테스트가 무가치하다는 의미 아님 |
| E-06 | 기존 actual DB lifecycle 테스트 존재 | [lean integration](../../tests/test_lean_audience_contract_integration.py) | run 직접 INSERT 및 마지막 rollback. 새 API commit 검증과 다름 |
| E-07 | V2 target 중복 binding 제약 확인 | [DDL](https://github.com/krafton-jungle-project-4team/loop-ad_data-source_contract/blob/0ec2cef0290f4659ad21ccc1dd2a20df2801ff50/postgres/schema.sql#L2187), [repository](../../app/decision/audience_snapshots.py) | 이번에는 코드·DDL 읽기만 수행 |
| E-08 | 별도 worktree·로컬 통합/PR 1 branch 준비 | integration/run-contract-gate 및 feat/run-contract-gate-db, 기준 E-01 | remote push·commit·PR·merge 없음 |

처음 제한된 네트워크 환경에서는 DNS 조회가 실패했다. 이후 허용된 네트워크 조회에서 E-01/E-02를 확인했다. 이 후속 증거로 원격 SHA의 미확인은 해소됐지만, GitHub Actions에서의 접근·실행은 여전히 미확인이다. 로컬 SHA가 같았으므로 추가 fetch는 필요하지 않았다.

원래 main 작업 공간의 사용자 변경과 untracked 파일은 그대로 두었다. 기존 학습·계획 자료 일부는 로컬에만 있고 기준 Git tree에 없었다. 새 문서는 그 자료를 복제하거나 필수 링크로 삼지 않고 실제 추적 코드·테스트를 연결했다.

## 이전부터 존재한 기능과 본인의 기존 기여

아래는 이번 문서 작업에서 구현한 기능이 아니다. Giran Oh의 author 정보와 해당 commit 내용을 기존 기여의 근거로 구분한다.

| ID | 기존 기여 | commit 근거 | 주장 범위 |
|---|---|---|---|
| P-01 | run의 고객군 scope 멱등성 보강 | [d07fd6d](https://github.com/krafton-jungle-project-4team/loop-ad_decision/commit/d07fd6d9545f7f60793b8782a5020bc8d60f4b8d) | 기존 구현 기여 |
| P-02 | scope 멱등성 검증 추가 | [6d9da30](https://github.com/krafton-jungle-project-4team/loop-ad_decision/commit/6d9da30600833dfd7f0ab399dbe2dbfc3fbcf39a) | 기존 테스트 기여; 새 Gate 결과 아님 |
| P-03 | 카드별 run을 lean snapshot 계약에 맞춤 | [f263626](https://github.com/krafton-jungle-project-4team/loop-ad_decision/commit/f2636263bfdda7eba5580193dcdc1dabd96f23b9) | 기존 application·contract 연결 |
| P-04 | lean 사용자군 lifecycle actual DB 통합 테스트 | [fa10316](https://github.com/krafton-jungle-project-4team/loop-ad_decision/commit/fa103168402c712bb53a0b912f15572d2622b9c6) | 기존 실제 DB 테스트; 이번 새 요청/commit Gate와 구분 |

run 서비스 전체, Dashboard, canonical schema, 배포 시스템을 혼자 만들었다고 쓰지 않는다. next-loop 재시도·동시성 관련 기존 기여가 있더라도 이번 Gate의 실제 동시성 검증이 완료됐다는 근거로 전용하지 않는다.

## 이번 과제에서 새로 한 일

| ID | 수행 내용 | 상태·근거 |
|---|---|---|
| N-01 | fake/actual DB/consumer 경계를 조사하고 개선 과제 선택 | 조사 완료 — E-04~E-07 |
| N-02 | 고정 필수·최신 경고, 컨테이너, baseline fixture, PR 전략 합의 | 설계 완료 — [개발 계획](implementation-plan.md) |
| N-03 | 문서 7개를 새 worktree에서 작성 | 문서 작성 — [안내](README.md) |
| N-03a | 문서 7개만 Git 변경으로 보이도록 .gitignore 예외 추가 | 기존 *.md 제외를 유지하고 파일별 예외 7줄 추가 |
| N-04 | Gate 실제 DB 테스트·runner·fixture | 전체 로컬 Gate·고정 기존 row fixture 완료 — E-12; CI는 PR 2 |
| N-05 | 로컬·CI 검증과 성능/신뢰성 측정 | 로컬 정상·실패 주입 E-12; CI·장기 신뢰성·생산성 비교 미측정 |

## 주요 결정의 이유

| 결정 | 선택 이유 | 수용한 비용·한계 |
|---|---|---|
| 실제 API → DB 먼저 | 기존 fake와 직접 SQL integration 사이의 공백을 보충 | Dashboard까지 보장하지 않음 |
| snapshot 기반 V2 run 우선 | 현재 source/binding 경계에 집중 | legacy 전체 제외 |
| 순차 retry 먼저 | 초기 구현·fixture·transaction을 좁은 범위에서 완성 | 실제 동시성은 필수 후속 |
| baseline 생성 row 고정 | 새 writer와 reader가 같은 방식으로 틀리는 위험 줄임 | provenance와 기준 유지 비용 |
| fixed/latest 분리 | 안정된 회귀 기준과 외부 변화 탐지를 함께 제공 | DB 검사 시간·네트워크 의존 증가 |
| runner와 DB 컨테이너 | 로컬·CI 실행 환경을 가깝게 맞춤 | image·lock·cleanup 관리 |
| 개인 통합 PR 2개 후 dev Draft | 로컬 도구와 CI를 나눠 검토하고 통합 검증 후 제출 | 순차 선행 관계 존재 |

## 앞으로 기록할 실행 근거

아래는 비어 있는 기록 양식이며 실제 실행 사실이 아니다.

| 필드 | 현재 값 |
|---|---|
| 실행 날짜·실행 ID | 미실행 |
| 실행 목적·명령 | 미실행 |
| 후보 commit·allowlist digest·dirty 여부 | 미기록 |
| baseline producer / fixed / latest SHA | 실행 결과 미기록 |
| dependency lock·image digest·architecture | 미검증 |
| fixed verdict·필수 case 수·누락/skip | 미실행 |
| latest verdict·경고 분류 | 미실행 |
| JSON·JUnit·CI run URL | 없음 |
| cleanup 결과 | 미실행 |
| 관찰한 문제와 분석 | 새 실행에서 채울 항목 |
| 다음 행동 | 새 실행에서 채울 항목 |

실행이 생기면 기록 양식을 날짜별 subsection으로 추가한다. 최신 결과로 과거 실패를 덮어쓰지 않는다.

## 측정 계획

| 지표 | 측정 방법 | 현재 |
|---|---|---|
| 필수 case 실행·통과·누락 | manifest와 실제 결과 대조 | E-12: RCG/CTRL 필수 전부 통과, 누락 0 |
| cold/warm 소요 시간 | dependency/image 준비와 test 시간을 구분 | 미측정 |
| fixed/latest 각각의 소요 시간 | lane별 monotonic elapsed | E-12 및 result.json |
| 반복 실행 신뢰성 | 실행 횟수·기간·환경 실패 종류를 함께 기록 | 미측정 |
| 탐지한 회귀 | 의도적 실패/자연 발생 결함을 구분해 case·commit 연결 | E-09 서비스 결함, E-12 제어 목적 실패 주입 |
| 개발 편의 개선 | 실제 사용 전후의 환경 준비·명령·수동 작업을 비교 | 미측정 |

검증 명령이 하나가 됐다는 사실과 개발 시간 감소율은 다르다. 시간 절감률은 이전 절차의 동일 조건 측정이 있을 때만 계산한다.

## 문서 검수 기록

문서 파일 수, 20개 개발 계획 항목, 상대 링크·고정 commit 링크의 형식, 필수 case 대응, 상태·revision 표기, 금지 경로 변경 여부를 정적으로 확인한다. 이 검수는 서비스 테스트·Gate·CI 성공을 의미하지 않는다.

검수 결과는 작업 완료 시 아래에 기록한다.

- 문서 정적 검수: 문서 7개, 개발 계획 20개 항목, 내부 링크 52개, 고정 commit 링크 12개의 로컬 object/path/line 존재를 확인했다. 실행 명세는 RCG-01~09와 CTRL-01~05에 대응한다. 원격 웹 페이지의 HTTP 응답이나 Gate 실행을 검증한 것은 아니다.
- 최초 문서 검수 시 서비스 테스트·컨테이너는 미실행이었다. 이후 초기 재현 실행은 E-09에 별도 기록했다.
- 최초 문서 검수 시 commit·push·PR·merge는 수행하지 않았다. 이후 승인된 PR 0 작업은 E-11에 기록했다.

## E-09: PR 1 초기 실행과 중단

2026-09-19 KST, 기존 worktree `/Users/ran/loop-ad/loop-ad_decision-run-contract-gate`, branch `feat/run-contract-gate-db`, HEAD `e1de8b29b902b54df3a58f21f1daa27c1171fe80`에서 시작했다. 기존 문서 7개와 .gitignore를 보존했다. 사용자 승인 범위는 PR 1이며 서비스 수정·PR 2·commit·push·PR·merge는 제외다.

### 실제 명령과 결과

```bash
./scripts/reproduce-run-contract-commit.sh   /Users/ran/loop-ad/loop-ad_data-source_contract   /private/tmp/rcg-commit-repro-20260919
```

실행 결과: **1 passed, 1 failed, skip 0, pytest 1.23초, exit 1**. 준비 시간을 포함한 cold/warm 성능은 측정하지 않았다. 다른 case를 반복 실행하거나 전체 suite를 통과했다고 주장하지 않는다.

| 항목 | 근거 |
|---|---|
| 실행 목적 | baseline 정상 commit 및 RCG-07의 지연 제약 오류와 HTTP 응답 순서 재현 |
| producer | e1de8b29b902b54df3a58f21f1daa27c1171fe80의 Git archive; 후보 writer로 대체하지 않음 |
| canonical DDL | 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50의 postgres/schema.sql, SHA-256 bad4948fe47485e7508a3cd389e9db84fdda3edfed4a1f4883b03554bcb38691 |
| 의존성 lock hash | 150c68697cc7eb2ec9df230ec705976a454c6d96b4cf366ec6ff100e778d82af |
| 주요 runtime | Python 3.12.14, FastAPI 0.141.1, Starlette 1.6.0, psycopg 3.3.6, PostgreSQL 16.10, Linux/arm64 |
| Python image | python:3.12-slim@sha256:b699c2a51f4f834fa1a5f7f7cba0356e71d82fae37d4628223b11d93b71e9ebe |
| PostgreSQL image | pgvector/pgvector:0.8.0-pg16@sha256:a132765ec351c65111b5b675928a3a0515a466a40f97277329db8b8209ad8bc9 |
| runner image ID | sha256:3986d52f880eb0887a67ec73b407166531ef9fc7328af64149255c9dd432406d |
| source 상태 | HEAD 유지, uncommitted Gate 파일 있음. 재현 application은 baseline archive이며 source/test hash는 provenance.txt에 기록 |
| artifact | `/private/tmp/rcg-commit-repro-20260919/`: RCG-01.json, RCG-07.json, junit.xml, pytest.log, provenance.txt, database.log, cleanup.log |
| cleanup | 단일 재현 명령의 소유 label 컨테이너·socket volume 삭제 확인. 별도 탐색용 DB·socket도 삭제. 이미지·build cache는 유지 |
| 한계 | 전체 Gate result.json 판정기·manifest·CTRL·latest·고정 기존 row fixture는 미완료 |

### 무엇이 통과했고 무엇이 실패했는가

- **RCG-01 PASS:** A/B/C seed에서 A/B 요청, HTTP 200; 독립 connection에서 run 1개·experiment 2개·binding 2개, 정확한 scope/fingerprint와 ID 관계, plan locked, A/B consumed·C reserved를 확인했다.
- **RCG-07 FAIL:** 테스트에서 binding 동작만 생략. 실제 run·experiment INSERT 뒤 실제 dependency commit에서 `CheckViolation`, SQLSTATE `23514`, `run binding set must match the run segment scope`가 발생했다. 독립 connection의 전체 관찰 row는 요청 전과 같아 rollback은 확인됐지만 HTTP는 200이었다.
- 필수 논리 case 14개(RCG 9 + CTRL 5) 중 2개 실행, 1개 통과·1개 실패·12개 미실행. RCG-05/06 하위 case는 전부 미실행이다.
- **fixed 기준:** 관찰한 RCG-07은 FAIL이다. 전체 fixed lane과 제어 판정기를 완성했다는 뜻은 아니다. **latest:** 미실행, WARN_UNVERIFIED에 해당하며 최신 resolved SHA를 이번 재현 명령이 조회하지 않았다.
- baseline이 생성한 정상 row는 독립 조회 artifact에만 남겼다. RCG-08용 관계 closure export·복원·고정 expected/provenance fixture는 생성하지 않았다. 이를 완료된 기존 row fixture로 취급하지 않는다.

### 재현과 원인 분석

처음 임시 probe로 정상 요청을 확인한 뒤 binding 생략을 주입했다. `raise_server_exceptions=False`에서는 200+빈 run 테이블, `True`에서는 router.py 201행의 실제 `connection.commit()`에서 CheckViolation이 확인됐다. 이를 정식 RCG-01/07 pytest 재현으로 남겨 동일 결과를 확인했다.

바깥 ASGI wrapper가 기록한 순서는 `binding_omitted_for_RCG_07` → `http.response.start(status=200)` → body 완료 → `CheckViolation(23514)`다. 단순히 TestClient가 예외를 숨겼다는 추정이 아니라 실제 응답 전송 이후 예외 순서를 기록했다. 로그도 commit 실패 전에 `http_request_completed` success를 남긴다.

[router dependency](../../app/decision/router.py)의 commit은 yield 뒤에 있고, [POST handler](../../app/decision/router.py)는 scope 없는 `Depends(get_promotion_run_service)`를 사용한다. 설치된 FastAPI 0.141.1의 `dependencies/utils.py`는 이 yield dependency를 request AsyncExitStack에 등록한다. `routing.request_response`는 `await response(scope, receive, send)` 후 그 stack을 종료하므로 성공 응답 이후 commit하게 된다. 실제 PostgreSQL deferred constraint는 올바르게 transaction을 거절했다.

검증된 것은 이 baseline과 고정 runtime 조합이다. 운영 배포의 FastAPI 버전과 실서비스 장애 여부는 확인하지 않았다. pyproject.toml은 `fastapi>=0.115.0`이므로 이번 고정 버전은 허용된 의존성 범위다. 구버전으로 낮춰 테스트를 녹색으로 만들지 않았다.

### 중단 결정과 별도 수정 후보

[개발 계획 18절](implementation-plan.md#18-구현-순서와-중단-기준)의 “서비스 결함은 재현과 원인 분석까지 수행하고 수정 범위를 별도로 결정한다”와 사용자의 같은 지시를 적용했다. 서비스 app/·pyproject.toml·기존 tests/conftest.py는 변경하지 않았다.

별도 승인받을 수정 목표는 **POST /runs 성공 응답 전에 commit 성공을 보장하는 트랜잭션 경계**와 해당 회귀 검증이다. endpoint의 dependency scope를 function으로 지정하는 방법 등을 비교하되, 현재 FastAPI 최소 버전 호환과 같은 dependency를 공유하는 다른 endpoint의 영향부터 검토해야 한다. 아직 방법을 확정하거나 runtime을 수정하지 않았다.

서비스 수정 범위가 결정된 뒤 후보 코드 검증 경로를 구현하고, 나머지 RCG/CTRL·기존 row fixture·fixed/latest·결과 판정·cleanup 예외 검증을 이어간다. 실제 동시성 검증은 **필수 후속 개발**로 유지한다.

### E-09 시점 검토한 정확한 파일 목록

기존 변경 보존: `.gitignore` 및 아래 문서 7개. 나머지 7개는 이번에 새로 작성한 재현 자산이다.

- `.gitignore`
- `docs/run-contract-gate/README.md`
- `docs/run-contract-gate/implementation-plan.md`
- `docs/run-contract-gate/verification-spec.md`
- `docs/run-contract-gate/developer-guide.md`
- `docs/run-contract-gate/learning-guide.md`
- `docs/run-contract-gate/evidence-log.md`
- `docs/run-contract-gate/portfolio-case-study.md`
- `Dockerfile.run-contract-gate`
- `Dockerfile.run-contract-gate.dockerignore`
- `scripts/reproduce-run-contract-commit.sh`
- `tools/run_contract_gate/requirements.lock`
- `tests/run_contract_gate/__init__.py`
- `tests/run_contract_gate/seed.py`
- `tests/run_contract_gate/test_run_db.py`

## E-10: 선행 서비스 수정 PR 0 분리

사용자가 “이 결함은 Run Contract Gate PR 1에 섞지 말고 선행 서비스 버그 수정 PR 0 범위로 분리”하도록 승인했다. 기존 PR 1 작업 공간과 변경을 보존하고, 같은 baseline e1de8b2에서 `/Users/ran/loop-ad/loop-ad_decision-run-commit-fix`, `fix/run-commit-before-response`를 만들었다. PR 0은 `app/decision/router.py`, `pyproject.toml`, `tests/test_decision_run_api.py` 3개 파일만 변경했다. commit·push·PR 생성·merge는 하지 않았다.

수정은 POST /runs dependency의 function scope와 FastAPI 최소 0.121.0이다. 정상 응답 전 commit/close, commit CheckViolation·SerializationFailure의 500/rollback/close 회귀 테스트를 추가했다. 새 테스트 3개는 수정 전 모두 실패했고, 수정 후 run API/service 115개가 통과했다. 최소 FastAPI 0.121.0에서 관련 API/config/service 218 passed·4 skipped이며, skip은 기존 generation PostgreSQL locking opt-in 테스트다.

실제 canonical DB에 PR 0 candidate를 연결한 RCG-01/07은 **2 passed**다. RCG-07은 HTTP 500, 동일한 SQLSTATE 23514, 전체 rollback이다. 별도 baseline producer 프로세스가 만든 row를 candidate 프로세스로 재요청해 같은 응답과 전체 관찰 row 불변도 확인했다. 테스트 기대값·canonical DDL·PR 1 코드는 바꾸지 않았다.

근거: `/private/tmp/rcg-pr0-20260919/PR0-review.md`, `artifacts/`의 red/current/minimum JUnit, `database-candidate/`의 RCG-01/07 JSON·JUnit, `old-rows/produce.json` 및 `consume.json`. 실제 테스트 컨테이너·socket volume은 정리했고 이미지/cache만 유지한다.

PR 순서는 **PR 0 서비스 수정 → PR 1 로컬 Gate → PR 2 CI**로 갱신했다. PR 1의 기존 baseline 재현 명령은 여전히 고정 과거 코드를 검사하므로 실패하는 것이 맞다. PR 0의 PASS를 PR 1 전체 Gate PASS로 기록하지 않는다. 선행 반영과 PR 1의 candidate 검사·나머지 필수 case 구현은 아직 남았다. Dashboard consumer 테스트·배포 smoke·실제 동시성은 이번 검증에 포함하지 않았다.

## E-11: PR 0 병합과 PR 1 기반 반영

사용자가 PR 0을 개인 통합으로 생성·검증·병합한 뒤 기존 PR 1 파일을 보존하고 개발을 계속하도록 승인했다. 각 원격 쓰기 전에 repository·base/head branch·commit 범위·3개 파일·77 additions/2 deletions와 diff를 제시했다.

- 서비스 commit: `ae741026488773b0bd0f404daf59e1f158251233`, branch `fix/run-commit-before-response`.
- 대상: `integration/run-contract-gate`, 기존 dev 기준 `e1de8b29b902b54df3a58f21f1daa27c1171fe80`에서 생성.
- [PR #394](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/394): 위 3개 파일만 확인하고 mergeable 상태를 조회했다. 이 개인 통합 대상에는 실행된 CI check가 없었으며, E-10의 로컬 검증을 근거로 사용했다.
- merge commit: `09442f29e8da514df1d1a5f2a52b03646c92e170`, GitHub merge 완료 시각 `2026-09-19T06:02:47Z`.
- 원격 main/dev·배포 workflow·branch protection은 변경하지 않았고 원격 branch를 삭제하지 않았다.

PR 1의 기존 작업 파일 15개를 `/private/tmp/rcg-pr0-merge-20260919/pr1-working-files.tar`와 hash 목록으로 보존했다. 기존 `feat/run-contract-gate-db`에서 통합 commit을 fast-forward 반영한 직후 15개 hash가 전부 같음을 확인했다. 서비스 수정은 기반 commit에 있고 PR 1 diff에 중복하지 않는다.

반영 직후 RCG-01/07은 2 passed였다. 이어 RCG-02/03/04/05/06/09를 추가해 실제 DB 16개 case가 통과했다. 오류 code 기대값의 오타는 실제 service 상수를 확인해 수정했으며 새 서비스 결함은 없었다. 해당 단계 artifact는 `/private/tmp/rcg-pr0-merge-20260919/recheck/`와 `scenarios-verified/`다.

## E-12: PR 1 로컬 Gate 검증

### 고정 baseline fixture

명령: `./scripts/generate-run-contract-baseline.sh /Users/ran/loop-ad/loop-ad_data-source_contract /private/tmp/rcg-baseline-final-20260919`.

고정 producer e1de8b2의 실제 API가 생성한 합성 row를 export했다. 18개 table의 초기 45행·commit 후 50행과 expected 응답을 보존했다. baseline 코드·DDL·lock·image·seed/생성 코드 hash·파일 hash·생성 시각은 [provenance](../../tests/fixtures/run_contract_gate/baseline/provenance.json)에 있다. 운영 dump·실제 사용자·credential은 포함하지 않는다.

초기 복원 시 DDL 기본 fallback 행 중복과 locked/consumed 직접 INSERT 금지에 걸렸다. DDL 기본 행을 제외하고, baseline의 생성 전/후 이미지를 valid SQL lifecycle 전이로 복원했다. canonical 제약·트리거를 비활성화하지 않았다. 새 DB에서 baseline reader가 같은 응답과 행을 재사용하는 검사는 **1 passed / exit 0**이다. 평소 Gate는 이 고정 fixture를 읽기만 하며 후보 writer로 재생성하지 않는다.

### 정상 실행

최종 후보 명령·digest·시간은 아래 최종 검증 표에 기록한다. PR 1은 기반 commit에 미커밋 Gate 파일을 더한 후보이며 clean 최종 commit 검증으로 표현하지 않는다. 고정·최신 SHA가 같아도 각각 새 case DB를 만들어 실행했다.

- RCG-01~09 논리 9개 → lane별 실제 17개 case. missing source 3개, wrong generation 1개, outside/empty/blank/fallback 4개, 실제 쓰기 후 실패 주입 2개를 모두 포함한다.
- CTRL-01~05 → 실제 30개 case. 실제 pytest subprocess로 skip/xfail/xpass·미수집을 확인하고, 결과/JUnit·SHA/DDL hash 불일치와 cleanup/timeout/중단을 검사한다.
- RCG-07은 HTTP 500·SQLSTATE 23514·run/experiment/binding/target/plan/exclusion member와 revision rollback을 확인한다.
- RCG-08은 고정 baseline 행을 fresh DDL에 복원한 뒤 후보 API의 동일 응답과 전체 export 행 불변을 확인한다.
- Python/pgvector image는 digest 고정, 직접·간접 Python dependency는 버전 lock이다. 패키지 wheel hash까지 고정한 lock은 아니다. 검증 architecture는 Linux/arm64다.

### 의도적 실패와 정리

아래는 서비스 결함으로 해석하지 않는 테스트 전용 임시 복사본의 주입이다. 실제 저장소 runtime·canonical 원본·고정 fixture는 변경하지 않았다. 각 디렉터리에 result.json, 로그, 실행 복사본을 남겼다.

| 주입 | 실제 결과 | artifact 디렉터리 (`/private/tmp/rcg-gate-fault-probes-20260919/` 아래) |
|---|---|---|
| 정상 DB 검사 뒤 assertion 실패 | fixed FAIL, latest WARN_DRIFT, exit 1, cleanup 성공 | `assertion-output/` |
| 최신 ref 조회 실패 | fixed PASS, latest WARN_UNVERIFIED, exit 0, cleanup 성공 | `latest-unavailable-output/` |
| runner timeout | INCOMPLETE, exit 2, cleanup 성공 | `timeout-bounded-output/` |
| 임시 fixed DDL에 잘못된 SQL 추가 | fixed INCOMPLETE, latest PASS여도 exit 2, cleanup 성공 | `schema-invalid-output/` |
| 실행 중 자기 Gate 프로세스에 SIGTERM | INCOMPLETE, exit 2, cleanup 성공 | `interrupt-reviewed-output/` |

첫 timeout 주입은 Docker CLI가 TERM을 proxy한 뒤 대기하여 47초가 걸렸다. CLI의 종료 유예도 제한하고 서버 측 컨테이너를 cleanup하도록 보강한 뒤 같은 유형의 주입은 전체 15초에 INCOMPLETE로 끝났다. 이는 준비 단계도 포함한 두 실행의 관찰값이며 성능 개선율로 환산하지 않는다. TERM 무시 subprocess를 사용하는 제어 case도 추가했다. 중단 시 터미널 요약이 lane 로그로 흘러가는 문제는 원래 출력 descriptor를 보존해 수정했다.

### 보장과 남은 범위

소유 label 조회로 검증 컨테이너·socket volume이 남지 않았음을 확인했다. 이미지/cache·artifact는 의도적으로 남긴다. `.gitignore`의 정확한 문서 예외 7줄과 Gate 자산만 PR 1 변경이며 app/·pyproject.toml·기존 API 테스트·tests/conftest.py·CI workflow에 추가 diff가 없다. PR 1 commit·push·PR 생성은 하지 않았다.

로컬 Gate 범위는 완료했다. PR 2 CI, 개인 통합의 clean 최종 commit 검증, dev Draft, 실제 동시성, Dashboard consumer·배포 smoke, 전체 legacy/migration, 다른 architecture는 완료하지 않았다. 특히 실제 동시 요청은 **필수 후속 개발**이다. 장기 신뢰성·cold build 시간·개발 시간 절감률·운영 장애 감소는 측정하지 않았다.

### 최종 검증 표

| 항목 | 실제 값 |
|---|---|
| 명령 | `./scripts/run-contract-gate.sh /private/tmp/rcg-gate-pr1-verified-20260919` |
| 실행 ID | `rcg-rcg-work.pa0nil` |
| 후보 기반 / dirty | `09442f29e8da514df1d1a5f2a52b03646c92e170` / `true` |
| allowlist source SHA-256 | `e632a860ae5f70c13f4a9c945ace0b636bf3f345edd390cff6bb827b37176803` |
| fixed / latest SHA | 모두 `0ec2cef0290f4659ad21ccc1dd2a20df2801ff50`, 별도 실행 |
| DDL SHA-256 | `bad4948fe47485e7508a3cd389e9db84fdda3edfed4a1f4883b03554bcb38691` |
| lock SHA-256 | `150c68697cc7eb2ec9df230ec705976a454c6d96b4cf366ec6ff100e778d82af` |
| runner image / architecture | `sha256:3986d52f880eb0887a67ec73b407166531ef9fc7328af64149255c9dd432406d` / `linux/arm64` |
| fixed | PASS, 17 passed, 누락·skip 0, 6.929초 |
| latest | PASS, 17 passed, 누락·skip 0, 5.947초 |
| controls | PASS, 30 passed, 누락·skip 0, 7.823초 |
| 전체 / cleanup | PASS / exit 0 / 34초 / 소유 자원 정리 성공 |
| artifact | `/private/tmp/rcg-gate-pr1-verified-20260919/result.json`, `fixed/junit.xml`, `latest/junit.xml`, `controls/junit.xml` |

마지막 실행에 기록된 allowlist 파일 hash와 현재 파일을 모두 대조해 동일함을 확인했다. 이후 문서만 갱신했다. 위 시간은 캐시가 있는 단일 환경의 한 실행이며 통계적 성능·장기 안정성 수치가 아니다. Python AST·Bash 구문·fixture 파일 hash·필수 manifest 17/30·문서 상대 링크 73개·diff whitespace도 확인했다.

## E-13: PR 1 제출 범위

사용자가 PR 1의 commit·push·PR 생성을 승인했고 merge는 명시적으로 제외했다. 제출 대상은 `feat/run-contract-gate-db` → `integration/run-contract-gate`다. 제출 준비 시 31개 파일이 E-12 뒤 보존한 검토 snapshot과 같음을 hash로 확인했다. 서비스·Gate 실행 소스는 유지하고, 문서에서 E-12의 커밋 전 근거와 제출 commit 검증을 구분했다.

제출 commit SHA·clean 상태 재실행 결과·원격 PR 정보는 PR 본문에 기록한다. 이 문단은 실행하지 않은 원격 작업의 완료 근거가 아니며 E-12의 과거 결과를 덮어쓰지 않는다.

## E-14: PR 2 CI 구현과 로컬 검증

2026-09-19 PR #395가 `integration/run-contract-gate`에 `fe7e8e67f58b51dc03779929f7040eea4bc744e1`로 병합됐음을 GitHub에서 확인하고, 원격 최신 통합에서 `feat/run-contract-gate-ci`를 생성했다. 변경은 `.github/workflows/run-contract-gate.yml`과 기존 안내 문서 7개다. 서비스·PR 1 Gate·판정·fixture·배포 workflow·branch protection은 변경하지 않았다.

workflow는 개인 통합·dev 대상 PR과 수동 실행을 선언하며, `ubuntu-24.04-arm`에서 기존 Gate 명령을 실행한다. Gate exit를 그대로 필수 check에 전달하고 latest는 최종 JSON 기반 경고로 표시한다. required와 latest JSON/JUnit은 별도 `always()` artifact로 보관한다. CI context에는 base/head/checkout SHA를 구분한다. `contents: read`와 credential 비보존 checkout을 사용하고, 공식 Actions는 조회한 v7.0.1 commit SHA로 고정했다.

| 검증 | 실제 결과 |
|---|---|
| actionlint v1.7.12 | 공식 release checksum 확인 후 실행, workflow 통과. shellcheck는 설치되지 않아 비활성화 |
| Bash 구문 | workflow에서 추출한 3개 run block의 `bash -n` 통과 |
| CI provenance | 실제 checkout SHA 기록 및 branch 문자열의 shell 비실행 확인 |
| 종료 코드 전달 | 임시 Gate 대역의 0/1/2를 그대로 전달. 실제 서비스 회귀 주입 검사가 아닌 CI 연결 경계 검사 |
| latest 표시 | PR 1 실제 PASS·WARN_DRIFT·WARN_UNVERIFIED JSON과 결과 누락을 입력해 표시·exit 0 확인; 필수 결과를 재계산하지 않음 |
| workflow 구조 | 두 PR 대상·수동 trigger·read 권한·SHA pin·3개 always step·분리 artifact·continue-on-error 미사용 확인 |
| 기존 명령 로컬 재실행 | `./scripts/run-contract-gate.sh /private/tmp/rcg-pr2-local-20260919`, CI와 같은 종료 코드 전달 block 사용 |
| 실제 DB 결과 | fixed 17 passed / latest 17 passed / controls 30 passed, JUnit errors/failures/skipped 모두 0 |
| 전체 / 정리 | PASS / exit 0 / 34초 / cleanup_ok=true; 소유 label 조회로 컨테이너·volume 잔존 없음 |
| 후보 기반 / dirty | `fe7e8e67f58b51dc03779929f7040eea4bc744e1` / true (workflow 미커밋 후보) |
| source SHA-256 | `e632a860ae5f70c13f4a9c945ace0b636bf3f345edd390cff6bb827b37176803` — PR 1과 동일 |
| Contract / architecture | fixed·latest 모두 `0ec2cef0290f4659ad21ccc1dd2a20df2801ff50`, 각각 실행 / linux/arm64 |

로컬 검증 도구·추출한 step은 `/private/tmp/rcg-pr2-validation-20260919/`, 실제 Gate JSON/JUnit은 `/private/tmp/rcg-pr2-local-20260919/`에 있다. 임시 도구와 raw artifact는 커밋하지 않는다. 문서에는 사람이 검토한 결과만 기록한다. 이 34초는 캐시가 있는 단일 로컬 실행이며 GitHub runner 성능 수치가 아니다.

제출 commit에서 clean 재실행 후 그 SHA와 결과를 PR 본문에 기록한다. 실제 GitHub-hosted runner의 실행·익명 Contract 취득·artifact 업로드는 PR 생성 후 확인한다. 실패 경로의 실제 GitHub artifact 업로드, 수동 dispatch, 장기 CI 안정성, 통합 최종 commit, dev Draft는 이 로컬 검증으로 완료 처리하지 않는다. PR 2는 생성까지만 승인됐으며 merge하지 않는다.

## E-15: PR 2 CI 성공과 개인 통합 병합

E-14는 PR 생성 전 정적·로컬 검증 기록으로 보존한다. 이후 [PR #396](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/396)을 생성했고 아래 결과를 확인했다.

| 구분 | 확인한 revision·결과 |
|---|---|
| 제출 head | `0908bb7bec2dc62b3376f3a72c99829d6178f387` |
| clean 로컬 | 동일 head / dirty=false, fixed 17·latest 17·controls 30 passed, exit 0·cleanup 성공, Gate 31초 |
| 로컬 artifact | `/private/tmp/rcg-pr2-clean-20260919/` |
| PR CI | [실행 35428223300](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35428223300), Fixed contract (required) SUCCESS |
| CI 실제 checkout | `a50132a86dad9496d1d9dca1672485f6f5c48793` (`refs/pull/396/merge`), dirty=false |
| CI base / head | `fe7e8e67f58b51dc03779929f7040eea4bc744e1` / `0908bb7bec2dc62b3376f3a72c99829d6178f387` |
| CI 검증 | linux/arm64, fixed 17·latest 17·controls 30 passed, errors/failures/skipped 0, Gate 65초·job 72초 |
| Contract | fixed/latest 모두 `0ec2cef0290f4659ad21ccc1dd2a20df2801ff50`, 각각 실행 |
| source digest | `e632a860ae5f70c13f4a9c945ace0b636bf3f345edd390cff6bb827b37176803`, 로컬·CI 일치 |
| required artifact | [10580205704](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35428223300/artifacts/10580205704), JSON/JUnit·CI context·exit 확인 |
| latest artifact | [10580185759](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35428223300/artifacts/10580185759), 별도 JSON/JUnit 확인 |
| artifact 다운로드 검증 | `/private/tmp/rcg-pr2-ci-35428223300/`; provenance와 실제 checkout·Gate 입력 일치, artifacts_ok/controls_ok/cleanup_ok=true |
| 개인 통합 병합 | 2026-09-19 07:07:11 UTC, GitHub 조회로 MERGED 확인. merge SHA `14e54cda5c4e92cf835a6ddffc5c20bac0d4ea1e` |

PR 2 작업에서는 PR 생성·CI 확인까지만 수행했다. 이 문서화 단계에서 이미 병합된 원격 상태를 조회했다. 위 CI checkout SHA와 이후 개인 통합 merge SHA는 다르며, 개인 통합 SHA를 새로 실행했다고 주장하지 않는다. artifact의 14일 보존 설정은 영구 보관이 아니다. 실패 경로의 실제 GitHub 업로드·수동 dispatch·장기 안정성·동시성·Dashboard 검증은 이 PASS의 의미에 포함하지 않는다.

## E-16: PR 3 milestone 결정과 증거 대장

2026-09-19 사용자가 저장소별 PR 3A/3B를 하나의 PR 3 milestone으로 묶는 문서화를 요청했다. 명칭은 논리 작업 이름이며 원격 GitHub PR/Milestone을 생성한 사실을 뜻하지 않는다.

### 확정한 역할과 조사 근거

- PR 3A / Decision: 실제 DB 동시성 검증과 Gate 결과·응답 artifact 생산.
- PR 3B / Dashboard: 실제 client·실제 공유 변환·실제 launch flow의 소비 검증.
- 실제 downstream 처리·브라우저 전체 E2E와 구분하고, producer/consumer revision·Contract·bundle hash를 하나의 조합으로 기록한다.
- 문서 기반은 PR 2 통합 `14e54cda5c4e92cf835a6ddffc5c20bac0d4ea1e`; 로컬 branch는 `docs/run-contract-gate-pr3-milestone`이다.
- Dashboard 코드는 로컬 `40af537b0e48a26f738f9cbf4dfc5fbcf2055d62`에서 읽었다. client의 fetch/schema, hook 안의 run 변환, launch의 scope 검사·operation 호출을 확인했다. 이는 Dashboard 배포 또는 원격 최신 확인이 아니다.
- 이번 변경은 Decision의 기존 문서 7개에 한정한다. PR 3 runtime·테스트·CI·bundle은 미구현이며 Dashboard 저장소는 변경하지 않았다.

### Revision 조합 대장 — 문서 작성 당시의 미실행 상태

아래 표는 문서 작성 당시의 상태다. 3A 구현 결과는 E-17에 별도 실행 조합으로 추가했다. 이후 소비 실행도 조합별로 추가한다. 실제 값이 없으면 `미실행`을 유지한다. 예시 hash·가상 PR 번호를 실제 근거처럼 넣지 않는다.

| 필드 | 현재 값 |
|---|---|
| milestone 상태 | 계획 확정 / 3A pending / 3B pending |
| PR 3A URL·merge 상태 | 미생성 |
| PR 3B URL·merge 상태 | 미생성 |
| Decision 제출 head / 실제 checkout / source digest / dirty | 미실행 |
| Dashboard 제출 head / 실제 checkout / source digest / dirty | 미실행 |
| baseline producer / fixture hash | 기존 fixture 사용 예정; 해당 실행 검증 미실행 |
| fixed Contract SHA / DDL hash | 기존 고정 SHA 유지 계획; PR 3 실행 미실행 |
| latest Contract SHA / DDL hash | 실행 시 한 번 해석·고정; 미실행 |
| runner image / Python lock / Node lock / architecture | 미실행 |
| 3A Gate run ID / JSON·JUnit / 경합 증거 | 미실행 |
| fixed bundle manifest·response hash / 3B 입력 hash 대조 | 미실행 |
| latest bundle manifest·response hash / 3B 입력 hash 대조 | 미실행 |
| 3B consumer run ID / JSON·JUnit / operation 호출 근거 | 미실행 |
| 각 fixed verdict / latest 경고와 이유 / 공통 cleanup | 미실행 |
| 로컬 명령·소요 시간 / 3A·3B CI URL | 미실행 |
| 보관 경로·artifact 링크·보존 만료 / 재현 방법 | 미실행 |
| reviewer의 조합 일치 확인 / 남은 범위 | 미실행 |

3A CI와 3B CI 안에서 재생성한 producer 실행은 run ID가 달라도 된다. 사용한 Decision source·Contract·환경을 대조하고 **3B가 실제 소비한 그 실행의 bundle hash**를 연결한다. 다른 실행의 hash를 대신 붙이지 않는다. Dashboard 변경 후 과거 소비 결과를 재사용하거나 Decision 최신 branch를 묵시적으로 선택하지 않는다.

최종 완료는 [계획 21절](implementation-plan.md#21-pr-3-milestone-실행-계약)의 체크리스트와 [검증 명세](verification-spec.md#pr-3-검증-경계--3a-구현-3b-계획)를 따른다. 실행 결과가 생기면 이 대장에 추가하고, 과거 E-09~15의 실패·성공 범위를 소급해서 바꾸지 않는다.

문서 검수: 변경 파일은 기존 문서 7개뿐임을 확인했다. 상대 파일 링크 98개·문서 anchor 37개, code fence 및 세로 Mermaid 3개의 기본 구조, diff whitespace를 검사했다. 영향 분류기는 문서 속 ID/DTO/DB 용어로 T0 신호를 냈지만 실행 코드 diff는 없다. 이 검수는 PR 3 테스트·CI 성공의 근거가 아니며 런타임 테스트는 실행하지 않았다.


## E-17: PR 3A 구현과 실제 경합·bundle 검증

### 기반과 변경 경계

문서 [PR #397](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/397)의 CI run `35429683125`가 성공한 뒤 사용자 승인으로 개인 통합에 merge했다. 통합 SHA는 `6de82a36ddc1cf51c88a431d65e84f873ea8f472`다. 이 SHA에서 `feat/run-contract-gate-concurrency`를 생성했다. PR 3A는 test·runner·manifest·artifact workflow·문서만 변경하며 서비스, DDL, baseline expected, Dashboard는 변경하지 않는다. PR 3A merge는 승인·수행 범위 밖이다.

### 실제 경합과 최초 assertion 조사

- RCG-10: A/B가 모두 scope 없음 확인 → A 실제 run/experiment/binding 쓰기 → B INSERT의 blocker PID 확인 → A commit → B insert false·재조회 → 같은 identity·단일 row 집합.
- RCG-11: 같은 경합에서 A가 실제 binding/consumption까지 쓴 후 테스트용 `RunConflictError`를 주입했다. A rollback·409 뒤 B insert true·commit·200. 최종 단일 집합과 소비 상태, exclusion revision 증가가 정확히 1회임을 검사한다.
- RCG-12: 좁은/넓은 scope 선행 순서를 바꾼 두 case. 승자는 success 응답에서 찾고 패자는 명시적 409·rollback이다. 최종 run 1개·승자 scope의 experiment/binding만 남고 다른 고객군은 reserved다.
- 요청별 PID·read committed·timeline·실제 `pg_blocking_pids`·SQL target 반환 row·commit/rollback·HTTP 시작·원본 응답·독립 connection의 최종 row를 기록한다. event 12초, statement 20초, lock 18초, worker join 12초 후 cancel/추가 join 3초, 외부 lane 상한 180초다.

최초 `/private/tmp/rcg-pr3a-first-contention` 실행은 lane별 19 pass/1 fail였다. overlap의 명시적 409 code가 초안에서 가정한 순차 RCG-09 코드와 달랐다. 실제 코드는 `segment_audience_run_binding_invalid`이며 `AudienceSnapshotContractError` → `RunAudienceContractError` → router HTTP 409로 변환됐다. lock 뒤 `consumed` target과 SQL snapshot의 `reservation_count=0`, `every_member_reserved=false`가 검증에서 거절되는 경로다. 문서가 요구한 “명시적 409 + 실제 handler 형태 고정”에 따라 assertion을 정정하고 반대 scope 순서도 추가했다. HTTP 500, 성공 후 rollback, 중복/부분 row, deadlock, 비결정적 timeout, 미변환 conflict는 관찰되지 않았다. 기존 RCG-06/07의 의도적인 fault injection은 기존 회귀 기준대로 유지한다. 서비스 코드를 고쳐 통과시킨 결과가 아니다.

### 초기 통합 실행 — commit 전 후보

| 필드 | 실제 결과 |
|---|---|
| milestone | 3A local verified / 3B pending |
| checkout / dirty | `6de82a36ddc1cf51c88a431d65e84f873ea8f472` / true; 아직 제출 commit 검증이 아님 |
| source digest | `f385e696f634eff61036cc600832a22a6cf5577d2c2152dea8286e39679f4727` |
| 명령 / 보관 | `bash scripts/run-contract-gate.sh /private/tmp/rcg-pr3a-bundle-first` |
| Gate run / 전체 시간 | `rcg-rcg-work.etqrey` / 37초 (로컬 cache 환경) |
| fixed / latest | 각 21 passed / 모두 `0ec2cef0290f4659ad21ccc1dd2a20df2801ff50`; 서로 다른 DDL drift를 관찰한 실행은 아님 |
| CTRL / cleanup | 52 passed / true |
| DDL hash | `bad4948fe47485e7508a3cd389e9db84fdda3edfed4a1f4883b03554bcb38691` |
| baseline producer | `e1de8b29b902b54df3a58f21f1daa27c1171fe80`; 기존 fixture/expected 유지 |
| lock / architecture | `150c68697cc7eb2ec9df230ec705976a454c6d96b4cf366ec6ff100e778d82af` / `linux/arm64` |
| runner image | `sha256:3986d52f880eb0887a67ec73b407166531ef9fc7328af64149255c9dd432406d` |
| fixed manifest hash | `512ce5b1232a667ec0b39ede9839aabb34beb7e7570561fab5a485972d6aea06` |
| latest manifest hash | `7a019418298de69d2ebe280695a54de567630d56ed5d1b7622bd8f869ad3e990` |
| bundle | 각 12개 원본 HTTP body, case/DB outcome, lane JSON/JUnit·경합 증거; VERIFIED |
| Dashboard / Node lock / 소비 hash | 미실행; 3B pending |
| 3A CI / PR | 제출 전; 최종 clean revision 검증과 PR 본문으로 추가 연결 |

추가 target read 진단 instrumentation과 최종 문서 정리 후 clean 제출 revision에서 전체 Gate를 다시 검증한다. 위 초기 실행의 hash를 후속 실행에 재사용하지 않는다. `consumer/manifest.json`의 `gate_status`/`lane_status`와 `result.json.consumer_bundles`를 함께 확인한다. 전체 raw log·환경변수·DSN은 consumer bundle에 포함하지 않는다.

### 실제 Docker timeout·중단 정리 probe

`/private/tmp/rcg-pr3a-lifecycle/probe-result.json`에 결과를 보관했다. 원래 Gate script의 SHA-256은 `7eec356bc110a79840547b675dee708c519d6e790eaa70f032b06715d8cfdd61`이다. 별도 임시 사본에서 lane 실행 상한만 180→1초로 바꿔 timeout을 주입했다. 사본 hash는 `0312cb421486a62ad381dc147c8dded94aca2126af1d3d285991be9d14cd259a`다. repository script에 시험용 runtime flag를 넣지 않았다.

| 주입 | 실제 run | 결과 |
|---|---|---|
| lane timeout | `rcg-rcg-work.q7j61s` | runner exit 124 → Gate INCOMPLETE/2, cleanup true |
| fixed runner 시작 후 SIGTERM | `rcg-rcg-work.c1toeu` | interrupted 기록 → Gate INCOMPLETE/2, cleanup true |

다른 owner label의 검증용 sentinel container·volume을 두 실행 내내 보존했고 각 실행 후 Docker 자원 inventory가 시작 전과 같았다. 마지막에 probe가 만든 sentinel만 명시적으로 지운 뒤 원래 inventory로 복원됐다. 실제 운영 데이터/컨테이너를 정리 대상으로 사용하지 않았다. 이 두 실행은 의도적 fault probe이며 정상 통과 횟수에 더하지 않는다. CTRL 52개에는 shell owner 확인·timeout/interrupt와 새 실제 worker thread cancel/join·경합 증거 누락·bundle 변조 제어가 포함된다.

### 남은 한계와 후속 연결

TestClient/별도 실제 PG connection 경합이며 TCP 서버·운영 pool·부하·장기간 flaky 비율은 검증하지 않았다. source/manifest hash는 byte 무결성 검사이며 서명이 아니다. Dashboard 3B, 실제 downstream assignment/start/dispatch, 브라우저·배포 smoke는 미실행이다. 따라서 PR 3 milestone 전체 verdict는 incomplete evidence다. PR 3A CI·clean 최종 SHA 결과는 제출 기록에 별도 연결하며 merge로 자동 전환하지 않는다.

### 최종 자체 검토 후보

`/private/tmp/rcg-pr3a-final-review/result.json`의 run `rcg-rcg-work.0xmwwf`는 fixed/latest 각 21 passed, CTRL 54 passed, cleanup true, Gate PASS/0이다. 전체 37초이며 두 lane의 원본 응답 12개씩을 검증했다. 초기 52개 CTRL에 fixed/latest bundle 누락을 실제 finalizer에 전달하는 2개를 추가해 공통 INCOMPLETE/2와 정상 bundle의 최종 verdict 갱신까지 검사했다. source digest는 `baa5da3c26d2a9f6e69d2cdc77eb280a3e526b33f8c5d8c4b40b3826bcf6c9bd`다. 이후 manifest의 JSON 줄바꿈만 정리했으며 clean commit에서 다시 실행한다.

자체 검토는 서비스/DDL/Dashboard 변경 없음, 허용한 파일만 staging, JSON·JUnit·실제 응답과 producer 연결, 명시적 409와 최종 row/소비 상태, bounded wait와 owner cleanup, 실패 artifact 보존을 확인했다. 문서 상대 파일 링크 104개·anchor 39개와 세로 Mermaid 3개의 기본 구조, Python compile·shell syntax·diff whitespace도 검사했다. Mermaid 렌더 QA·운영 부하 시험의 근거는 아니다.

### Clean 구현 commit 재검증

| 필드 | 실제 값 |
|---|---|
| 구현 commit / dirty | `187432d49f83b6bb488094ef26a4057a7c478532` / false |
| 실행 명령 | `bash scripts/run-contract-gate.sh /private/tmp/rcg-pr3a-clean-submit` |
| run / 시간 | `rcg-rcg-work.cxrjhi` / 37초 |
| 결과 | fixed 21 passed / latest 21 passed / CTRL 54 passed / PASS·exit 0 / cleanup true |
| source digest | `2b83b5433f7271559ffb23fc69b1d3221850816abcc3505e068ba649fcba19f1` |
| fixed bundle hash | `03f0aff8eadfbf2bc5fb6ede33800e74368fa977cd603d741394ae7bef0aabd6` |
| latest bundle hash | `587b685e31d9b5d821be700b2e73f46a3f66467d4519b5f1f5388454a4d13c45` |
| 응답 수 | 각 12개, bundle VERIFIED |

같은 경합에서 target read 진단을 확인했다. 두 overlap 모두 후행 요청의 `audience_reservation_state=consumed`, `reservation_count=0`, `every_member_reserved=false`를 실제 `_load_binding_target` 반환값으로 기록했다. 최종 응답은 409이고 rollback 완료가 HTTP 응답 시작보다 앞선다.

이 검증 기록을 추가하는 후속 commit은 문서 7개만 바꾼다. 검증한 구현 commit과 문서 기록 commit을 구분하고, Gate allowlist의 source/test/tool/workflow 변경이 없는지 확인한다. 최종 제출 head와 CI checkout SHA·source digest·artifact 링크는 PR 본문에서 이 로컬 근거와 연결한다. PR 3A는 merge하지 않고 PR 3B는 pending으로 남긴다.

### PR #398 CI와 업로드 artifact 검증

[PR #398](https://github.com/krafton-jungle-project-4team/loop-ad_decision/pull/398)은 `feat/run-contract-gate-concurrency` → `integration/run-contract-gate`, OPEN이며 merge하지 않았다. [Actions 35430880753](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35430880753)의 `Fixed contract (required)`가 SUCCESS다.

| 필드 | 실제 값 |
|---|---|
| PR head / base | `c333c83730a5825152cbb48f73900e30b0ab60cf` / `6de82a36ddc1cf51c88a431d65e84f873ea8f472` |
| 실제 CI checkout / dirty | `be3b5024e43eb1b9d01fbcba01413a352374be6d` / false (`refs/pull/398/merge`) |
| Gate run / 시간 | `rcg-rcg-work.xgqchw` / Gate 74초, job 83초 |
| 결과 | fixed 21 / latest 21 / CTRL 54 모두 pass; cleanup·artifacts·controls true |
| source digest | `2b83b5433f7271559ffb23fc69b1d3221850816abcc3505e068ba649fcba19f1`; clean 로컬 구현 실행과 동일 |
| fixed manifest | `431408e4e530a95114494da69b28f588164ea92d1691513a9c66ae24b8b71ae4` |
| latest manifest | `519c8164c29f3939eda335910b4336e22cc626be0f987a8ca57adedbf9b1a56f` |
| required artifact | [10580128654](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35430880753/artifacts/10580128654) |
| latest artifact | [10580103725](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35430880753/artifacts/10580103725) |
| 다운로드 검증 | `/private/tmp/rcg-pr3a-ci-35430880753/`; 두 bundle의 파일 inventory·digest·lane·producer·JSON/JUnit·원본 응답 12개씩 재검증 |

CI metadata의 PR head/base와 실제 checkout SHA를 구분하고, Gate producer가 checkout과 일치함을 확인했다. 위 수치는 이 CI 실행의 값이다. 이 기록 및 포트폴리오의 과거 미래형 표현을 정리하는 후속 commit은 문서만 바꾸며 최종 head CI는 PR check/본문에서 확인한다. PR 3 milestone은 **3A verified / 3B pending**이고 전체 consumer 경계는 incomplete evidence다. artifact 보관 기간은 workflow의 14일이며 이후 필요하면 같은 명시적 revision으로 재생성하고 새 run/hash를 기록한다.


### PR 3B merge producer 고정 및 RCC-04 중단 (2026-09-19)

**당시 중간 상태: `3B stopped / fix pending`.** 아래는 중단 당시 기록이며, 후속 사용자 지시에 따른 stacked 검증 결과는 다음 절에 분리한다. 중복 experiment identity는 별도 Dashboard `fix/reject-duplicate-experiment-identity` 브랜치에서 수정한다. 3B의 기대값·변환 추출·재현 evidence는 보존하며 이 로컬 문서는 commit·push하지 않는다. 수정 PR merge와 3B 전체 검증 완료 뒤에만 최종 revision 조합으로 갱신한다. 별도 [Dashboard 수정 PR #246](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/pull/246)의 head는 `b77b90165682e4dc9bb63bf93b04f19133e2965d`이며 생성 후 merge하지 않았다. 이 SHA는 수정 PR 참고 정보이며 최종 consumer 검증 revision이 아니다.

이 기록은 위 PR #398 OPEN 시점의 후속 상태다. 사용자가 PR #398을 2026-09-19 08:32:15 UTC에 `integration/run-contract-gate`로 merge했으며 merge SHA는 `9ace3b6ef5d1aaa7851d7ffbb180f1802f5c7008`이다. [최종 PR head CI 35431051367](https://github.com/krafton-jungle-project-4team/loop-ad_decision/actions/runs/35431051367)는 SUCCESS이고 PR head `71c989d`와 merge SHA의 tree diff는 없다. merge SHA 자체의 Actions 실행은 확인되지 않아 CI 성공을 그 SHA의 직접 실행으로 표시하지 않는다.

| 조합 필드 | 실제 값 |
| --- | --- |
| Decision producer revision | `9ace3b6ef5d1aaa7851d7ffbb180f1802f5c7008`, clean detached checkout |
| Dashboard base revision | `df9daf13b57324d52a0eacb15b33c845a399c79f` |
| Dashboard 실행 상태 | `feat/run-consumer-integration`, uncommitted extraction/probe; 실행 파일별 hash는 consumer JSON에 기록 |
| fixed / latest Contract | 각각 `0ec2cef0290f4659ad21ccc1dd2a20df2801ff50` (같은 revision) |
| producer source hash | `2b83b5433f7271559ffb23fc69b1d3221850816abcc3505e068ba649fcba19f1` |
| Gate run | `rcg-rcg-work.zzd4hw` |
| producer 결과 | fixed 21 / latest 21 / controls 54 PASS, cleanup true |
| fixed bundle manifest hash | `179ada3261572a329bd7437a8e4751a7e8c279018934b10260274ac1d2297223` |
| latest bundle manifest hash | `f8a82b130996fae3e5cd827dda31244473c0399478c734eea9814482f486851a` |
| bundle 검증 | pinned producer validator로 각 lane VERIFIED; inventory, provenance, hashes, JSON/JUnit, 원본 응답 12개 검증 |
| consumer focused probe | 6개 실행: 원본/중복 행 control 4 PASS, 중복 experiment identity 2 FAIL; exit 1 / STOP_REQUIRED |
| consumer CI / artifact / PR | 미구현·미제출. 로컬 JSON/JUnit만 보존. 전체 RCC gate는 incomplete |

Dashboard의 실제 client → 추출된 공유 변환 → 실제 `launchPromotionExperiment`를 loopback HTTP replay로 실행했다. RCG-01 원본에서 `ad_experiments[1].ad_experiment_id`만 첫 번째 ID로 바꾸고 서로 다른 segment IDs를 유지하면 client/launch가 거절하지 않고 build 대역을 1회, 같은 experiment ID의 start 대역을 2회 호출한다. fixed/latest 모두 동일하다. 실험 행 전체를 중복하면 launch가 downstream 전에 거절하므로 행 중복과 identity 중복은 구분한다. 원본 producer가 중복 ID를 반환한 것은 아니며 RCC-04용 파생 입력이다. 원본 채널이 onsite_banner여서 dispatch 호출은 없었다.

원인은 client의 문자열 타입 검사와 launch의 고객군별 개수 검사에 experiment ID 유일성 검사가 없기 때문이다. 해결은 Dashboard 검증 동작 변경이어서 사용자가 지정한 중단 조건에 따라 production 수정 없이 멈췄다. 기존 hook 반환식은 그대로 순수 함수로 추출했으며 AST 동일성, 기존 관련 테스트 31 PASS, web typecheck, 변경 파일 eslint로 확인했다. 실제 assignment/start/dispatch·browser E2E·배포는 수행하지 않았다.

재현 코드와 상세 보고서는 로컬 Dashboard worktree `/Users/ran/loop-ad/loop-ad_dashboard-run-consumer`의 `tools/run-consumer/reproduce-duplicate-identity.ts`, `docs/run-consumer/PR3B-STOP.md`, `docs/run-consumer/evidence/2026-09-19-duplicate-identity/{result.json,junit.xml}`에 있다. parent/body/source hash와 원본·파생 body, 호출 순서·인자, 미구현 목록을 보존했다. producer 전체 출력은 `/private/tmp/rcg-pr3b-producer-9ace3b6`에 있다. RCC-02/03/05/06과 나머지 RCC-01/04, 전체 consumer gate 및 CI/artifact는 완료로 표시하지 않는다. 이 E-17 후속 기록은 `docs/run-consumer-stop-evidence` 로컬 branch의 미커밋 문서 변경이며 고정 producer/보호 branch에는 쓰지 않았다. 중단 조건으로 commit·push·PR 생성은 하지 않았다.


### PR 3B: 미병합 fix 위의 전체 consumer 검증 (2026-09-19)

**현재 상태: `3A merged / 3B stacked verified / Dashboard #246·#247 unmerged`.** 사용자의 후속 지시에 따라 수정 PR을 먼저 merge한다는 이전 조건을 변경했다. 정확한 `b77b901`에서 새 worktree와 `feat/run-consumer-integration`을 만들고, [Dashboard PR #247](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/pull/247)의 base를 `fix/reject-duplicate-experiment-identity`로 지정했다. main으로 rebase·retarget하지 않았다. 아래 성공은 이 명시적 조합의 검증이며 main·배포의 성공을 뜻하지 않는다.

| revision 연결 | 검증한 실제 값 |
| --- | --- |
| Decision producer | `9ace3b6ef5d1aaa7851d7ffbb180f1802f5c7008` (clean checkout에서 새 bundle 생성) |
| producer source SHA-256 | `2b83b5433f7271559ffb23fc69b1d3221850816abcc3505e068ba649fcba19f1` |
| Dashboard fix / PR #247 base | [#246](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/pull/246) / `b77b90165682e4dc9bb63bf93b04f19133e2965d` |
| PR #247 head | `58a2133d3d612b1a23c462295265dc4b6fa7c4fb` |
| 실제 CI checkout / dirty | `5535ffb287aa9d92ace7c65b6f6f9c34d69e16da` / false (`refs/pull/247/merge`) |
| Dashboard source SHA-256 | `c20e803e173279d71711cf5e1b41e76144ed03288be6a137d65061f20238d5ce`; clean local head와 파일별 hash도 동일 |
| baseline producer / expected hash | `e1de8b29b902b54df3a58f21f1daa27c1171fe80` / `0fb6c279a84c65e585cc17edaa0103288f952a7061cb5c4d4de0711814f6c2f9` |
| fixed / latest Contract | 각각 `0ec2cef0290f4659ad21ccc1dd2a20df2801ff50`; 별도 lane이지만 같은 DDL revision |
| DDL SHA-256 | `bad4948fe47485e7508a3cd389e9db84fdda3edfed4a1f4883b03554bcb38691` |
| Node / npm | `v25.2.1` / `11.6.2`; lock와 파일 hash는 artifact `inputs.json`에 기록 |

#### CI bundle과 artifact

[Actions 35438441960, attempt 1](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/actions/runs/35438441960)의 `Fixed consumer (required)`는 SUCCESS다. job은 2026-09-19 10:49:32~10:51:41 UTC에 실행됐다. 실행 시작·종료 시 #246이 OPEN, 미병합이고 head가 고정 SHA와 같음을 구조화 자료로 보관했다.

| CI 실행 필드 | 실제 값 |
| --- | --- |
| producer run | `rcg-rcg-work.ec2rlb` |
| fixed bundle manifest SHA-256 | `2320861c624832e9fd71d1256b1639b2c53247fe477aef70adac63a7caf87b76` |
| latest bundle manifest SHA-256 | `addaf58d9895f3107405d05a5704ec5b5760d0d6904c466a557ab6b7689d6a8c` |
| producer 결과 | fixed 21 / latest 21 / controls 54 PASS, cleanup true |
| consumer 결과 | fixed 34 / latest 34 / controls 23 PASS, exit 0, cleanup true |
| 기존 관련·변환/wiring 테스트 | 40 PASS; workspace(shared/api/web)·consumer typecheck PASS |
| artifact | [run-consumer-gate-35438441960-1 / 10583515965](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/actions/runs/35438441960/artifacts/10583515965) |
| 업로드 ZIP digest | `sha256:99f72129caedc0636ceff2b323ab56a9a7d5427555e4f80720bee05bbd0fac76` (Actions metadata) |
| 내부 artifact-manifest.json SHA-256 | `3a93d63b494f9c970ccc54a560aa30b83c379329e62f02d90d6ba2549c236a0f` |
| 다운로드 후 검증 | 164개 파일 inventory/hash, 두 bundle provenance·원본 응답·JSON/JUnit·controls VERIFIED |

검증 명령은 Dashboard의 `python3 tools/run-consumer-gate/verify.py /private/tmp/rcc-ci-35438441960 --producer-checkout /private/tmp/rcg-decision-producer-9ace3b6`이다. 외부 producer validator의 소스 파일도 고정 commit의 실제 파일과 대조했다. artifact 보관 기간은 14일이다. hash는 무결성 연결이며 서명이나 운영 배포 증거가 아니다.

최종 clean local head에서도 `bash scripts/run-consumer-gate.sh /private/tmp/rcc-resumed-complete-head`로 producer부터 새로 실행했다. producer run은 `rcg-rcg-work.nvbvgv`, fixed manifest는 `f2c380ace942697ba75a6d69796df731f95000f55f27eef5af651858aa80c8ad`, latest manifest는 `27f9e73e70774c27df5e70551f57fb36d6c93ecc2b0d411ffbb6bf05808cf5b3`이다. consumer fixed/latest 각 34, controls 23 PASS와 164개 artifact 파일을 재검증했다. 내부 artifact manifest hash는 `d4561d44fdfdda46e22115463ec0b8649744f900c35a0aa5d49355960764c4d8`이다. 로컬과 CI는 별도 run이므로 서로의 bundle hash를 대신 쓰지 않는다.

#### RCC 범위와 중단 재현의 전후 비교

실제 Dashboard HTTP client → hook과 공유하는 순수 변환 → 실제 `launchPromotionExperiment`를 loopback HTTP로 실행한다. downstream build/start/dispatch만 호출 인자를 기록하는 대역이다. RCC-01 2개, RCC-02 7개, RCC-03 1개, RCC-04 8개, RCC-05 3개, RCC-06 13개로 lane당 34개다. 각 lane의 원본 응답 12개를 모두 소비하며 정상·retry·동시 성공·baseline·실제 409와 파생 오류 DTO/scope/ID·fallback 분기를 구분한다. 요청과 변환 결과의 scope/ID, 호출 순서·인자·미호출, 거절 단계를 JSON/JUnit으로 검증한다. baseline 기대값은 producer 출력에서 다시 만들지 않고 고정 expected 파일을 사용한다.

중단 당시와 동일한 파생 중복 ID body SHA-256 `18a1d5a32de8d8e320f9ae17b59fbe17eeb12af8f67343a4a5897ee35ac6a34a`를 원래 bundle로 재실행했다. 이전에는 build 1회/start 2회였고, `b77b901` 위에서는 실제 launch가 거절해 build/start/dispatch가 모두 0회였다. 원본 producer 응답의 결함으로 서술하지 않으며 RCC-04 기대값도 완화하지 않았다. #247의 production 변경은 hook의 기존 반환식 추출 두 파일뿐이고, base 반환식과 AST 동일성 및 hook wiring을 검증했다. client/launch의 추가 동작 변경은 없다.

원래 중단 worktree와 `PR3B-STOP.md`, JSON/JUnit 및 미커밋 추출은 보존 브랜치 `wip/run-consumer-integration-stopped-20260919`에 남겼다. 원본 중단 JSON hash는 `45541ded4f8cf977ba91716361245171f427e919180ae23021c79fa0d5e375d1`, JUnit hash는 `adc4a6ed829ce49bc1e1c2f3129ef46d748013f27befe1b371507775aa958d8e`다. 기존 Decision `docs/run-consumer-stop-evidence`의 미커밋 문서도 그대로 보존했다. 최종 검증은 별도 clean checkout에서 수행했다.

#### CI 실패 이력과 증거의 한계

초기 [35438021217](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/actions/runs/35438021217)은 workflow의 job-level env에서 허용되지 않는 `runner.temp` context 때문에 실행 전 실패했다. `5416a22`에서 해당 env를 step으로 옮기고 actionlint로 검증했다. 다음 [35438163351](https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/actions/runs/35438163351)은 테스트·RCC가 통과한 뒤 Linux Docker 출력의 소유권 때문에 임시 디렉터리 정리가 실패했다. 이 실패 run의 PASS JSON은 완료 증거로 채택하지 않는다.

최종 `58a2133`은 producer 원본 출력을 임시 scratch 밖의 별도 보존 경로에 두고, host scratch 정리 결과를 확인한 뒤 최종 verdict를 기록한다. 정리 오류는 INCOMPLETE/2·cleanup false가 되며 실제 PermissionError 주입 control도 추가했다. 이 변경은 검증 도구의 실행·정리 수정이고 서비스 계약·기대값 변경이 아니다. 최종 성공 실행에서 추가 production 계약 불일치·500·비결정적 timeout은 관찰하지 않았다.

실제 downstream assignment/start/dispatch, 브라우저 E2E, 배포·발송, 운영 부하 및 장기간 flaky 비율은 검증하지 않았다. Decision producer와 Dashboard consumer의 제한된 경계 검증이 완료된 것이며 PR 병합이나 main 승인을 대신하지 않는다. #246 변경 또는 merge가 생기면 이 조합과 차이를 먼저 보고하고 자동 rebase·retarget하지 않는다. 본 Decision 후속 PR은 `docs/run-contract-gate/evidence-log.md` 한 파일만 변경하며 producer 구현·bundle을 변경하지 않는다.
