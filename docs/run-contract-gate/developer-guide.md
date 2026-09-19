# Run Contract Gate 개발자 사용 안내

| 항목 | 내용 |
|---|---|
| 대상 독자 | Decision 변경을 검증하는 개발자·리뷰어 |
| 상태 | PR 0·1·2·milestone 문서 통합 완료 · PR 3A clean 로컬 PASS · 제출/CI는 PR 기록 · PR 3B pending |
| 기준 revision | PR 3A base 6de82a36ddc1cf51c88a431d65e84f873ea8f472 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |

## 1. 준비할 환경

host에는 로컬 Docker daemon, Git, Bash와 기본 Unix 도구가 필요하다. Python·pytest·DB client는 전용 이미지에 포함한다. Linux/arm64에서 검증했으며 다른 architecture는 아직 검증하지 않았다. CI workflow는 같은 arm64 runner를 사용한다. 정적·로컬 검증은 E-14, 실제 Actions 성공은 E-15에서 확인한다.

처음에는 공개 Python/pgvector 이미지와 pinned Python 의존성을 내려받는다. 매 실행에서 공개 Contract 고정 SHA를 취득하고 main을 한 번 조회해 최신 SHA를 고정한다. 준비 단계는 네트워크가 필요하다. Git credential helper와 사용자 Git config를 끄고, Docker도 비어 있는 임시 config를 사용한다. private registry나 운영 credential은 필요하지 않다.

실제 테스트는 `--network none`인 별도 runner·PostgreSQL 컨테이너에서 공유 Unix socket으로 연결한다. PostgreSQL은 TCP를 열지 않는다. 운영 DSN 입력·기존 DB 선택·pytest case 제외 옵션은 없고, `.env`나 Docker socket도 mount하지 않는다. settings는 명시적인 합성 mapping이다.

## 2. 로컬 기본 실행

저장소 루트에서 다음 명령을 실행한다. 출력 경로는 아직 존재하지 않아야 한다. 인수를 생략하면 새 임시 디렉터리를 만든다.

```bash
./scripts/run-contract-gate.sh /tmp/run-contract-gate-local
```

실행 순서:

1. `app/`의 Python·JSON과 `pyproject.toml`, 지정한 Gate 파일만 임시 공간에 복사한다. DB schema는 저장소에 복제하지 않는다.
2. 이미지·lock, 후보 commit·dirty 여부·allowlist 파일 hash와 전체 digest를 기록한다. build context에는 전용 Dockerfile·lock만 넣는다.
3. 고정 Contract와 실행당 한 번 해석한 최신 SHA의 DDL을 취득한다. 같은 SHA여도 두 lane을 각각 실행한다.
4. fresh DB를 case별로 만들고 고정 lane → 제어 검사 → 최신 lane을 실행한다.
5. 이번 실행의 정확한 이름과 소유 label이 맞는 컨테이너·socket volume만 정리한다.
6. case manifest·각 pytest phase·JUnit을 대조하고 전체 `result.json`, 터미널 요약과 exit code를 남긴다.

fixed assertion 실패 후에도 가능한 경우 latest를 실행한다. 최신 조회 실패는 fixed를 취소하지 않는다. timeout·중단·공통 준비/결과/정리 실패는 성공으로 표시하지 않는다. image·build cache와 결과 artifact는 유지한다.

미커밋 코드도 검사하지만 commit만으로 증거를 식별하지 않는다. 최종 제출 전 clean한 최종 commit에서 다시 실행해야 한다.

## 3. 결과 읽기

정상 실행의 실제 요약 예:

```text
Run Contract Gate: PASS (exit 0)
fixed: PASS; latest: PASS; controls: PASS; cleanup: True
fixed: SHA=0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 cases=21 reason=
latest: SHA=0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 cases=21 reason=
```

| 결과 | 의미 |
|---|---|
| fixed PASS / exit 0 | 고정 계약의 21개 case와 필수 제어 검사를 통과하고 결과·정리 확인 |
| fixed FAIL / exit 1 | 실제 assertion 불일치 확인 |
| fixed INCOMPLETE / exit 2 | 준비 실패·필수 case 누락/skip·timeout 등으로 필수 검증 불가 |
| latest PASS | 해석한 최신 SHA에서 21개 case 통과 |
| latest WARN_DRIFT | 최신 lane의 assertion 불일치. fixed 결과는 유지 |
| latest WARN_UNVERIFIED | 최신 취득·환경·setup 등으로 결론 불가. fixed 결과는 유지 |

공통 제어 검사·결과 무결성·cleanup 실패는 두 lane의 성공 여부와 관계없이 전체 INCOMPLETE다. pytest exit 0만으로 PASS를 만들지 않는다. skip·xfail·xpass·미수집·중복 수집도 필수 case 통과가 아니다.

| 파일 | 내용 |
|---|---|
| `result.json` | 최종 판정, 실행 ID, 입력 provenance, lane·제어 결과, 소요 시간, cleanup |
| `inputs.json` | 후보 commit·dirty·파일별 hash·전체 소스 digest, lock·image·architecture |
| `fixed/result.json`, `latest/result.json`, `controls/result.json` | 수집 이름과 setup/call/teardown 결과 |
| 각 디렉터리의 `junit.xml`, `pytest.log` | case별 traceback·pytest 원문 |
| `fixed/RCG-01.json`, `fixed/RCG-07.json`, `fixed/RCG-08.json` | 정상 저장, commit 실패 순서, baseline 재사용 증거; latest에도 생성 |
| `contract.log`, `build.log`, `database.log`, `cleanup.log` | 취득·환경·DB·소유 자원 정리 진단 |

준비 초기에 Docker 자체를 사용할 수 없으면 가능한 최소 INCOMPLETE JSON과 로그만 남는다. 실행하지 못한 검사의 JUnit을 성공처럼 만들지 않는다. latest JUnit 실패를 fixed 필수 실패에 다시 합산하지 않는다.

## 4. 기존 baseline 재현과 fixture 생성

과거 결함을 재현하는 별도 명령은 고정 baseline e1de8b2를 실행한다. PR 0 이후에도 예상 결과는 RCG-01 PASS·RCG-07 FAIL, exit 1이다.

```bash
./scripts/reproduce-run-contract-commit.sh \
  /path/to/loop-ad_data-source_contract \
  /tmp/run-contract-original-defect
```

후보 코드는 기본 Gate 명령으로 검사한다. 위 baseline SHA나 기대값을 바꿔 과거 실패를 지우지 않는다.

fixture 생성은 평소 Gate 실행과 분리한 수동 절차다.

```bash
./scripts/generate-run-contract-baseline.sh \
  /path/to/loop-ad_data-source_contract \
  /tmp/run-contract-baseline-review
```

생성기는 고정 baseline·DDL로 합성 seed와 실제 POST /runs를 실행해 commit한다. 생성 전후 행·expected 응답을 export하고, 새 DB에 복원해 같은 baseline API가 재사용하는지 검증한다. canonical lifecycle을 지키도록 finalized/reserved 초기 행을 복원한 후 보존된 SQL 변경분을 적용한다. 제약을 끄거나 후보 writer를 실행하지 않는다.

`rows.json`, `expected.json`, `provenance.json`을 검토한 뒤 별도 승인된 기준 갱신 변경에 포함한다. 생성기는 저장소 fixture를 자동 덮어쓰지 않는다. 운영 dump·실사용자 데이터는 사용하지 않는다.

## 5. 실패 조사 순서

1. 후보 digest·Contract SHA·lock·architecture를 확인한다.
2. 취득, image, DDL, seed/fixture, API assertion, 결과 집계, cleanup 중 실패 단계를 찾는다.
3. [manifest](../../tools/run_contract_gate/manifest.json)와 JUnit을 대조한다. 매개변수 case도 모두 필수다.
4. fixture 오류는 canonical DDL과 provenance를 대조한다. 자동 재생성으로 차이를 지우지 않는다.
5. 서비스 결함이면 재현·원인 분석 후 별도 수정 범위를 결정한다. Gate PR에 runtime 수정을 숨기지 않는다.
6. 해당 실패와 영향 범위를 다시 검증하고, 실행하지 않은 범위는 미검증으로 남긴다.

## 6. 자원 정리와 중단

정상 종료·실패·INT/TERM에서 소유 자원 정리를 시도한다. Docker CLI가 TERM을 proxy하며 계속 기다릴 수 있으므로 timeout 후 짧은 유예를 거쳐 CLI를 종료하고 서버 측 컨테이너도 정리한다. `SIGKILL`, host 종료, Docker daemon 장애에서는 자동 정리를 보장할 수 없다.

남은 자원은 `result.json`의 실행 ID·resources와 `resources.txt`를 확인하고, 삭제 전 label `org.loopad.run-contract-gate`가 해당 실행 ID와 같은지 대조한다. 다른 실행의 자원을 prefix로 일괄 삭제하거나 `docker system prune`을 쓰지 않는다. image·cache·artifact는 자동 삭제 대상이 아니다.

## 7. 고정 Contract 갱신

별도 PR에서 이전/제안 SHA의 DDL diff를 읽고 기존 fixture를 두 DDL에 실행한다. 결과와 영향을 검토한 뒤 fixed SHA를 변경한다. 최신 PASS는 자동 갱신 승인이 아니다. 기존 fixture 제거는 별도의 호환성 범위 축소이므로 새 baseline 추가와 구분한다.

## 8. PR 2 CI와 최종 제출

[Run Contract Gate workflow](../../.github/workflows/run-contract-gate.yml)는 `integration/run-contract-gate`·`dev` 대상 PR의 opened/synchronize/reopened와 `workflow_dispatch`를 선언한다. 경로 필터는 없다. 수동 실행은 workflow가 기본 브랜치에도 있어야 사용할 수 있으므로 개인 통합 PR 단계에서는 Actions UI의 실행 가능 여부를 따로 확인한다.

GitHub-hosted `ubuntu-24.04-arm`에서 로컬과 같은 `./scripts/run-contract-gate.sh "$RCG_OUTPUT"`를 실행한다. checkout/upload-artifact는 공식 commit SHA에 고정하고 `contents: read`, `persist-credentials: false`만 사용한다. Contract는 PR 1의 익명 취득 절차를 그대로 사용하며 배포 credential은 필요 없다.

`Fixed contract (required)` check는 Gate의 exit 0/1/2를 그대로 전달한다. FAIL·INCOMPLETE는 check 실패이며, latest WARN_DRIFT·WARN_UNVERIFIED는 별도 `Latest contract (warning only)` step의 경고다. 이 step은 최종 `result.json`을 읽기만 하고 판정을 다시 계산하지 않는다. check 이름의 required는 검증 역할을 뜻하며 branch protection을 설정하지 않는다.

결과 보관 step은 `always()`로 실행한다. 두 artifact를 14일 보관한다.

| artifact 이름 앞부분 | 보관 내용 |
|---|---|
| `run-contract-gate-required-` | CI context·Gate exit, 전체 result/inputs JSON, fixed·controls JSON/JUnit |
| `run-contract-gate-latest-warning-` | latest JSON/JUnit; 필수 검사 통계에 합산하지 않음 |

이름 뒤의 run ID·attempt로 재실행을 구분한다. artifact에서 먼저 전체 `result.json`과 `run-contract-gate-ci.json`을 대조한다. 후자는 PR base/head SHA와 실제 checkout SHA를 구분한다. 일반 PR에서는 checkout SHA가 GitHub의 PR merge commit이며 PR head SHA와 다를 수 있다. Gate의 `inputs.candidate_commit`은 실제 checkout SHA와 일치해야 한다.

JSON·JUnit과 CI 실행 메타데이터만 명시한 경로로 업로드하며 전체 작업 디렉터리·환경변수·원문 로그는 업로드하지 않는다. 준비 실패로 생성되지 않은 JUnit을 만들어내지 않는다. 일반 실패에서도 생성된 결과의 업로드를 시도하지만 runner 강제 종료·job timeout·GitHub artifact 서비스 장애까지 보관을 보장하지는 않는다. artifact 업로드 자체의 실패는 check 실패다.

정적·로컬 검증은 [E-14](evidence-log.md#e-14-pr-2-ci-구현과-로컬-검증), 제출 commit의 clean 재실행·실제 Actions 성공과 개인 통합 병합은 [E-15](evidence-log.md#e-15-pr-2-ci-성공과-개인-통합-병합)에 기록했다. PR 2 CI는 PR의 checkout merge SHA를 검사한 결과이며 이후 개인 통합 merge SHA를 다시 실행한 결과로 바꾸어 기록하지 않는다. 실제 동시성과 응답 bundle은 3A로 구현했다. Dashboard 소비 검증은 3B 계획 범위다.

## 9. PR 3A에서 PR 3B로 넘기는 절차

2절의 같은 Gate 명령이 동시성 검사와 lane별 consumer bundle 생성까지 수행한다. 추가 dependency·환경변수·production flag는 없다. 3B의 실제 consumer 명령은 아직 구현하지 않았다.

### Decision 개발자: PR 3A

1. PR 2가 반영된 개인 통합 SHA를 확인하고 3A 작업 branch를 만든다. 기존 Gate를 실행해 기반 상태를 확인한다.
2. [RCG-10~12](verification-spec.md#pr-3a-실제-db-동시성-검증)의 실제 경합을 실행한다. 요청별 PID·단계·대기·commit/rollback·최종 DB 증거를 확인한다.
3. 정상·retry·baseline·동시 요청의 원본 status/body를 consumer bundle로 기록하고 manifest·hash·누락 제어 검사를 연결한다. 기존 fixed/latest 결과와 섞지 않는다.
4. clean한 제출 SHA에서 로컬·CI를 실행하고 결과·소요 시간·정리를 기록한다. 아직 실패 중인 bundle은 진단용이라고 표시한다.
5. Dashboard에 전달할 자료는 producer SHA·source digest, Contract SHA, bundle/response hash, 생성 명령, Gate result·JSON/JUnit, CI 실행 링크다. 고정 baseline expected는 함께 참조하되 재생성하지 않는다.

### Bundle 생성·검증 명령

```bash
./scripts/run-contract-gate.sh /tmp/rcg-new-output
python3 -m tools.run_contract_gate.bundle /tmp/rcg-new-output/fixed/consumer \
  --lane fixed \
  --producer-sha <검토한-실제-checkout-40자리-SHA> \
  --source-sha256 <result.inputs.source_sha256> \
  --manifest-sha256 <검토한-result.consumer_bundles.fixed.manifest_sha256>
```

`<...>`는 실제 실행에서 검토한 값으로 바꾼다. 검증기는 Python 표준 라이브러리만 사용하고 exit 0=VERIFIED, 2=INCOMPLETE다. **VERIFIED는 bundle의 무결성 판정**이며 Gate 성공을 대신하지 않는다. 출력의 `gate_status`, `lane_status`와 원본 Gate/CI 결과를 함께 확인한다. producer SHA·manifest hash는 같은 bundle의 자기 선언만 믿지 말고 선택한 checkout·CI provenance와 대조한다. latest는 별도로 `--lane latest`와 그 lane의 hash를 사용한다.

`consumer/` 하나에는 manifest·digest, 원본 `responses/*.body` 12개, `result.json`, `junit.xml`, RCG-01/02/08/10/11/12/reverse 진단 JSON이 있다. 요청과 HTTP status/content type은 manifest의 `samples`에 있고 body는 decode·정규화하지 않은 원본 바이트다. RCG-11의 A 응답은 **테스트에서 실제 write 후 주입한 conflict**이며 자연 발생 실패로 표시하지 않는다. RCG-12 두 오류는 실제 runtime 경로에서 발생한 409다. 각 sample의 case outcome·DB assertion 여부를 확인한다.

응답 12개의 구성은 정상 1·순차 재사용 2·baseline 1·경합 4쌍 8이다. 실패 실행은 partial `responses/`와 진단 JSON을 먼저 보존한다. 준비 실패로 생성되지 않은 응답을 만들지 않는다. 필수 bundle 누락·무결성 오류는 공통 INCOMPLETE로 판정한다. latest 자체를 실행하지 못한 경우는 기존 WARN_UNVERIFIED 규칙을 유지한다. CI는 두 lane의 `responses/`와 `consumer/`를 각각 기존 artifact에 보관한다.

### Timeout·중단 검증 범위

CTRL은 실제 shell cleanup/timeout 함수에 Docker 대역을 연결해 이름·owner label 거절을 검사하고, 새 worker 제어는 실제 대기 thread를 cancel 후 join한다. E-17에서는 별도 로컬 probe로 실제 Docker도 확인했다. timeout은 임시 runner 사본의 lane 제한만 180초→1초로 바꿨고, interrupt는 fixed runner 시작 후 원래 Gate process에 SIGTERM을 보냈다. 다른 owner label의 검증용 container/volume은 그대로 남았고 각 Gate 소유 자원만 사라졌다. 두 probe는 INCOMPLETE/exit 2이며 일반 PASS 실행으로 합산하지 않는다. 강제 종료(SIGKILL)·host/Docker daemon 자체 소실은 이 검증 범위 밖이다.

### Dashboard 개발자: PR 3B

1. Dashboard의 실제 base·checkout SHA와 lock을 기록하고, Decision producer SHA를 3A의 검증 revision에 고정한다. 파일이 오래됐으면 branch 이름만 보고 최신으로 간주하지 않는다.
2. 로컬에서는 3A의 검토된 bundle 경로를 입력받는다. CI에서는 고정 Decision SHA를 별도 임시 checkout하고 같은 Gate 명령으로 bundle을 생성한다. cross-repository artifact 다운로드 권한을 암묵적으로 요구하지 않는다.
3. bundle의 schema·lane·case·source revision·파일 hash·Gate 결과를 먼저 검증한다. 재생성 실행은 새로운 run ID·bundle hash를 가질 수 있으므로 그 실행을 새 소비 결과에 연결한다. 과거 실행 hash와 같아야 한다고 가정하지 않는다.
4. 실제 client·공유 변환·launch를 [RCC-01~06](verification-spec.md#pr-3b-dashboard-consumer-검증-계획)으로 검사한다. replay 서버는 원본 응답을 제공하고 downstream operation 대역은 호출 인자를 기록한다. 배포 Dashboard에 요청하거나 실제 발송하지 않는다.
5. 소비 결과에는 Dashboard SHA·dirty/source digest·Node lock, 입력 producer/bundle hash, case별 결과·거절 층·호출 기록, JSON/JUnit, cleanup과 CI 링크를 남긴다.
6. fixed는 필수, latest는 별도 경고로 기록한다. fixed producer/bundle이 없거나 검증되지 않으면 consumer의 성공만으로 전체 PASS를 만들지 않는다.

### 리뷰어: 같은 revision 조합인지 확인

[근거 대장 E-16](evidence-log.md#e-16-pr-3-milestone-결정과-증거-대장)의 한 행이 아래 연결을 모두 가리켜야 한다.

`Decision checkout/source digest → Contract/DDL → Gate run/bundle hash → Dashboard checkout/source digest → consumer result`

PR head와 CI checkout merge SHA, baseline producer와 현재 candidate, fixed와 latest를 각각 구분한다. 3A와 3B의 독립 PASS만 있고 입력 hash가 연결되지 않으면 milestone 완료가 아니다. CI artifact 보존 기간이 끝나기 전에 합성 데이터·민감정보 여부를 검토한 증거를 보존하거나 같은 revision에서 재실행하고 새 근거를 기록한다. 영구 artifact 저장소·업로드 정책은 임의로 추가하지 않는다.

### 실패 조사와 재실행

| 실패 위치 | 조사할 내용 | 다음 행동 |
|---|---|---|
| 경합 준비/관찰 | 실제 connection 분리, barrier 순서, blocker PID, timeout | harness 문제를 분리하고 INCOMPLETE 유지; sleep 연장만으로 통과시키지 않음 |
| DB/응답 불일치 | commit/rollback 시점, 원본 응답, 독립 DB 조회 | 서비스 결함이면 별도 수정 범위로 중단·보고 |
| bundle 입력 | producer/Contract SHA, hash, lane, 필수 case | 맞는 실행을 재생성/전달; latest나 이전 성공으로 대체 금지 |
| client | 실제 HTTP 오류·schema 거절 | 원본 body와 API 계약 확인 |
| 변환/launch | 공유 함수 사용, scope·ID, operation 인자·호출 순서 | 동작 변경이 필요하면 별도 수정 범위; 테스트 기대값 임의 보정 금지 |
| 공통 runner/cleanup | 누락 결과·남은 자기 자원·worker | 필수 INCOMPLETE; 원인 해결 후 해당 조합 재검증 |

생산 코드·DDL·lock이 바뀌면 3A 생산과 연결된 3B 소비를 다시 검증한다. Dashboard만 바뀌면 유효성이 확인된 동일 producer bundle을 소비하는 3B를 다시 검증한다. 어느 경우든 과거 결과를 덮어쓰지 않고 새 revision 조합으로 기록한다.
