# Run Contract Gate 개발자 사용 안내

| 항목 | 내용 |
|---|---|
| 대상 독자 | Decision 변경을 검증하는 개발자·리뷰어 |
| 상태 | PR 0 개인 통합 병합 완료 · PR 1 로컬 구현·검증 완료 · PR 2 CI 미구현 |
| 기준 revision | 후보 기반 09442f29e8da514df1d1a5f2a52b03646c92e170 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |

## 1. 준비할 환경

host에는 로컬 Docker daemon, Git, Bash와 기본 Unix 도구가 필요하다. Python·pytest·DB client는 전용 이미지에 포함한다. Linux/arm64에서 검증했으며 다른 architecture·CI는 아직 검증하지 않았다.

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
fixed: SHA=0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 cases=17 reason=
latest: SHA=0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 cases=17 reason=
```

| 결과 | 의미 |
|---|---|
| fixed PASS / exit 0 | 고정 계약의 17개 case와 필수 제어 검사를 통과하고 결과·정리 확인 |
| fixed FAIL / exit 1 | 실제 assertion 불일치 확인 |
| fixed INCOMPLETE / exit 2 | 준비 실패·필수 case 누락/skip·timeout 등으로 필수 검증 불가 |
| latest PASS | 해석한 최신 SHA에서 17개 case 통과 |
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

CI workflow는 이번 PR 1에 없다. PR 2에서 같은 명령을 개인 통합·dev 대상 PR과 수동 실행에 연결하고, 실패 시에도 JSON/JUnit을 보존할 예정이다. 최소 권한과 익명 Contract 읽기부터 검증한다. 로컬 성공은 GitHub Actions 성공의 근거가 아니다.

개인 통합에 반영된 최종 commit을 로컬·CI에서 재검증한 뒤 dev Draft 제출을 준비한다. E-12는 커밋 전 로컬 증거이고, PR 1 제출 commit 재검증 결과는 PR 본문에서 확인한다. 배포 workflow·branch protection·Dashboard·운영 DB는 이번 범위에 포함하지 않는다. 실제 동시 요청 검증은 **필수 후속 개발**이다.
