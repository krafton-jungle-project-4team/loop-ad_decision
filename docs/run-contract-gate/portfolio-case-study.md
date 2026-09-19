# 실제 PostgreSQL 경계를 검증하는 Run Contract Gate

| 항목 | 내용 |
|---|---|
| 대상 독자 | 채용 담당자, 백엔드 면접관, 프로젝트 리뷰어 |
| 상태 | PR 0·1·2·milestone 문서 통합 완료 · PR 3A clean 로컬 PASS · 제출/CI는 PR 기록 · PR 3B pending |
| 기준 revision | PR 3A base 6de82a36ddc1cf51c88a431d65e84f873ea8f472 / baseline e1de8b2 / Contract 0ec2cef0290f4659ad21ccc1dd2a20df2801ff50 |
| 마지막 확인 | 2026-09-19 KST |

## 한 문장 설명

LoopAd Decision의 run 생성 코드가 실제 PostgreSQL 계약에서도 같은 고객군 요청을 안전하게 재사용하고 실패 시 부분 데이터를 남기지 않는지, 로컬에서 반복 확인하는 검증 도구를 구현했다. 같은 명령의 PR CI workflow도 구현했고 정적·로컬 검증을 마쳤다. 실제 Actions에서도 고정·최신 검사와 분리 artifact 보관을 확인했다. PR 3A에서 실제 DB 경합·원본 응답 bundle까지 구현하고 로컬 검증했다. Dashboard 소비 경계는 PR 3B 후속 계획이다.

초기 RCG-07에서 응답 전송 뒤 commit이 실패하는 결함을 재현했다. 별도 서비스 PR #394로 수정·개인 통합 병합한 뒤 고정·최신 lane 각각 17개, 제어 30개가 통과했다. 고정 baseline row 복원·재사용과 단일 실행 명령도 검증했다. 아래 수치는 커밋 전 로컬 후보의 결과다. 제출 commit의 clean 로컬·CI 근거는 E-15에 별도로 기록했다. 운영 효과는 측정하지 않았다.

## 문제: 기존 테스트와 실제 저장 경계 사이

광고 실험을 시작할 때 선택한 고객군별로 run과 experiment를 만들고 snapshot·allocation에 연결한다. 같은 요청이 반복돼도 다른 실험이 늘어나면 안 되고, 중간에 실패하면 부분 row와 소비 상태가 남으면 안 된다.

기존 service 테스트는 fake repository로 scope와 재사용 분기를 검사한다. API wiring 테스트도 RecordingConnection으로 commit 호출을 확인한다. 실제 PostgreSQL lifecycle 테스트는 있지만 run을 직접 INSERT하고 마지막에 rollback한다.

따라서 “실제 요청 → 실제 service/repository → canonical DDL → commit 후 재조회”를 하나의 반복 실행 절차로 검증할 여지가 있었다. 이는 코드·테스트를 읽어 발견한 검증 공백이며 실제 운영 장애가 발생했다는 주장은 아니다. [조사 근거 E-04~E-07](evidence-log.md#현재-조사-기록)

## 내 기여를 구분해서 설명하기

| 구분 | 설명 | 근거 |
|---|---|---|
| 이전 본인 기여 | scope 멱등성과 관련 테스트, lean snapshot 계약 연결·실DB lifecycle 테스트 | [P-01~P-04](evidence-log.md#이전부터-존재한-기능과-본인의-기존-기여) |
| 팀과 기존 시스템 | Decision·Dashboard·canonical DDL·배포 구조 | 실제 코드의 소유 경계를 존중; 전체 시스템 단독 구현 주장 제외 |
| 이번 설계 기여 | 실제 DB Gate 경계, baseline fixture, fixed/latest 판정, 컨테이너·PR 전략, 검증 명세 | [개발 계획](implementation-plan.md), [검증 명세](verification-spec.md) |
| 초기 구현·진단 | RCG-01/07과 합성 seed·컨테이너 재현 명령 | E-09: 1 passed / 1 failed; 서비스 미수정 |
| 로컬 Gate 구현 | RCG-01~09, 제어 판정·cleanup, baseline fixture·provenance | E-12: fixed/latest 각 17 passed, controls 30 passed |
| 선행 서비스 수정 | 응답 전에 commit·close, commit 실패 회귀 검사 | PR #394, E-10/E-11 |
| CI 연결 구현 | 동일 명령·종료 코드 전달, latest 경고, 분리 artifact | E-14/E-15: 정적·로컬·실제 CI 검증 |
| PR 3A 구현 / 3B 계획 | 실제 동시성 4개·원본 응답 bundle·무결성 제어, 후속 consumer 대장 | E-17: 로컬 각 21개·CTRL 54개, timeout/중단 정리 확인; 3B 미실행 |

## 선택한 설계와 이유

- **실제 DB 경계부터:** 외부 AI·전체 Dashboard E2E를 묶기 전에 가장 직접적인 persistence 계약을 확인한다.
- **실제 transaction 관찰:** HTTP 응답뿐 아니라 새 connection에서 commit 결과와 rollback 상태를 읽는다.
- **고정된 기존 row:** 현재 writer와 reader가 함께 바뀌어 과거 호환성 오류를 놓치는 일을 줄인다.
- **고정 필수·최신 경고:** 재현 가능한 기준을 유지하면서 외부 DDL 변화도 드러낸다.
- **별도 runner image:** 로컬·CI의 실행 환경을 가깝게 맞추되 의존성과 image 고정 비용을 수용한다.
- **PR check부터:** 검증기 안정성의 증거 없이 배포 차단 효과를 주장하지 않는다.

단순한 SQL suite 재실행은 application 경계를 놓치고, 모든 서비스를 즉시 연결하는 방식은 첫 과제의 비용을 키운다. 선택하지 않은 대안과 비용도 [개발 계획 5절](implementation-plan.md#5-대안과-선택-이유)에 남겼다.

## JD와의 연결

| JD 관심사 | 이 과제의 증거 | 표현의 경계 |
|---|---|---|
| 안전한 개발 환경·자동화 도구 | 실제 DB 검증과 단일 명령·JSON/JUnit·실패/정리 제어 | 기존 로컬·CI 및 artifact 확인; PR 3A 로컬 PASS·3B 계획 |
| 배포·모니터링·장애 대응 | 배포 전 계약 검증과 실패 원인 분리 | 운영 모니터링·실제 장애 대응 경험으로 바꾸지 않음 |
| 기존 동작 분석·호환성 검증 | 기존 테스트 분석, baseline row와 후보 reader | 전체 legacy migration 수행 주장 제외 |
| AI 코딩 도구 활용 | AI 변경을 사람의 명세와 deterministic assertion으로 검증하는 절차 | 도구를 만들기 전 생산성 향상 수치 주장 제외 |
| DB·웹 서비스·테스트 역량 | scope identity, UNIQUE, transaction, deferred constraint, API/DB integration | 실제 증거를 설명할 수 있어야 함 |

## 검증 결과 — E-12 로컬 후보의 과거 기록

| 질문 | 현재 |
|---|---|
| 어떤 commit을 검사했는가? | 후보 기반 09442f2 + 미커밋 Gate 소스 digest; baseline e1de8b2 / Contract 0ec2cef |
| 몇 개 필수 case가 통과했는가? | RCG 9 + CTRL 5 논리 ID 전체, fixed/latest 각각 17개·제어 30개 통과 |
| 어떤 문제를 확인했는가? | binding 생략 주입 시 HTTP 200 전송 이후 commit 실패·rollback; 기존 baseline/고정 FastAPI 조합 |
| fixed/latest 결과는 어떠했는가? | fixed PASS, latest PASS, exit 0; 의도적 실패 exit 1 / timeout exit 2도 확인 |
| 로컬·CI에서 걸린 시간은? | 캐시가 있는 Linux/arm64 환경 전체 34초; fixed 6.929초·latest 5.947초; CI 미실행 |
| 준비 절차나 수동 작업이 얼마나 줄었는가? | 비교 측정 없음 |
| 남은 검증 범위는? | 실제 동시성·Dashboard 소비 경계 등 |

위 표는 [E-12](evidence-log.md#e-12-pr-1-로컬-gate-검증)의 당시 기록이다. 이후 [E-15](evidence-log.md#e-15-pr-2-ci-성공과-개인-통합-병합)에서 clean head 0908bb7 로컬 31초, CI checkout a50132a의 Gate 65초·job 72초 및 두 artifact를 확인했다. 모두 fixed/latest 각 17개·controls 30개이며 PR 3 case는 포함하지 않는다. 테스트 개수를 임의로 늘리거나 운영 효과로 환산하지 않는다.

## 이력서 문장

### 현재 사실로 사용할 수 있는 문장

- LoopAd Decision의 기존 fake·실DB 테스트 경계를 분석하고, run 생성의 실제 commit·재시도·rollback을 검증하는 Contract Gate의 시나리오와 PR 구현 계획을 설계.
- canonical DDL의 고정 필수 검사와 최신 경고성 검사를 분리하고, 기준 코드가 생성한 기존 row fixture로 후보 reader를 검증하는 호환성 기준 수립.

다음 문장은 로컬 구현 증거까지 포함한다.

- 실제 FastAPI·PostgreSQL 경계를 검증하는 Gate를 구현해 고정·최신 계약 각각 17개 DB 시나리오와 30개 제어 검사를 통과하고, commit 실패 후 HTTP 성공 응답이 전송되는 문제를 별도 서비스 PR로 수정.
- 고정 baseline의 생성 전후 행과 expected 응답을 보존하고, canonical lifecycle 제약을 유지한 복원·후보 reader 재사용으로 한 baseline의 호환성을 검증.

PR 2 CI 한 실행의 결과는 확보했다. 장기 안정성·생산성 향상률·운영 개선 수치는 확보하지 않았다. PR 3A의 동시성과 bundle은 로컬 구현·검증 성과로 설명할 수 있다. 3A의 제출/CI revision·결과는 PR 본문에서 별도로 확인한다. Dashboard 소비·운영 개선 수치는 주장하지 않는다.

### 구현 완료 후에만 사용할 문장 틀

- [확인한 검증 공백]을 해결하기 위해 실제 API·PostgreSQL Gate를 구현하고, [최종 commit/필수 case 결과]로 run 재사용·scope·rollback을 검증했다.
- 로컬·PR CI에 동일 검증 명령과 고정/최신 Contract 판정을 적용해 [측정한 준비 절차/실행 시간/탐지 사례]를 확보했다.

대괄호는 제출 전 실제 근거로 바꾸거나 해당 표현을 삭제한다. 미측정 수치를 추정해서 채우지 않는다.

PR 3A 추가 근거: fixed/latest 각 21개·제어 54개, lane별 원본 응답 12개, 실제 timeout/SIGTERM에서 INCOMPLETE와 소유 자원 정리를 확인했다. 이는 E-17의 해당 revision·환경에 한정하며 두 lane이 같은 Contract SHA를 가리킨 실행이라는 점도 함께 설명한다.

## 30초 설명 — 기존 로컬·CI와 PR 3A 로컬 검증 완료

“광고 실험을 시작하는 API에서 같은 요청을 다시 보내도 실험이 중복되지 않고, 실패하면 데이터가 반쯤 남지 않아야 합니다. 기존 테스트가 실제 DB의 commit까지 확인하는지 조사했고, 그 경계를 반복 검증하는 도구를 설계했습니다. 실제 DB 검증을 시작하자 commit이 실패해도 HTTP 성공 응답이 먼저 나가는 문제가 재현됐습니다. 실패를 보존하고 별도 서비스 PR로 수정했습니다. 이후 기존 행 호환성까지 포함한 Gate를 로컬과 CI에서 검증했습니다. 다음 단계는 Decision의 실제 동시성 검사와 Dashboard의 실제 소비 검사를 저장소별 PR로 나누고, 두 revision의 결과를 연결하는 것입니다.”

## 3분 기술 설명의 순서

1. **문제와 기존 기여:** 기존 run scope 구현·테스트를 소개하고, 새 과제와 구분한다.
2. **공백:** fake repository, RecordingConnection, 직접 SQL integration의 차이를 실제 코드로 설명한다.
3. **설계:** 실제 dependency commit 후 독립 connection으로 확인하며 baseline fixture를 보존하는 이유를 말한다.
4. **실패 가능성:** 중간 쓰기 실패, 지연 제약 실패, 누락된 case의 잘못된 PASS를 예로 든다.
5. **Trade-off:** DB 컨테이너만 쓰는 방식보다 재현성 비용을 수용했고, 최신 DDL은 배포를 흔드는 필수 기준으로 쓰지 않았다고 설명한다.
6. **결과와 한계:** E-12 미커밋 후보, E-15 clean head·CI checkout의 차이를 밝힌다. 실제 결과 하나를 제시하고 동시성·Dashboard·운영 효과의 미검증 범위를 덧붙인다.

## 3분 데모 순서

| 시간 | 보여줄 것 | 증명할 내용 |
|---|---|---|
| 0:00~0:30 | 문제와 검사 대상 commit·Contract SHA | 무엇을 어떤 기준으로 검사하는가 |
| 0:30~1:20 | 기본 명령 및 정상·재시도 결과 | 단일 명령의 실제 DB 검증 |
| 1:20~2:10 | 통제된 실패 case와 비통과 결과 | 실패가 녹색으로 숨겨지지 않음 |
| 2:10~2:40 | fixed/latest JSON·JUnit 또는 CI 결과 | 기준과 경고의 분리 |
| 2:40~3:00 | 미검증 범위와 필수 후속 | 보장의 한계를 정확히 설명 |

실행이 3분보다 길면 실제 사전 실행 artifact를 제시하고 “사전 실행 결과”라고 표시한다. 녹화·저장된 결과를 즉석 실시간 실행처럼 보여주지 않는다. 현재 로컬 Gate와 저장된 JSON/JUnit을 시연할 수 있다. CI는 E-15의 실제 실행과 artifact를 제시할 수 있다. PR 3 동시성·consumer 데모는 아직 없다.

## 예상 면접 질문과 답변 방향

| 질문 | 답변 방향 |
|---|---|
| 왜 이 문제를 골랐나? | 본인의 scope·contract 관련 기존 기여와 실제 테스트 공백 연결 |
| 기존 방식의 가장 큰 위험은? | Python 수준 기대와 실제 SQL·제약·commit 결과의 차이 |
| fake 테스트를 없애야 하나? | 빠른 분기 검증으로 유지, 실DB 경계를 보충 |
| AI가 만든 코드를 어떻게 검증하나? | 승인한 불변식·필수 case·baseline expected와 실제 DB 결과 대조 |
| 같은 writer로 만든 fixture는 충분한가? | 순차 retry와 시간에 따른 호환성의 차이; baseline 보존 필요 |
| 동시성은 검증했나? | PR 3A에서 서로 다른 PID·실제 blocker 대기와 commit/rollback을 관찰한 4개 case를 두 lane에서 검증. 부하·운영 pool 검증은 제외 |
| latest 실패를 왜 필수 실패로 만들지 않나? | 고정 기준의 재현성과 외부 변화 탐지 역할을 분리 |
| 자동화가 잘못 PASS하는 것은 어떻게 막나? | case manifest, skip/누락·result corruption 검사, 독립 DB 관찰 |
| 다른 개발자는 어떻게 쓰나? | 개발자 안내의 단일 명령과 result.json/JUnit 사용 |
| 실제 운영 경험과 무엇이 다른가? | 배포 전 검증 준비도를 개선하는 과제이며 운영 사고 대응 근거는 없음 |
| 도구가 배포를 막나? | PR 2에서 PR check workflow를 추가했다. deploy·branch protection은 바꾸지 않음 |
| 어떤 성과를 측정했나? | E-09 결함 탐지, E-12 로컬 실패 주입, E-15 clean 로컬·CI·artifact; 생산성 개선율은 미측정 |
| 왜 PR 3A/3B로 나누나? | writer·transaction은 Decision, client·변환·launch는 Dashboard가 소유. 특정 revision 조합의 입력 hash로 하나의 검증 근거를 연결 |

## 제출 전 확인

- [ ] 구현 완료 문장을 실제 commit과 실행 결과로 뒷받침한다.
- [ ] 기존 기여·이번 기여·팀 기여를 구분한다.
- [ ] 최신 최종 commit의 artifact와 링크가 유효하다.
- [ ] 미측정 숫자·placeholder·실행 전 표현을 제거하거나 명시한다.
- [ ] 실제 운영·장애·매출·CTR·대규모 사용자 처리·ANN 운영 배포를 근거 없이 주장하지 않는다.
- [ ] 실제 동시성·Dashboard·migration의 검증 범위를 정확히 말한다.
