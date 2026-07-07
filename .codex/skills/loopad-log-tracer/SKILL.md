---
name: loopad-log-tracer
description: ap-northeast-2의 AWS CloudWatch 로그로 LoopAd dev ECS 서비스 동작을 조사한다. 사용자가 요청 ID, API 경로, 버그 증상, 의심 데이터 식별자, 상태 코드, 에러 문구, 시간대를 기반으로 LoopAd dashboard-api, decision-api, event-collector 문제를 디버깅해 달라고 요청하거나 aws login 후 LoopAd AWS 로그 검색을 요청할 때 사용한다.
---

# LoopAd 로그 추적기

## 개요

관련 CloudWatch 로그 그룹을 조회해 요청 식별자, API, 증상, 데이터, 시간대로 서비스 버그를 좁히고 로그 근거를 보고한다. 절차는 읽기 전용으로 유지하고 시크릿이나 민감한 전체 payload를 노출하지 않는다.

## 로그 그룹

리전은 `ap-northeast-2`를 사용한다.

- `/loop-ad/dev/ecs/dashboard-api`
- `/loop-ad/dev/ecs/decision-api`
- `/loop-ad/dev/ecs/event-collector`

## 필요한 단서

조회 전 최소 하나의 구체적인 단서를 확보한다.

- 요청 ID, trace ID, correlation ID, 로그 라인 일부.
- 특정 API 경로, HTTP method, 상태 코드, 프론트엔드 action.
- 구체적인 버그 증상, 에러 문구, exception 이름, 예상과 다른 동작.
- campaign, creative, placement, user, device, event, record ID 같은 의심 데이터 식별자.
- timezone이 포함된 시간대. 사용자가 상대 시간을 주면 조회 전에 정확한 날짜와 timezone으로 변환한다.
- 의심 서비스 또는 로그 그룹.

단서가 부족하면 가장 작은 유용한 단서를 묻는 짧은 후속 질문을 한다. 사용자가 추가 정보가 없지만 그래도 찾아 달라고 하면 제공된 모든 로그 그룹을 검색하고, 실제로 스캔한 정확한 시간대와 로그 그룹을 명확히 보고한다.

## 작업 절차

1. 로컬 AWS 접근을 확인한다.
   - `aws --version`을 실행한다.
   - `aws sts get-caller-identity`를 실행한다.
   - 자격 증명이 없거나 만료됐으면 `aws login` 실행 전에 사용자에게 확인한다. 로그인 후 `aws sts get-caller-identity`를 다시 실행한다.
   - 기존 profile 선택을 보존한다. profile을 사용하면 모든 명령에 `--profile <name>`을 일관되게 전달한다.

2. 범위를 정한다.
   - 서비스 단서가 있으면 해당 로그 그룹을 고른다. 서비스가 불명확하면 세 로그 그룹을 모두 사용한다.
   - 사용자의 시간대를 `--start-time`, `--end-time` epoch seconds로 변환한다. 사용자가 timezone을 명시하지 않고 LoopAd dev 운영 문맥이면 KST를 사용한다.
   - 먼저 선택적인 query로 시작한다. 첫 조회에서 아무것도 찾지 못했거나 사용자가 전체 검색을 명시적으로 요청한 경우에만 확장한다.

3. CloudWatch Logs Insights로 조회한다.
   - `aws logs start-query`와 `aws logs get-query-results`를 우선 사용한다.
   - 항상 `--region ap-northeast-2`를 포함한다.
   - status가 `Complete`, `Failed`, `Cancelled`, `Timeout` 중 하나가 될 때까지 폴링한다.
   - AWS resource에 쓰기 작업을 실행하지 않는다.

4. 결과를 연결한다.
   - 같은 요청 ID 또는 데이터 식별자를 로그 그룹 전체에서 추적한다.
   - 서비스 간 timestamp를 시간순으로 비교한다.
   - 근거와 추론을 분리한다. 로그가 직접 증명하지 않으면 root cause는 가능성으로 표현한다.
   - 로그가 결론적이지 않으면 무엇을 검색했는지와 불확실성을 줄일 다음 단서를 함께 말한다.

## Query 패턴

아래 예시를 시작점으로 사용하고 실제 로그 형식에 맞춰 field를 조정한다.

요청 ID 또는 정확한 식별자:

```sql
fields @timestamp, @log, @logStream, @message
| filter @message like "REQUEST_ID_OR_IDENTIFIER"
| sort @timestamp asc
| limit 200
```

API 경로와 에러:

```sql
fields @timestamp, @log, @logStream, @message
| filter @message like "API_PATH"
| filter @message like /(?i)(error|exception|timeout|failed|failure|invalid|warn|500|502|503|504)/
| sort @timestamp desc
| limit 200
```

증상 또는 에러 문구:

```sql
fields @timestamp, @log, @logStream, @message
| filter @message like /(?i)SYMPTOM_OR_ERROR_REGEX/
| sort @timestamp desc
| limit 200
```

단서가 부족한데도 사용자가 전체 조사를 요구하는 경우:

```sql
fields @timestamp, @log, @logStream, @message
| filter @message like /(?i)(error|exception|timeout|failed|failure|panic|fatal|unauthorized|forbidden|validation|invalid|500|502|503|504)/
| stats count(*) as count by @log, bin(5m)
| sort count desc
| limit 100
```

전체 집계 query 후 count가 가장 높은 로그 그룹과 time bucket에 대해 sample query를 실행한다.

```sql
fields @timestamp, @log, @logStream, @message
| filter @message like /(?i)(error|exception|timeout|failed|failure|panic|fatal|unauthorized|forbidden|validation|invalid|500|502|503|504)/
| sort @timestamp desc
| limit 100
```

## AWS CLI 형태

epoch 값과 query string을 바꿔 아래 형태로 실행한다.

```bash
aws logs start-query \
  --region ap-northeast-2 \
  --log-group-names /loop-ad/dev/ecs/dashboard-api /loop-ad/dev/ecs/decision-api /loop-ad/dev/ecs/event-collector \
  --start-time START_EPOCH_SECONDS \
  --end-time END_EPOCH_SECONDS \
  --query-string 'fields @timestamp, @log, @logStream, @message | filter @message like "REQUEST_ID" | sort @timestamp asc | limit 200'
```

이후 폴링한다.

```bash
aws logs get-query-results --region ap-northeast-2 --query-id QUERY_ID
```

## 보고 형식

아래 순서로 보고한다.

1. 검색 범위: 로그 그룹, 정확한 시간대, timezone, query term.
2. 발견 사항: 짧은 결론 또는 "찾지 못함".
3. 근거: timestamp, 로그 그룹, 요청/데이터 ID, 핵심 로그 발췌. token, cookie, credential, 개인정보는 redact한다.
4. 로그로 뒷받침되는 경우 가능한 원인과 영향받은 경로.
5. 남은 불확실성과 다음에 필요한 최소 query/input.
