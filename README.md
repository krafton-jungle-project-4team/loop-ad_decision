# loop-ad_decision
loop-ad 프로젝트 사용자 행동 분석 및 마케팅 액션 추천하는 AI레포입니다.

## Docker Compose 실행

AI 분석 서버와 로컬 ClickHouse를 함께 실행합니다.

```bash
docker compose up decision
```

ClickHouse만 실행하려면 다음 명령을 사용합니다.

```bash
docker compose up -d clickhouse
```

서비스 상태와 헬스체크는 다음 명령으로 확인할 수 있습니다.

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8123/ping
```
