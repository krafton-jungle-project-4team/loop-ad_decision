# Logging Reference

## Purpose

This document defines how Loop-Ad Decision API logs must be written. It adapts
the shared Loop-Ad logging standard from the Dashboard API reference:
https://github.com/krafton-jungle-project-4team/loop-ad_dashboard/blob/main/apps/api-server/docs/reference_logging.md

Read this file before adding, changing, or reviewing logs in this repository.

## Required Shape

All application logs must be single-line JSON records emitted through
`app.logging.log`. Do not use `print`, `console.*`, or direct ad hoc text logs.

Common records include these fields:

| Field | Source | Meaning |
| --- | --- | --- |
| `timestamp` | logger | ISO timestamp |
| `level` | logger | `debug`, `info`, `warn`, `error` |
| `service` | settings | service id |
| `environment` | settings | runtime environment |
| `version` | package | service version |
| `region` | runtime env | cloud region when available |
| `runtime` | runtime env | Python or cloud runtime |
| `operation` | `@log_context_scope` | public use-case function |
| `requestId` | HTTP middleware | request correlation id |
| `event` | log call | stable event name |
| `err` | payload | serialized error |
| `durationMs` | payload | elapsed milliseconds |

Context and payload field names must use camelCase. `event` values must use
lowercase snake_case.

## Logger API

Use the common logger helpers:

```py
from app.logging import log, log_context_scope, now_ms, duration_ms


@log_context_scope
def run_use_case(project_id: str, promotion_id: str, request: object) -> object:
    started_at = now_ms()
    log.assign_context({"projectId": project_id, "promotionId": promotion_id})
    log.info("started", {"request": request})

    response = do_work()

    log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
    return response
```

Use `None` in `assign_context` only when intentionally removing a context field.

## Log Call Formatting

Keep a log call on one line when it is 140 characters or shorter.

```py
log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
```

Use multi-line formatting only when the call exceeds 140 characters or the
payload is large enough that one line is harder to scan.

## FastAPI Placement Rules

HTTP middleware owns request-level logs:

- Ensure `requestId`.
- Add `requestId`, `method`, and `path` to context.
- Emit one `http_request_completed` log with `statusCode`, `outcome`, and
  `durationMs`.

Routers should not normally log. They should rely on HTTP completion logs and
service logs unless they handle streaming/manual responses or perform real
business branching.

Public service/use-case methods must:

- Use `@log_context_scope`.
- Assign known IDs at the start.
- Emit `started`.
- Emit `completed` with `response` and `durationMs`.
- Let the scope emit `failed` for unexpected exceptions.
- Emit `warn` before expected domain failures that become 4xx responses.
- Emit `info` for persisted state changes and meaningful loop/job/group
  boundaries.

Repositories should not log ordinary SELECT/INSERT/UPDATE calls. Log only DB
constraint, retry, lock, conflict, dynamic SQL, or multi-step write behavior that
cannot be diagnosed from service logs.

External provider/client boundaries must log:

- `provider_request_prepared` before the call.
- `provider_request_completed` on success.
- `provider_request_failed` on failure.
- `provider_response_invalid` when a provider response cannot be parsed.

Never log API keys, authorization headers, cookies, passwords, session tokens,
refresh tokens, or credentials.

## Event Rules

Use stable event names that describe facts, not function names.

Approved examples:

- `started`
- `completed`
- `failed`
- `promotion_loaded`
- `promotion_not_found`
- `provider_request_prepared`
- `provider_request_completed`
- `provider_request_failed`
- `segment_assignments_created`

Do not use sentence-style messages, method names, or generic `message` payloads
as event names.

## Review Checklist

Reject changes that:

- Emit plain text logs.
- Bypass `app.logging.log`.
- Use `message` instead of `event`.
- Log every helper entry.
- Add try/except only to log and re-raise.
- Repeat context IDs in every payload instead of using `assign_context`.
- Log secrets.
- Add user-managed environment variables only for logger base fields.

Approve changes that:

- Put public use cases inside `@log_context_scope`.
- Make the first service log searchable by key IDs.
- Let downstream logs inherit context.
- Use `started`, `completed`, and scope-managed `failed` consistently.
- Cover external I/O, persisted state changes, expected domain failures, and
  meaningful loop/job/attempt boundaries.
