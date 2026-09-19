#!/usr/bin/env bash
# Local-only Gate. Download preparation precedes network-disabled test containers.
set -euo pipefail
exec 3>&1 4>&2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FIXED=0ec2cef0290f4659ad21ccc1dd2a20df2801ff50
REMOTE=https://github.com/krafton-jungle-project-4team/loop-ad_data-source_contract.git
PG_IMAGE=pgvector/pgvector:0.8.0-pg16@sha256:a132765ec351c65111b5b675928a3a0515a466a40f97277329db8b8209ad8bc9
if [ "$#" -gt 1 ]; then echo 'Usage: run-contract-gate.sh [NEW_OUTPUT_DIRECTORY]' >&2; exit 2; fi
if [ "$#" -eq 1 ]; then mkdir "$1"; OUTPUT=$(cd "$1" && pwd); else OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/rcg-output.XXXXXX"); fi
WORK=$(mktemp -d "${TMPDIR:-/tmp}/rcg-work.XXXXXX")
RUN_ID="rcg-$(basename "$WORK" | tr '[:upper:]' '[:lower:]')"
DB="$RUN_ID-db"; RUNNER="$RUN_ID-runner"; SOCKET="$RUN_ID-socket"
LABEL=org.loopad.run-contract-gate
ACTIVE_PID=; START=$SECONDS; IMAGE=; READY=0
mkdir -p "$WORK/docker-config" "$WORK/source" "$WORK/gate/tests/run_contract_gate" "$WORK/gate/tests/fixtures/run_contract_gate/baseline" "$WORK/gate/tools/run_contract_gate" "$WORK/context/tools/run_contract_gate" "$OUTPUT/fixed" "$OUTPUT/latest" "$OUTPUT/controls"
source "$ROOT/tools/run_contract_gate/resources.sh"
# A valid conservative result survives even Docker/preparation failure.
printf '%s\n' '{"schema_version":1,"status":"INCOMPLETE","exit_code":2,"reason":"runner preparation did not complete"}' > "$OUTPUT/result.json"
D=(docker --config "$WORK/docker-config")
finish() {
    local incoming=$? status=2
    exec 1>&3 2>&4
    trap - EXIT INT TERM
    if [ "$incoming" -ne 0 ]; then printf '%s\n' "$incoming" > "$OUTPUT/host.error"; fi
    if [ -n "$ACTIVE_PID" ]; then kill -KILL "$ACTIVE_PID" 2>/dev/null || true; wait "$ACTIVE_PID" 2>/dev/null || true; ACTIVE_PID=; fi
    if run_bounded 45 cleanup_resources > "$OUTPUT/cleanup.log" 2>&1; then touch "$OUTPUT/cleanup.ok"; fi
    if [ "$READY" -eq 1 ]; then
        status=0
        run_bounded 45 "${D[@]}" run --rm --name "$RUNNER" --label "$LABEL=$RUN_ID" --network none \
            --mount "type=bind,src=$WORK/gate,dst=/gate,readonly" --mount "type=bind,src=$OUTPUT,dst=/output" \
            -e PYTHONPATH=/gate -e RCG_RUN_ID="$RUN_ID" -e RCG_DURATION="$((SECONDS-START))" \
            "$IMAGE" python -m tools.run_contract_gate.finalize > "$OUTPUT/summary.txt" 2>&1 || status=$?
        if [ ! -f "$OUTPUT/finalizer.exit" ] || [ "$(cat "$OUTPUT/finalizer.exit")" != "$status" ]; then
            status=2
            printf '%s\n' '{"schema_version":1,"status":"INCOMPLETE","exit_code":2,"reason":"finalizer failed or its exit disagreed with result"}' > "$OUTPUT/result.json"
        fi
        cleanup_resources >> "$OUTPUT/cleanup.log" 2>&1 || { status=2; printf '%s\n' '{"schema_version":1,"status":"INCOMPLETE","exit_code":2,"reason":"finalizer cleanup failed"}' > "$OUTPUT/result.json"; }
        cat "$OUTPUT/summary.txt"
    fi
    if [ "$status" -gt 2 ]; then status=2; fi
    rm -rf "$WORK"
    echo "Artifacts: $OUTPUT"
    exit "$status"
}
trap finish EXIT
trap 'touch "$OUTPUT/interrupted"; exit 130' INT
trap 'touch "$OUTPUT/interrupted"; exit 143' TERM
ENDPOINT=${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}
case "$ENDPOINT" in unix://*) ;; *) echo 'Only local Unix Docker is supported.' >&2; exit 2;; esac
D=(docker --config "$WORK/docker-config" --host "$ENDPOINT")
# Copy only source code/data and exact Gate artifacts. Reject symlinks and odd names.
while IFS= read -r file; do
    case "$file" in app/*.py|app/*.json|pyproject.toml) ;; *) continue;; esac
    case "$file" in *$'\n'*|*,*) echo 'Unsupported source filename' >&2; exit 2;; esac
    [ ! -L "$ROOT/$file" ] || { echo 'Source symlink refused' >&2; exit 2; }
    [ -f "$ROOT/$file" ] || continue
    mkdir -p "$WORK/source/$(dirname "$file")"; cp "$ROOT/$file" "$WORK/source/$file"
done < <(git -C "$ROOT" ls-files --cached --others --exclude-standard app pyproject.toml)
for file in __init__.py seed.py baseline.py test_run_db.py test_runner_control.py; do cp "$ROOT/tests/run_contract_gate/$file" "$WORK/gate/tests/run_contract_gate/$file"; done
for file in rows.json expected.json provenance.json; do cp "$ROOT/tests/fixtures/run_contract_gate/baseline/$file" "$WORK/gate/tests/fixtures/run_contract_gate/baseline/$file"; done
for file in __init__.py report.py lane.py finalize.py inputs.py resources.sh manifest.json control-manifest.json; do cp "$ROOT/tools/run_contract_gate/$file" "$WORK/gate/tools/run_contract_gate/$file"; done
cp "$ROOT/tools/run_contract_gate/requirements.lock" "$WORK/context/tools/run_contract_gate/requirements.lock"
cp "$ROOT/Dockerfile.run-contract-gate" "$WORK/context/Dockerfile.run-contract-gate"
cp "$ROOT/scripts/run-contract-gate.sh" "$WORK/gate/run-contract-gate.sh"
LOCK_HASH=$(shasum -a 256 "$WORK/context/tools/run_contract_gate/requirements.lock" | awk '{print $1}')
IMAGE="rcg-gate:$LOCK_HASH"
run_bounded 600 "${D[@]}" build -t "$IMAGE" -f "$WORK/context/Dockerfile.run-contract-gate" "$WORK/context" > "$OUTPUT/build.log" 2>&1
run_bounded 180 "${D[@]}" pull "$PG_IMAGE" > "$OUTPUT/postgres-image.log" 2>&1
HEAD=$(git -C "$ROOT" rev-parse HEAD)
DIRTY=false; [ -z "$(git -C "$ROOT" status --porcelain)" ] || DIRTY=true
run_bounded 45 "${D[@]}" run --rm --name "$RUNNER" --label "$LABEL=$RUN_ID" --network none \
    --mount "type=bind,src=$WORK/source,dst=/source,readonly" --mount "type=bind,src=$WORK/gate,dst=/gate,readonly" \
    --mount "type=bind,src=$OUTPUT,dst=/output" -e PYTHONPATH=/source:/gate:/gate/tests \
    -e RCG_HEAD="$HEAD" -e RCG_DIRTY="$DIRTY" -e RCG_PG_IMAGE="$PG_IMAGE" \
    -e RCG_IMAGE="$("${D[@]}" image inspect "$IMAGE" --format '{{.Id}}')" \
    -e RCG_ARCH="$("${D[@]}" image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}')" \
    "$IMAGE" python -m tools.run_contract_gate.inputs > "$OUTPUT/inputs.log" 2>&1
READY=1
# Ignore all user Git credentials/config for this public, read-only download.
G=(env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c core.askPass=)
"${G[@]}" init --bare "$WORK/contract.git" > "$OUTPUT/contract.log" 2>&1
acquire() {
    local sha=$1 lane=$2
    run_bounded 180 "${G[@]}" -C "$WORK/contract.git" fetch --depth=1 "$REMOTE" "$sha" >> "$OUTPUT/contract.log" 2>&1 || return 1
    "${G[@]}" -C "$WORK/contract.git" show "$sha:postgres/schema.sql" > "$WORK/$lane.sql" || return 1
    printf '%s\n' "$sha" > "$OUTPUT/$lane.sha"
    shasum -a 256 "$WORK/$lane.sql" | awk '{print $1}' > "$OUTPUT/$lane.ddl.sha256"
}
acquire "$FIXED" fixed || echo 'fixed Contract fetch failed' > "$OUTPUT/fixed.error"
if run_bounded 60 "${G[@]}" ls-remote "$REMOTE" refs/heads/main > "$WORK/latest-ref" 2>> "$OUTPUT/contract.log"; then
    LATEST=$(awk 'NR==1 {print $1}' "$WORK/latest-ref")
    if [[ "$LATEST" =~ ^[0-9a-f]{40}$ ]]; then acquire "$LATEST" latest || echo 'latest Contract fetch failed' > "$OUTPUT/latest.error"; else echo 'latest SHA unresolved' > "$OUTPUT/latest.error"; fi
else echo 'latest Contract resolution failed' > "$OUTPUT/latest.error"; fi
"${D[@]}" volume create --label "$LABEL=$RUN_ID" "$SOCKET" > "$OUTPUT/resources.txt"
"${D[@]}" create --name "$DB" --label "$LABEL=$RUN_ID" --network none \
    --mount "type=volume,src=$SOCKET,dst=/var/run/postgresql" -e POSTGRES_HOST_AUTH_METHOD=trust \
    -e PGDATA=/tmp/rcg-data "$PG_IMAGE" -c listen_addresses= >> "$OUTPUT/resources.txt"
"${D[@]}" start "$DB" > /dev/null
ready=0
for attempt in $(seq 1 30); do
    if "${D[@]}" exec "$DB" sh -c 'test "$(head -1 /proc/1/comm)" = postgres && pg_isready -U postgres' > /dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
[ "$ready" -eq 1 ] || { "${D[@]}" logs "$DB" > "$OUTPUT/database.log" 2>&1; exit 2; }
for lane in fixed controls latest; do
    schema="$WORK/$lane.sql"; controls=0
    if [ "$lane" = controls ]; then schema="$WORK/fixed.sql"; controls=1; fi
    [ -s "$schema" ] || continue
    sha=$FIXED; [ "$lane" != latest ] || sha=$LATEST
    code=0
    run_bounded 180 "${D[@]}" run --rm --name "$RUNNER" --label "$LABEL=$RUN_ID" --network none \
        --mount "type=volume,src=$SOCKET,dst=/var/run/postgresql" \
        --mount "type=bind,src=$WORK/source,dst=/source,readonly" --mount "type=bind,src=$WORK/gate,dst=/gate,readonly" \
        --mount "type=bind,src=$schema,dst=/schema.sql,readonly" --mount "type=bind,src=$OUTPUT/$lane,dst=/output" \
        -e PYTHONPATH=/source:/gate:/gate/tests -e RCG_SCHEMA=/schema.sql -e RCG_OUTPUT=/output \
        -e RCG_CONTRACT_SHA="$sha" -e RCG_CONTROLS="$controls" \
        "$IMAGE" python -m tools.run_contract_gate.lane > "$OUTPUT/$lane/pytest.log" 2>&1 || code=$?
    if [ "$code" -le 2 ]; then touch "$OUTPUT/$lane.completed"; else echo "runner stopped with exit $code" > "$OUTPUT/$lane.error"; fi
    # A timed-out Docker client can leave its server-side container running.
    if [ "$code" -eq 124 ]; then break; fi
done
"${D[@]}" logs "$DB" > "$OUTPUT/database.log" 2>&1
