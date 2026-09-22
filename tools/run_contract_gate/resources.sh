# Sourced by the host runner; exact names and owner labels are mandatory.
run_bounded() {
    local limit=$1; shift
    "$@" & ACTIVE_PID=$!
    local deadline=$((SECONDS + limit))
    while kill -0 "$ACTIVE_PID" 2>/dev/null; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            kill "$ACTIVE_PID" 2>/dev/null || true
            # Docker may proxy TERM without exiting. Bound its grace period too.
            local grace=$((SECONDS + 2))
            while kill -0 "$ACTIVE_PID" 2>/dev/null && [ "$SECONDS" -lt "$grace" ]; do sleep 1; done
            kill -KILL "$ACTIVE_PID" 2>/dev/null || true
            wait "$ACTIVE_PID" 2>/dev/null || true
            ACTIVE_PID=
            return 124
        fi
        sleep 1
    done
    local status=0
    wait "$ACTIVE_PID" || status=$?
    ACTIVE_PID=
    return "$status"
}
cleanup_resources() {
    local failed=0 name owner exists
    for name in "$RUNNER" "$DB"; do
        exists=$("${D[@]}" container ls -aq --filter "name=^/${name}$") || { failed=1; continue; }
        [ -n "$exists" ] || continue
        owner=$("${D[@]}" inspect --format "{{index .Config.Labels \"$LABEL\"}}" "$name") || { failed=1; continue; }
        if [ "$owner" = "$RUN_ID" ]; then "${D[@]}" rm -f -v "$name" || failed=1; else failed=1; fi
    done
    exists=$("${D[@]}" volume ls -q --filter "name=^${SOCKET}$") || return 1
    if [ -n "$exists" ]; then
        owner=$("${D[@]}" volume inspect --format "{{index .Labels \"$LABEL\"}}" "$SOCKET") || return 1
        if [ "$owner" = "$RUN_ID" ]; then "${D[@]}" volume rm "$SOCKET" || failed=1; else failed=1; fi
    fi
    return "$failed"
}
