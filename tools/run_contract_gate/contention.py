"""Test-only contention evidence and bounded worker recovery; no production hooks."""
import time


def join_workers(threads, connections, seconds, on_timeout):
    deadline = time.monotonic() + seconds
    for thread in threads:
        thread.join(max(0, deadline-time.monotonic()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        on_timeout(alive)
        for connection in connections:
            if not connection.closed:
                connection.cancel()
        deadline = time.monotonic() + 3
        for thread in threads:
            thread.join(max(0, deadline-time.monotonic()))
    return [thread.name for thread in threads if thread.is_alive()]


def verify_contention(result):
    """Lack of proof is a setup error (INCOMPLETE), never a passing scenario."""
    events = result['timeline']
    if result['workers_alive'] or any(e['event'] in ('preparation_error', 'worker_timeout') for e in events):
        raise TimeoutError('contention scheduling or worker completion was not verified')
    pids = result.get('pids', {})
    if set(pids) != {'A', 'B'} or pids['A'] == pids['B']:
        raise RuntimeError('two distinct request backends were not verified')
    blocks = [e for e in events if e['event'] == 'blocking_observed']
    if len(blocks) != 1 or not any(row['pid'] == pids['B'] and pids['A'] in row['blockers'] for row in blocks[0]['rows']):
        raise RuntimeError('actual request-to-request blocking was not verified')
    block = blocks[0]['sequence']
    for name in ('A', 'B'):
        if not any(e['request_id'] == name and e['event'] == 'scope_read' and not e['found'] and e['sequence'] < block for e in events):
            raise RuntimeError('both absent scope reads were not verified')
    if not any(e['request_id'] == 'A' and e['event'] == 'bind_complete' and e['sequence'] < block for e in events):
        raise RuntimeError('blocking after real binding writes was not verified')
    if not any(e['event'] == 'release_A' and e['sequence'] > block for e in events):
        raise RuntimeError('release after observed blocking was not verified')
