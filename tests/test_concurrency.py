"""Concurrency stress on the LRU cache (prompt §8 thread-safety claim).

Many threads pin/mark/enforce/release the same LayerCache. The invariants: no
exception, hit+miss accounting is exact (every touch counted once), all pins get
released, and capacity holds once the dust settles.
"""

from __future__ import annotations

import random
import threading

from streamllm.cache import LayerCache


class _FakeHandle:
    def __init__(self, index: int) -> None:
        self.index = index
        self.nbytes = 0
        self.evicted = False

    def evict(self) -> bool:
        self.evicted = True
        return True


def test_concurrent_touch_release_is_consistent():
    capacity = 4
    n_layers = 16
    n_threads = 8
    ops = 250
    cache = LayerCache(capacity)
    handles = {i: _FakeHandle(i) for i in range(n_layers)}
    errors: list[Exception] = []

    def worker(seed: int) -> None:
        rnd = random.Random(seed)
        try:
            for _ in range(ops):
                idx = rnd.randrange(n_layers)
                h = handles[idx]
                cache.touch(h)  # pins
                cache.mark_resident(h)
                cache.enforce_capacity(protect=idx)
                cache.release(h)  # unpins
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # Every touch was counted exactly once as a hit or a miss.
    assert cache.hits + cache.misses == n_threads * ops
    # All pins released.
    assert all(v == 0 for v in cache._in_use.values())
    # Once nothing is pinned, capacity can be enforced down to the limit.
    cache.enforce_capacity(protect=-1)
    assert cache.resident_count <= capacity
