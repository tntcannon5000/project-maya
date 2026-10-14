# Bounded verification ownership diagnostic

Set `MAYA_WIN_OWNERSHIP_VERIFY=3` to inspect the first three `window()` boundaries
on each device. Values are clamped to 0..16; unset/zero disables the diagnostic.
Prefill and ordinary decode boundaries do not consume this budget.

Before each shadow-backed GPU eviction in those boundaries, the diagnostic
copies the entire resident expert blob to the host, waits for that copy, and
compares every byte against the retained RAM backing. It checks the backing
before dropping the GPU owner. A difference reports the device, boundary,
layer, expert, RAM class, GPU/RAM slots and first differing byte offset, then
returns an error rather than continuing the request.

After boundary table updates complete, it reads the published GPU and RAM
pointer tables and checks each live pointer against its slot, owner key and
host mapping. A draining GPU owner is allowed when its matching demotion is
still tracked; a pinned RAM source remains valid during background promotion.
Unreferenced duplicate owners are not rejected by this pointer audit.

Each inspected boundary reports the number of full shadow blobs/bytes compared
and live GPU/RAM pointers checked. This is a synchronous diagnostic and can
change copy progress, cache retirement and timings. Do not use its throughput
as a performance result. Passing checks cover published pointer identity and
the shadow blobs actually evicted in the inspected boundaries; they do not
establish correctness of every expert blob or all later boundaries.

For serial/async source-topology comparisons, set `MAYA_WIN_FETCH_STAGE=1` in
both cases and confirm matching actual RAM/GPU cache sizes in the logs.
`MAYA_WIN_ASYNC_FETCH=1` also implies that staging allocation. Pressure serial
self-comparison previously differed from row 14, so an async-only cause has
not been established by the earlier pressure pair.
