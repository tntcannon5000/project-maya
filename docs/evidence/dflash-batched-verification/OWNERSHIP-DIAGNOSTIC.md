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

## Used-expert immutable pack audit

`MAYA_WIN_WEIGHT_VERIFY_P0=60` selects only the first `window()` starting at
position 60 in each device's model lifetime. Reset/repeat does not select it
again. In the fixed-input fixture this is the sixth window, whose logit rows
31..38 include the first combined-HC mismatch at row 34.

After routed fetch completion and before expert consumers, every distinct
`(expert ID, GPU pointer)` pair in each eligible layer is compared byte for byte
against a private reference read directly from the immutable pack shards.
The check includes resident/main/tail, promoted, scratch and borrowed GPU
destinations. It verifies `Ly.blob` active bytes, excluding slot-stride padding.
Shard ranges, expert IDs, destination ranges and RAM/staging source ranges are
validated before comparison. A mismatch aborts with device, position, layer,
token, rank, expert, pointer, provenance and first differing byte offset.

The initial diagnostic supports complete full-layer native GGUF groups with
CPU placement disabled and sufficient row scratch. EXL3, split/staggered
groups, omitted experts and token-path fallbacks explicitly report skipped
coverage. The summary counts verified layer scopes, expert/pointer pairs,
bytes and skipped scopes. An all-zero skipped count is required for complete
coverage of routed expert weights in the selected window.

This performs full model-weight disk reads and GPU readbacks for one window;
it can substantially change timing and cache retirement. Passing it covers
the bytes observed before those consumers, not all later windows, shared/dense
weights, activation arithmetic or unperturbed execution.
