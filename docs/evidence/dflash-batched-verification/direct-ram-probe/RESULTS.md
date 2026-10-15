# Direct mapped-RAM arithmetic: reject generic integration

One fixed-input coding fixture, 30 occupied prompt tokens plus 16 continuation
inputs repeated twice, allocated context131072. The isolated probe replays the
last eight address sets. RAM94GiB, requested GPU expert pool22GiB, retained
CPU plan011223344. Four complete replay modes, each with one warm-up pass and
two measured passes. Same source addresses, weights, activation and arithmetic.
This is a mechanism test, not end-to-end DFlash throughput or original-token parity.

| Expert-work replay per token-equivalent | Staged copy | Direct RAM |
|---|---:|---:|
| GPU timeline, no CPU contention (ms) |6.413|12.784|
| Wall completion, no CPU contention (ms) |6.951|13.299|
| GPU timeline, CPU co-execution (ms) |7.872|14.318|
| Wall completion, CPU co-execution (ms) |10.705|16.496|

CPU-contention measured passes are10.526/10.884ms staged and16.327/16.665ms
direct: the gap substantially exceeds their variation. GPU and CPU durations
overlap and must not be summed. Direct reads increase GPU time by82% and wall
completion by54% with CPU contention. Do not interpret these as whole-token TPS.

| Quant class | Direct/staged GPU-time ratio, CPU active | Wall ratio |
|---|---:|---:|
| Q2_K |1.20x|1.07x|
| Q3_K |5.02x|3.77x|
| Q4_K |2.37x|2.24x|

All three classes lose; the relatively smaller Q2_K loss does not justify a
selective implementation. The control uses full-copy-then-compute rather than
retained component overlap, so this experiment does not disadvantage direct
reads by giving the staged control a more advanced overlap schedule.

## Engagement, correctness and capacity

Direct plans point at the same owned pinned addresses used by the copy kernel.
There is no explicit weight copy in direct graphs; complete native GU,
activation quantization and down arithmetic still run.2688 measured layer
records all pass finite and byte-exact4096-element GPU output and quantized
hidden-buffer checks; unscored warm-up passes also check exactly. CPU experts
execute concurrently and their outputs are finite, but CPU contributions are
not merged into the checked GPU result. No serving arithmetic or cache policy
was changed. Legacy probe modes remain intact behind a separate opt-in branch.

Each replay uses225.56MB/token-equivalent logical fetched blobs. The staged
mode copies those bytes explicitly; direct mode consumes them through GPU
host reads. **Zero explicit copy bytes is not zero PCIe traffic.** Hardware
transaction counts were not measured; inefficient native host-read transactions
and latency are explanations consistent with the result, not proven attribution.
The shorter fixture has a different route/capacity history from the earlier
383/414MB samples; do not claim reduced traffic from that difference.

Cache scrub201326592bytes runs outside timing. It and replay destinations add
scratch beyond the live model, with no cache-slot resizing after load. Peak
sampled GPU usage30931MiB; minimum available physical RAM16.18GiB. Timed
intervals exclude graph creation, cache scrub, output readback and validation.
This limits hot-L2 replay bias but is not a measurement of complete serving
scheduling. Cold/warm original inference and diagnostic replay remain distinct.

Current bytes in recorded slots may differ from original route-time experts.
Their quant-class layouts remain valid and all modes consume identical current
bytes under exclusive ownership. There is no multi-token pair/group or live
transient-only qualification; these limitations do not warrant integrating a
mechanism whose complete singleton kernel path is substantially slower.

## Decision

Reject generic direct-native RAM arithmetic. Preserve the isolated tool and
negative result; do not add production flags or run another workload sweep.
The retained candidate, defaults and launchers remain unchanged. Main
verification-speed objective remains unmet; no60TPS claim. Evidence includes
binary/model hashes in screen/manifest.json, exact checks in the CSV and
per-pass/per-format numbers in screen/summary.json. Raw logits stay local.
