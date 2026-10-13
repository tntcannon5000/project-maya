# Transfer bandwidth attribution

Start: eca6b16, isolated branch research/maya-transfer-attribution. Retained
candidate and build/maya_transfer_lab_pre_attribution.exe preserved. No publication.

Question: warm demand is 383 MB/token; actual mapped pull 37 GB/s versus isolated
52 GB/s. Packing and ready-to-consumer gaps do not explain the deficit.

Bounded investigation: one fixed-input workload, one repeated request. Retain the
last eight tokens' actual source addresses, CPU selections/input and GPU resident
addresses while model allocations remain alive. Replay into private destinations,
using real CUDA fetch kernels, per-layer shapes and grid limits. Compare copies
alone, recorded CPU expert work, routed GPU expert work, both and DMA. CUDA graphs
remove per-kernel host submission variation. GPU timestamps measure copy intervals;
readback checks every byte outside timing. No destinations belonging to the model
are modified. Replay must wait for service and GPU completion before reading.

Limitations: RAM slots may have new same-format contents at replay time. Replay
measures actual addresses, sizes and computations but does not reproduce earlier
numerical outputs. GPU replay excludes attention and shared expert computation;
synthetic input is used for routes without a published CPU input. No claim that
these timings are whole-engine throughput. Model-output parity is checked before
replay. Fixed order and profiling/allocation effects are recorded, not ignored.

Hypotheses: CPU DDR contention; GPU compute contention; scattered/large allocation
placement; small per-route launch/transaction overhead; PCIe payload efficiency;
unprofiler/profiler effects. Do not confuse a copy-only speedup with a deployable
DMA scheduler. Traffic reduction is a separate cache-policy opportunity.

Outcome: real-address copies alone 51.95 GB/s; CPU/GPU concurrent replay 43.21.
Live CPU-first ablation restores 50.78 GB/s from 37.32, removing 2.72 ms of fetch
time, but token wall grows 23.49 -> 27.90 ms. Reject serialization as an
optimization. Eight CPU workers are neutral at 45.10 TPS versus retained 45.20;
keep default 16. Exact logits for replay recording, serialization and worker cap.
No production performance settings changed. See
[full evidence](evidence/maya-transfer-attribution/RESULTS.md).

First replay failed to time captured internal events. Replaced with graph-local
GPU globaltimer stamps and reran the same fixture; completed 3,360 replay records
and byte-exact readbacks. Additional safety checks were then compiled (RAM slot
ownership, invalidated-capture cleanup, explicit HIP unsupported result). These
failure-path hardening checks have not been fault-injected. Raw failed evidence
is preserved. CPU-first trace and one unprofiled worker-cap test complete this
bounded investigation; no broad suite or cache-policy sweep was run.
