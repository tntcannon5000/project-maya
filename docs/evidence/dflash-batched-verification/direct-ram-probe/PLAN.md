# Direct mapped-RAM expert arithmetic: bounded feasibility gate

Compare copied-to-VRAM native gate/up + activation quantization + down against
those same kernels reading their expert blobs from mapped pinned RAM. Keep
resident pointers and arithmetic unchanged. Four modes: staged/direct, each
with CPU expert co-execution off/on. Run one fixed-input coding fixture and
replay its last eight token address sets; no additional workload suite.

The captured slots may have been reused since original routing; this compares
current bytes at stable model-owned addresses of the recorded quant class,
not the original token logits or live cache policy. Native singleton kernels
are the first gate. Saved verification traces show most transient groups are
singletons, but this test does not qualify multi-token grouped scheduling.
All fetched sources can be remapped in this isolated test, including sources
that ordinary inference would promote. A serving implementation would initially
be transient-only; there is no promotion or model-cache mutation in replay.

Require finite, byte-exact GPU contribution outputs versus staged control.
CPU output is computed as contention, not merged into the checked GPU output.
Separate output validation and cache scrubbing from measured graph execution.
Keep nominal blob bytes distinct from measured PCIe bytes: this test does not
measure physical PCIe transactions. Direct access may amplify them.

Decision: only consider integration if complete expert-work timings show
material headroom (5% or larger), including CPU co-execution. Inspect quant
classes before rejecting a potential selective path. A large consistent loss
ends the generic direct-native experiment; no block-size sweep merely to
rescue a poor mechanism. A win requires follow-up grouped-window ownership,
full logits, and one matched end-to-end run before any TPS claim.
