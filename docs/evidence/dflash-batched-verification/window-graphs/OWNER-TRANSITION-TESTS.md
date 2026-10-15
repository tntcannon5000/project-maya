# Production promotion-retirement transition qualification

2026-10-15. Extracted the existing opt-in superseded-upload retirement transition into FastState::retire_promotion_owner, called from the same fast_boundary point. No arithmetic, copy schedule, default or launcher change. Caller still queries the copy event first, keeps pending moves reserved, drains compute/service and holds the metadata lock before retirement.

The native unit test instantiates actual FastState pools/maps and invokes this production method, not a duplicate implementation.24cases pass:

- Superseded upload with original RAM backing still current.
- GPU winner still resident, or evicted with its former slot reused by another expert.
- Replacement RAM backing held, pinned, pending demotion, loading, newly read, free, absent, foreign-key or out of range.
- Pending/unavailable backing publishes zero; ready owned backing publishes its pointer before the old source is released. New GPU TAB/map and reused slot remain untouched.
- Invalid reserved source/destination or unmarked competing owner fail before ownership/counter/table mutation. Correcting metadata does not bypass a sticky fatal error.
- Ordinary unsuperseded upload returns Publish without mutations, leaving the existing publication path in control.
- A repeated retirement fails without further mutation.

Review corrected an initial test weakness: merely setting slot_of=-1 left an orphan resident; the final fixture actually reuses the winner's slot for another expert and checks its mapping. Failure and Publish cases compare keys, states, timestamps, maps, emitted tables and counters.

Actual-model integration after extraction: same6793-token prefix, T2/T4windows, partial commits, two resets, hybrid CPU011223344.5,575,680window logits and309,760prefix logits bitwise identical to pre-extraction artifacts.33superseded uploads retired,16sampled boundaries with zero duplicates/orphans/invalid sources. No thresholds relaxed.

Scope limitation: the24unit cases qualify the metadata transition. They do not simulate pending DMA or prove all races; integration exercises real GPU work but does not separately force each unit scenario with a pending CUDA transfer. Historical original numerical failure remains unattributed.

## Background demotion audit

Read-only source audit found no valid competing-promotion path for a background demotion: GPU TAB and slot_of remain live while its source slot is kDraining. Demand sees a resident hit; prefetch and lookahead skip that owner. Event completion alone does not change mappings. Retirement follows compute/service drain under the metadata lock. Ordinary spare-refill demotions differ and their retirement already preserves a newer GPU owner. No new demotion bug was established and no demotion code was changed.

Build/run: tools/build-promotion-owner-test.bat, then build/maya_promotion_owner_test.exe with CUDA runtime DLL directory on PATH. No model or GPU allocation required by the metadata test itself.
