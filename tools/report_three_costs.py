"""Build the campaign's compact report from saved, separate measurement runs."""
import csv,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'docs/evidence/maya-three-costs'
def read(n,f='summary.json'):return json.loads((P/n/f).read_text())
def speed(n):
    s=read(n);return f"{s['0']['TPS']:.2f} / {s['1']['TPS']:.2f}"
def prefill(n):
    return [float(x) for x in re.findall(r'prefill repeat=\d+ tokens=8192 ms=([\d.]+)',(P/n/'engine.log').read_text())]
experiments=[
 ('Dense Q4 activation reuse','q4-rows2','Reject: dense Q4 time ~3.21→3.18ms; no useful whole-loop gain'),
 ('Earlier CPU answer','early-cpu','Retain in combination; answer precedes cache bookkeeping; stable request snapshot'),
 ('Windows spin PAUSE','pause','Retain in combination; same fences and time budgets'),
 ('Resident GU during prefix fetch','resident-gu','Reject: measured overlap rose, arithmetic and CPU tail grew; diagnostic TPS'),
 ('Down row streaming + resident GU','bands-combination','Dependency experiment: resident split unnecessary'),
 ('Down row streaming without resident split','bands-no-resident','Retain: exact row ordering, each byte copied once'),
 ('CPU expert-ready tasks','cpu-pipeline','Alone not helpful; retain only with phase schedule; diagnostic TPS'),
 ('CPU prefix-phase gate','phase-limit','Retain in combination; verified gated callbacks, not core affinity'),
 ('Warp top-k/cooperative publication','router','Small route-stage saving; retain in combination'),
 ('MLA MoE suffix capture','suffix-bands','Retain graph infrastructure; isolated speed increment within variation'),
 ('HC normalization stage split','hc-split','Reject: exact but extra nodes/reads outweigh removed barriers'),
 ('Full scheduler/GPU combination','scheduler-combination','46.49 was a screen, not the confirmation result'),
 ('Single-expert CPU pipeline coverage','scheduler-single','Reject extension: engaged but no extra gain'),
 ('Same GPU combination without CPU schedule','simple-router-combination','Ablation: CPU schedule helps the combined path'),
]
lines=['# Maya three-costs campaign — results','',
 '2026-10-13. Branch `research/maya-three-costs`; measured retained baseline preserved at `4f2f6e8`. No external publication.', '',
 '## Confirmed performance','',
 'Identical GSQ-RCO 3.0bit Q4-attention packed weights, c=1, fixed input histories, fixed CPU plan, 22GiB expert pool. No MTP, speculative tokens, K4V, skipping or new quantization. TPS is scored tokens / summed forward time. First means first request after tier warming, not disk-cold startup.', '',
 '| Workload | Baseline first / repeat TPS | Candidate first / repeat TPS | Repeat change |',
 '|---|---:|---:|---:|']
for label,b,c in [('Short coding continuation','baseline-confirm','candidate-confirm'),('8192 occupied code tokens','baseline-8k','candidate-8k')]:
    B,C=read(b),read(c)
    lines.append(f"| {label} | {speed(b)} | {speed(c)} | {(C['1']['TPS']/B['1']['TPS']-1)*100:+.2f}% |")
lines+=['','Each continuation has 128 scored tokens, two requests. The fresh short baseline 44.36 differs from the initial 43.83 screen; individual sub-percent improvements are not established. The gain is modest and 60 TPS was not reached.','',
 '## Hypotheses, supporting combinations and decisions','',
 '| Mechanism | First / repeat TPS | Result |','|---|---:|---|']
for label,n,decision in experiments:lines.append(f'| {label} | {speed(n)} | {decision} |')
lines+=['',
 'Dependency findings: down bands do not need resident-GU splitting. The CPU task pipeline needs phase-aware scheduling to help this combination; broader single-expert coverage did not improve it. Suffix capture enables component overlap without uncaptured event submission. Routing publication advances CPU work but does not remove payload bytes. Combined benefits are not additive. Detailed CPU-stage NVTX spans in early diagnostic runs add overhead; final diagnostics disable those optional spans.', '',
 '## Prefill and numerical qualification','',
 '| 8192-token prefill | Baseline ms | Candidate ms |','|---|---:|---:|']
for i,(b,c) in enumerate(zip(prefill('baseline-8k'),prefill('candidate-8k'))):lines.append(f'| Request {i+1} | {b:.3f} | {c:.3f} |')
for n in ['candidate-confirm','candidate-8k','retained-trace']:
    s=read(n)['parity'];lines+=['',f"`{n}`: {s['count']:,} logits; exact={s['exact']}; max absolute difference={s['max_abs']}; greedy agreement={s['greedy_agreement']:.0%}."]
lines+=['','Mixed Q2_K/Q3_K/Q4_K experts are exercised by the model. Full logits are compared, not just sampled tokens. Tier counters and bytes agree for matched inputs. Route masks are separately checked from diagnostic traces; no claim that aggregate hit counts alone prove routing equivalence.','',
 '## Capacity and traffic','',
 '| Repeated workload | Baseline H2D MB/token | Candidate H2D MB/token | Baseline/candidate D2H MB/token |','|---|---:|---:|---:|']
for label,b,c in [('Short','baseline-confirm','candidate-confirm'),('8192 occupied','baseline-8k','candidate-8k')]:
    B,C=read(b)['1'],read(c)['1'];lines.append(f"| {label} | {B['fetch_bytes']/1e6:.3f} | {C['fetch_bytes']/1e6:.3f} | {B['demotion_bytes']/1e6:.3f} / {C['demotion_bytes']/1e6:.3f} |")
lines+=['','Both retain 2693 GPU slots (21.999 GiB) and 11819 RAM slots (97.822 GiB). Long prefill lends/reclaims the cache and produces more subsequent demotions; short repeated demotion elimination must not be generalized to that lifecycle. Short CPU expert logical payload remains 564.156 MB/token; payload accounting is not measured physical DRAM traffic.','',
 '| Run | Minimum system RAM available GiB | Peak GPU allocation MiB (whole device) |','|---|---:|---:|']
for n in ['baseline-confirm','candidate-confirm','baseline-8k','candidate-8k']:
    mem=read(n,'memory.json');gpu=list(csv.DictReader((P/n/'gpu.csv').open()))
    key=next(k for k in gpu[0] if 'memory.used' in k)
    lines.append(f"| {n} | {min(x['available_physical'] for x in mem)/2**30:.2f} | {max(float(x[key].split()[0]) for x in gpu):.0f} |")
lines+=['','GPU memory includes foreground/driver allocations; profiler runs are separate and can use additional memory. Hardware logs preserve clocks, power and temperature; this is a local short qualification, not a population confidence interval.','',
 '## Diagnostic token timeline','',
 'The bars are disjoint interval unions and sum to elapsed token time. Fetch and computation overlap is counted once. These diagnostic loops exclude output-logit readback/harness overhead; use the unprofiled table for TPS.','']
base=json.loads((ROOT/'docs/evidence/maya-decode-latest/latest/partition-means.json').read_text())['1']
cand=read('retained-trace','partition-means.json')['1']
keys=[k for k in base if k!='wall_ms']
lines+=['| Warm diagnostic stage | Previous retained ms/token | New candidate ms/token |','|---|---:|---:|']
for k in keys:lines.append(f'| {k} | {base[k]:.3f} | {cand[k]:.3f} |')
lines.append(f"| **Total** | **{base['wall_ms']:.3f}** | **{cand['wall_ms']:.3f}** |")
fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
colors=['#237e9e','#76b7b2','#d68736','#ba4d61','#9178a7','#d5d9dc']
left=[0,0]
for k,color in zip(keys,colors):
    v=[base[k],cand[k]];ax.barh([1,0],v,left=left,color=color,label=k)
    for y,w,x in zip([1,0],v,left):
        if w>.65:ax.text(x+w/2,y,f'{w:.2f}',ha='center',va='center',fontsize=9)
    left=[x+w for x,w in zip(left,v)]
ax.set_yticks([1,0],['Previous retained','New candidate']);ax.set_xlabel('ms per token — warm diagnostic, 128 tokens per bar')
ax.set_xlim(0,max(left)*1.06);ax.set_title('Where decode time went — no speculative decoding')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=8)
fig.savefig(P/'comparison.png',dpi=160);plt.close(fig)
lines+=['','![Token partition](comparison.png)','',
 'Individual median/fast/slow token timelines and layer-local component timings are in `retained-trace/`. The baseline is the previously retained diagnostic with identical slots/payloads, not an unprofiled run.','',
 'The profiled total is effectively unchanged, despite the unprofiled improvement. More overlap is proven, but increased arithmetic duration, fetch duration and CPU waits consume it in this trace. Added graph nodes and profiler sensitivity prevent assigning the full unprofiled saving to a particular stage. Do not present the diagram as proof of a net profiled speedup.','',
 '## Lifecycle and held-out generation','',
 'The current binary was tested with retained-baseline flags and with candidate flags on the same warmup → STOP after eight tokens → recovery → web-generation sequence. RAM shadow ownership verification was enabled for both. Cancellation reported cancel, recovery completed, QUIT exited cleanly, and every output token ID matched. This is a same-binary flag ablation, separate from the preserved-binary fixed-input comparisons above.','',
 '| Held-out free generation | Control TPS | Candidate TPS | Interpretation |','|---|---:|---:|---|',
 '| Web page, 128 tokens | 38.78 | 39.01 | +0.59%; too small to establish a reliable gain from one run |','',
 'All 10,752 diagnostic layer-route mask records match exactly (fetch, CPU, disk-miss, promotion masks and blob bytes in token/layer order). Those marks do not contain expert IDs; full-logit equality supplies the complementary numerical check. See `route-validation.json` and `lifecycle-validation.json`.','',
 'Cold-tier qualification disabled tier warming and exercised 761 actual disk reads over the scored 16-token continuation. All 2,478,080 logits and all non-timing per-token counters matched the preserved baseline. That cold configuration selected zero CPU experts, so it qualifies disk ownership/stream readiness rather than CPU scheduling. Warm hybrid qualification is provided by the short and 8K comparisons. See `cold-validation.json`.','',
 'Mechanism proof: the retained trace has 42 optimized router launches, 42 prefix copies, 84 down-band copies, 84 down consumers and 42 prefix-ready publications per token. Overlap is measured by interval intersection. Unprofiled counters recorded 51,621 gated callbacks in the confirmation run. The rejected HC trace has 68 launches of each split phase per token and exact prefix logits; its phase2 alone takes about 0.97 ms, explaining why removing spin barriers did not help.','',
 '## Reproduction and rollback','',
 '`tools/run_three_costs.py NAME --candidate` selects the retained configuration. Add `--occupied 8192` for the corpus fixture, or `--trace` for diagnostics. `--baseline` uses the preserved pre-campaign executable. All mechanisms remain opt-in; rejected prototypes remain checkpointed for audit, not enabled in the selected configuration.', '',
 'Research bounds, source reviews, implementation rationale and checkpoints are recorded in `docs/MAYA-THREE-COSTS-CAMPAIGN.md` and the three linked research reports. No residency or capacity improvement is credited to this campaign. Remaining costs are host-memory expert traffic, its contention with CPU expert work, exposed CPU completion, and GPU arithmetic preceding the route. Reaching 60 TPS needs substantially more than this measured scheduling gain.']
(P/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
