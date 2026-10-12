"""Build the local campaign report from saved measurements; runs no inference."""
import csv,json,re,statistics,html
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'docs/evidence/maya-transfer'
OUT=ROOT/'report';OUT.mkdir(exist_ok=True)
def fixed(name):
    return list(csv.DictReader((ROOT/name/'fixed.csv').open()))
def aggregate(name,repeat):
    rows=[r for r in fixed(name) if int(r['repeat'])==repeat]
    times=[float(r['ms']) for r in rows]
    out={k:statistics.mean(float(r[k]) for r in rows) for k in rows[0]}
    out.update(TPS=1000/statistics.mean(times),p95=float(np.percentile(times,95)))
    return out
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pair(name):return ' / '.join(f'{aggregate(name,r)["TPS"]:.2f}' for r in [0,1])
def prefill(name):
    return [float(x) for x in re.findall(r'prefill repeat=\d+ tokens=8192 ms=([\d.]+)',(ROOT/name/'engine.log').read_text())]

sections=['# Maya transfer campaign: results',
'''Eight distinct mechanisms were experimentally evaluated. The retained combination is bounded immutable RAM backing plus demotion submission after the embedding upload. This removes copy-back traffic instead of trying only to accelerate it. **60 TPS was not reached.** No changed arithmetic kernels, token speculation, MTP, K4V, expert skipping or new quantization are in the retained implementation.

The same GSQ-RCO 3.0bit Q4-attention pack was used throughout. The routed weights contain Q2_K, Q3_K and Q4_K (GGML types 10/11/12). c=1; GPU expert pool cap 22 GiB; 2,693 GPU slots; CPU plan 011223344 unless an ablation explicitly changes it. CPU/GPU/hardware clocks were not locked; these are bounded paired screens, not confidence intervals from independent repeated sessions.

## Controlled fixed-input decode

128 saved input tokens per request, two repetitions in one process. First means first scored request after tier warmup, not disk-cold. Repeat benefits from cache adaptation. Short prompt: 30 occupied tokens; long prompt: 8,192 occupied tokens from frozen source code, followed by the same continuation. Context capacity is 131,072, distinct from occupancy. Host logits download is included, output-file writing excluded. No diagnostic CUDA profiler in these timings.''']
rows=[]
for label,b,c in [('Short, first','fixed-baseline','retained-clean'),('Short, repeat','fixed-baseline','retained-clean'),('8K, first','baseline8k-corpus','retained8k-corpus'),('8K, repeat','baseline8k-corpus','retained8k-corpus')]:
    r=int('repeat' in label);a=aggregate(b,r);z=aggregate(c,r)
    rows.append([label,f'{a["TPS"]:.2f}',f'{z["TPS"]:.2f}',f'{z["TPS"]/a["TPS"]:.3f}x',f'{a["ms"]:.2f} → {z["ms"]:.2f}',f'{a["p95"]:.2f} → {z["p95"]:.2f}'])
sections.append(table(['Fixture','Stock TPS','Candidate TPS','Ratio','Mean ms/token','p95 ms/token'],rows))
sections.append('The 8K and diagnostic traces use the pre-cleanup candidate; the short and free-generation qualification uses the cleaned build. Binary hashes are saved per run. Cleanup removed disabled DMA/resident-gate prototypes and restored stock kernels. All 39,649,280 hybrid logits before/after cleanup are bit-identical (cleanup-parity.json). Original pre-cleanup short result: '+pair('retained-short')+' TPS. The cleaned repeat is ~4% slower despite identical outputs/traffic, demonstrating run variation; do not interpret the small short-context H3-on-H1 increment as statistically established. H3 has clearer standalone and 8K evidence, so it remains in the candidate.')
sections.append('## Free generation on the cleaned build\n\nGreedy CSV/Python, explanation and HTML/JavaScript prompts, 128 output tokens each, two requests per task after warmup. Different outputs/routes are allowed here; fixed-input tests above control that confounder. These are short throughput workloads, not a webapp-completion quality benchmark.')
fr=[]
for task in ['csv','explanation','web']:
    vals=[]
    for name in ['free-stock','free-retained']:
        rr=[r for r in json.loads((ROOT/name/'requests.json').read_text()) if r['task']==task]
        vals.append(sum(r['tokens'] for r in rr)*1000/sum(r['decode_ms'] for r in rr))
    fr.append([task,f'{vals[0]:.2f}',f'{vals[1]:.2f}',f'{vals[1]/vals[0]:.3f}x'])
sections.append(table(['Task, pooled over two requests','Stock TPS','Candidate TPS','Ratio'],fr))
sections.append('## Hypotheses, engagement and decisions')
specs=[
('H1: RAM backing','shadow-complete','Actual copies eliminated; mixed-format bytes and ownership checked','Retain expanded backing. +10.49 GiB pinned RAM; limited 4 GiB backing gained only ~4%.'),
('H2: reuse admission','admission4','Promotion/D2H counters decrease','Do not add to retained combination; full-cpu-admission loses first-request speed.'),
('H3: deferred demotions','deferred-layers','Actual D2H/arithmetic overlap 2.33 ms/token; exact logits','Retain layers scheduling. Table-publication-only attempt was neutral.'),
('H4: resident gate/up overlap','overlap','1.04 ms/token actual fetch/arithmetic overlap; exact logits','Reject this implementation: extra arithmetic launches and GPU gaps outweigh overlap.'),
('H5: DMA transport','dma-overlap','Byte-exact microbenchmark 57–58 GB/s vs pull 45–53; integrated DMA bytes counted','Reject integrated DMA and overlap: 29.93/32.26 TPS. DMA-only first request 30.41 TPS.'),
('H6: next-layer prediction','full-predict','Useful predicted bytes measured; repeated useful fraction ~63.5%','Reject tested policy: excessive total H2D even with RAM backing.'),
('H7: lossless compression',None,'All sampled matrices roundtrip exactly; LZ4 ratio .9712, Zstd .9332','Reject CPU transport (3.52/1.83 GB/s decompression); GPU codec not built, small optimistic byte benefit.'),
('H8: CPU assignment','cpu-more','Lower transfer demand, CPU execution counters engaged','Do not add to expanded backing; full-cpu 36.70/41.05 TPS is not better.')]
sections.append(table(['Hypothesis','First/repeat TPS','Verified mechanism','Decision'],[(h,pair(n) if n else 'Feasibility only',v,d) for h,n,v,d in specs]))
sections.append('''Each experiment's manifest, counters and notes are preserved in the parent directory. Bounds, dependencies, risks and falsifiers were written before implementation in [campaign plan](../../../MAYA-TRANSFER-CAMPAIGN.md); public/local sources and intermediate decisions are in [research](../RESEARCH.md). Rejected screens were not all taken through long-context/prefill and Nsight qualification: those expensive metrics were reserved for promising mechanisms/complete combinations, not inferred from their flags. H7 is a measured feasibility rejection, not a claim all compression is impossible.

## Dependencies and ablations

H1 → eliminates most D2H; H3 → overlaps remaining D2H. H2 and H8 → reduce transfer demand, but compete with gains already supplied by H1. H4 → requires independent transfer progress; H5 supplied DMA, but the complete combination still lost. H6 → relies on accurate/timely predictions and cheap transfers; H1 removes copy-back overhead but cannot remove wasted prediction uploads. H7 → needs sufficient byte savings plus fast decompression before integration is justified.''')
sections.append(table(['Combination/ablation','First / repeat TPS'],[(n,pair(n)) for n in ['combo-policy','combo-no-shadow','shadow-complete','full-cpu','full-cpu-admission','dma-overlap','full-transfer','full-predict','retained-short']]))
sections.append('## Transfer traffic and capacity\n\nDecimal MB/token; no speculative prediction in retained candidate. Demand counters measure complete fetched blobs. Trace-control is instrumented stock scheduling; its workload/slot count matches the first short fixture. Stock v2 has no byte counters, so its zero placeholders are not reported as zero traffic.')
traffic=[]
for n,r in [('trace-control',0),('deferred-layers',1),('retained-clean',0),('retained-clean',1),('retained8k-corpus',0),('retained8k-corpus',1),('full-predict',1)]:
    a=aggregate(n,r)
    traffic.append([n+' / '+str(r),*[f'{a[k]/1e6:.2f}' for k in ['fetch_bytes','demotion_bytes','predicted_bytes','predicted_used_bytes']],f'{a["hits"]:.1f}',f'{a["cpu_experts"]:.1f}',f'{a["disk_reads"]:.2f}'])
sections.append(table(['Run / repeat','Demand H2D','D2H','Predicted H2D','Useful prediction','GPU hits/token','CPU experts/token','Disk reads/token'],traffic))
sections.append('Deferred-layers retains stock placement and exact stock logits; its repeat counters provide the scheduling-only reference (403.51 MB demand + 339.78 MB demotion/token). Candidate repeat is 413.79 + 0.71 MB/token: demand is slightly higher, but total bidirectional expert traffic is about 44% lower. Copy-back drops ~99.8%. Counters describe bytes submitted during the scored window; traces separately account for copies crossing token boundaries.')
sections.append('''Stock pinned expert RAM: **87.33 GiB**; candidate: **97.82 GiB** (+10.49). Same **22.00 GiB GPU expert pool**. These are expert allocations, not total process/GPU usage. A 24 GiB duplicate budget is a retention limit inside the RAM tier, not an additional 24 GiB allocation. Windows pinning fallback reduced one class's allocation; not every weight has two copies. The candidate preserves headroom checks and logs actual capacity. Prefill lending releases duplicate backing when needed; this explains why repeated long prompts still cause more demotions than repeated short prompts.''')
memory=[]
for n in ['retained-clean','retained8k-corpus','trace-retained']:
    mm=json.loads((ROOT/n/'memory.json').read_text())
    memory.append([n,f'{min(r["available_physical"] for r in mm)/2**30:.2f}',f'{min(r["available_commit"] for r in mm)/2**30:.2f}'])
sections.append(table(['Run','Minimum physical RAM available, GiB','Minimum commit available, GiB'],memory))
sections.append('## Diagnostic token timelines\n\nGPU timestamp interval unions/intersections, not CPU enqueue durations. Stage times overlap and **must not be summed**. D2H-minus-arithmetic-overlap still includes overlap with pulls, so it is not a causal exposed-latency estimate. The wall-time difference measures net removal/overlap including contention. Profiler RAM overhead is recorded separately; GPU slot count stays 2,693. These diagnostic timings are not the unprofiled TPS table.')
traces=[]
for name,rep in [('trace-control',None),('trace-deferred-layers',None),('trace-retained','0'),('trace-retained','1')]:
    q=json.loads((ROOT/name/('trace-summary.json' if rep is None else 'trace-repeats.json')).read_text())
    if rep is not None:q=q[rep]
    traces.append((name+(' / '+rep if rep is not None else ''),q))
sections.append(table(['Trace','Wall','Expert fetch','D2H','Arithmetic','D2H∩arithmetic','CPU∩arithmetic','No GPU activity'],[[n,*[f'{q[k]:.3f}' for k in ['wall_ms','fetch_ms','d2h_ms','arithmetic_ms','d2h_arithmetic_overlap_ms','cpu_arithmetic_overlap_ms','no_gpu_ms']]] for n,q in traces]))
fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
labels=['Stock first','Deferred first','Combined first','Combined repeat']
axs[0].barh(labels,[q['wall_ms'] for _,q in traces],color=['#67798d','#e1a33b','#219c83','#147761']);axs[0].set_xlabel('Mean profiled token wall time (ms)')
for i,(_,q) in enumerate(traces):axs[0].text(q['wall_ms']+.2,i,f'{q["wall_ms"]:.2f}',va='center')
x=np.arange(4)
for i,(k,label) in enumerate([('fetch_ms','RAM pull'),('d2h_ms','Demotion'),('arithmetic_ms','GPU arithmetic')]):
    axs[1].bar(x+(i-1)*.24,[q[k] for _,q in traces],.24,label=label)
axs[1].set_xticks(x,labels,rotation=15);axs[1].set_ylabel('Stage active ms (overlapping, not additive)');axs[1].legend()
fig.savefig(OUT/'mean-timings.png',dpi=160);plt.close(fig)
sections.append('![Mean profiled timings](mean-timings.png)\n\n![Stock individual token](../trace-control/timeline-median.png)\n\n![Deferred demotions individual token](../trace-deferred-layers/timeline-median.png)\n\n![Retained individual token](../trace-retained/timeline-median.png)')
sections.append('Demotion scheduling alone makes copies last longer (bandwidth contention) but overlaps them, reducing wall time. Backing removes the work instead. Once warmed, fetch remains ~11.05 ms and GPU arithmetic ~9.14 ms; serial dependency, CPU answer waits and host gaps remain. Reaching 60 TPS requires a <=16.67 ms loop and thus another substantial change; the present results do not establish it.')
sections.append('## Prefill and correctness')
sections.append(table(['8,192 occupied-token prompt','First prefill ms','Repeat prefill ms'],[[n,*[f'{x:.2f}' for x in prefill(n)]] for n in ['baseline8k-corpus','shadow8k-corpus','retained8k-corpus']]))
sections.append('''Measured 8K prefill is preserved (about 3.56 s stock vs 3.48 s retained). This is a two-request fixture, not proof at every prompt length. An earlier 16K test used the wrong forward API and was stopped/excluded; corrected driver uses batched prefill followed by the last-token forward. Same corrected driver on the unchanged stock engine supplies the 8K baseline.

Scheduling-only H3 compares all 39,649,280 logits exactly. Original resident-overlap and completed DMA-only tests also match exactly. RAM-backing ownership is isolated using a GPU-only plan; cold loading and per-class byte checks verify immutable copies independently of CPU/GPU arithmetic placement. No numerical threshold was loosened.''')
checks=json.loads((ROOT/'screen-logit-comparisons.json').read_text())
chosen=['deferred-layers','gpu-shadow','gpu-cold-shadow','gpu-clean-cold','retained-clean','retained8k-corpus']
sections.append(table(['Fixed-input test','Reference','Positions','Bit exact','Max absolute delta','RMS delta','Top-1 agreement'],[[q['experiment'],q['reference'],q['positions'],q['exact'],f'{q["max_abs"]:.5f}',f'{q["rms"]:.5f}',f'{q["top1_agreement"]*100:.2f}%'] for q in checks if q['experiment'] in chosen]))
sections.append('''**Hybrid bit-exact parity is not claimed.** Retaining backing changes placement and therefore CPU/GPU assignment. Stock itself changes logits across repeated cache states (max 3.877, RMS .1382; 98.44% top-1 agreement). GPU-only exact comparisons isolate memory/ownership correctness; restoring unchanged arithmetic kernels and exact H3 tests isolate scheduling. This supports retaining the engineering change, but does not turn hybrid numerical differences into a bit-exact parity pass or a broad model-quality guarantee.''')
life=json.loads((ROOT/'lifecycle-retained/requests.json').read_text())
sections.append(table(['Lifecycle request','Completion'],[[r['task'],r['done']] for r in life]))
sections.append('''Successful process exit checks QUIT/teardown after STOP and recovery. Every 64 diagnostic boundaries, dual-owner mappings are checked and a complete GPU/RAM expert payload is compared per live quant class. Representative short/8K repeated requests exercise eviction pressure and prefill lending; deliberately cold GPU-only execution exercises disk demand. This does not certify all multi-GPU/AMD configurations or untested speculative paths.

## Retained implementation and rollback

Branch: `research/maya-transfer-campaign`. Baseline: `615973d` / preserved reproduction branch. All prototypes remain at `research/maya-transfer-prototypes` (`3cf31ef`). Retained cleanup checkpoint: `84c4f17`. Separate unchanged-engine baseline worktree/driver: `research/maya-transfer-baseline-tools` (`4bf9d4c`). No external publication.

Opt-in candidate settings (stock behavior remains available with these MAYA settings unset):

```text
STRATA_GLM_NO_MTP=1
STRATA_GLM_CPU_PLAN=011223344
STRATA_GLM_POOL_GB=22
STRATA_GLM_RAM_GB=101
STRATA_GLM_USAGE=0
MAYA_RAM_SHADOW_MB=24576
MAYA_RAM_SHADOW_EXPAND=1
MAYA_DEFER_DEMOTIONS=layers
```

The headroom-limited RAM budget is not a guarantee that every foreground workload fits simultaneously. Use the logged allocation and available-memory record. The standard daily launcher/config was not silently changed.

Reproduce via `tools/run_transfer_lab.py`, `tools/profile_decode.py`, `tools/validate_retained_transfer.ps1`; summarize via `tools/summarize_transfer.py` then `tools/report_transfer_campaign.py`. Evidence includes manifests, per-token CSVs, memory samples, logits locally, CUDA traces locally, exported timing tables, request outputs and failure logs. Rejected prototype flags require the prototype branch. Large logits/traces are retained locally and intentionally not committed.

Next bottleneck is demand RAM→VRAM traffic and dependent compute, not bulk warm copy-back. Revisit prediction only with materially better usefulness/demand priority; revisit overlap only with lower submission/launch cost. More RAM duplication alone has diminishing returns once demotions are almost zero.''')
text='\n\n'.join(sections)+'\n';(OUT/'RESULTS.md').write_text(text,encoding='utf-8')
# Portable HTML view: same Markdown source rendered by a small local converter.
try:
    import markdown
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
except ImportError:
    body='<pre>'+html.escape(text)+'</pre>'
(OUT/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Maya transfer R&D</title><style>body{max-width:1200px;margin:40px auto;font:16px/1.5 system-ui;padding:0 20px;color:#192b3c}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{border:1px solid #ccd4dc;padding:8px;text-align:left}th{background:#edf4f6}img{max-width:100%}pre{white-space:pre-wrap;background:#edf4f6;padding:16px}h1,h2{color:#126d5b}</style><body>'+body+'</body></html>',encoding='utf-8')
print(OUT/'RESULTS.md')
