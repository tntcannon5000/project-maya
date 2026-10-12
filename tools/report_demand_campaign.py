"""Campaign-two report, generated only from persisted evidence; no inference."""
import csv,json,re,statistics,html
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'docs/evidence/maya-demand'
OUT=ROOT/'report';OUT.mkdir(exist_ok=True)
def tbl(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def stats(name,rep):
    rows=[r for r in csv.DictReader((ROOT/name/'fixed.csv').open()) if int(r['repeat'])==rep]
    z={k:statistics.mean(float(r[k]) for r in rows) for k in rows[0]}
    z.update(TPS=1000/z['ms'],p95=float(np.percentile([float(r['ms']) for r in rows],95)))
    return z
def pair(n):return ' / '.join(f'{stats(n,r)["TPS"]:.2f}' for r in [0,1])
def j(path):return json.loads((ROOT/path).read_text())
sections=['# Maya demand-transfer campaign: incremental results',
'''This campaign starts from the **previous retained candidate `20af539`**, including expanded immutable RAM backing and after-embedding demotion scheduling. Its gains over stock Maya are not gains of this campaign. Same GSQ-RCO 3.0bit Q4-attention pack, c=1, no speculative token decoding, MTP, K4V, expert skipping or additional quantization. **60 TPS has not been measured.**

Seven distinct hypotheses were exercised. The simplest promising candidate enables Maya's existing KDA layer graph replay. This is a modest scheduling/configuration improvement, not a newly invented graph engine. Rejected policy/transport prototypes remain preserved separately. The preceding campaign's demotion fix was not reimplemented.

## Baseline versus graph candidate

Unprofiled, without route logging. Short fixture: 30 prompt tokens, then 128 forced inputs; two repetitions with tier adaptation. Long fixture: 8,192 actually occupied source-code tokens, then 128 forced inputs. Allocation capacity 131,072 is not occupancy. Both use a 22 GiB expert pool and the previous RAM backing settings.''']
rows=[]
for label,b,c in [('Short','baseline','confirm-graphs'),('8K occupied','long-baseline','long-graphs')]:
    for rep in [0,1]:
        a=stats(b,rep);z=stats(c,rep)
        rows.append([label+(' first' if rep==0 else ' repeated'),f'{a["TPS"]:.2f}',f'{z["TPS"]:.2f}',f'{(z["TPS"]/a["TPS"]-1)*100:+.1f}%',f'{a["ms"]:.2f} → {z["ms"]:.2f}',f'{a["p95"]:.2f} → {z["p95"]:.2f}'])
sections.append(tbl(['Fixture','Retained R1 TPS','Graph TPS','Change','Mean ms/token','p95 ms/token'],rows))
sections.append('Short confirmation uses the experimental executable with other new switches off. Final free generation and 8K qualification use **the same preserved R1 executable on both sides**, with only graph replay changed. Prior campaign repeat variation was ~4%; do not sell a single 3–4% difference as a precisely established universal speedup.')
free=[];outputs_equal=True
for task in ['csv','explanation','web']:
    sides=[];ids=[]
    for n in ['free-baseline','free-graphs']:
        rs=[r for r in j(n+'/requests.json') if r['task']==task]
        sides.append(sum(r['tokens'] for r in rs)*1000/sum(r['decode_ms'] for r in rs));ids.append([r['output_ids'] for r in rs])
    outputs_equal &= ids[0]==ids[1]
    free.append([task,f'{sides[0]:.2f}',f'{sides[1]:.2f}',f'{(sides[1]/sides[0]-1)*100:+.1f}%',ids[0]==ids[1]])
sections.append('## Writing/coding and workload transitions\n\nOne process per side: warmup, CSV → explanation → HTML/JavaScript, then repeat. Same greedy prompts and 128-output-token cap. This checks cross-task cache transitions; these are continuations, not completed webapp tasks or a model-quality benchmark.')
sections.append(tbl(['Task, two requests pooled','Retained R1 TPS','Graph TPS','Change','Same output IDs'],free))
sections.append('## Hypotheses/results matrix\n\nThese screens enabled route logging (which also computes 16 near ranks). They are **diagnostic**, not directly comparable to the unlogged headline. The initial retained route collection overlapped CPU compilation; use it for route/capacity evidence, not a definitive timing baseline.')
spec=[('D1','Cost/reuse-weighted layer residency','trained-placement','Train writing/web; hold out CSV; same byte pool. Demand falls only ~2%.','No incremental retention: graph+placement does not materially beat graphs.'),
('D2','LRU versus aged-frequency eviction','lru','Actual repeat demand rises, consistent with fixed-route replay.','Reject.'),
('D3','Component-ready GU/down streaming','components','Original single gate/up; down fetch on side stream; graph-compatible fork/join.','Reject standalone; evaluate overlap trace and graph ablation.'),
('D4','Independent expert pulls in one grid','batch-pull','Same bytes, destinations and exact logits; no extra host handshake.','No justified gain alone or with graphs.'),
('D5','Layer graph replay','graphs','34 captures, 10,710 replays; exact logits and ordered routes.','Simplest candidate; modest gains require workload-specific interpretation.'),
('D6','Measured per-layer CPU/GPU balance','cpu-plan','Cost model changes seven of 42 plans; actual masks/traffic change.','Reject: extra complexity without speed gain.'),
('D7','Selective rank-0 demand-first prediction','selective-predict','~96% of prefetched bytes selected next, but total H2D rises.','Reject: selected does not mean avoided GPU demand.')]
sections.append(tbl(['ID','Mechanism','First / repeat TPS','Verified engagement','Decision'],[[i,h,pair(n),v,d] for i,h,n,v,d in spec]))
sections.append('Mechanisms, source references, pre-experiment bounds and falsifiers: [campaign plan](../../../MAYA-DEMAND-CAMPAIGN.md), [research/audit](../RESEARCH.md). Public grounding reuses archived CUDA/Fiddler/ProMoE material and measured local DS4/Strata failures; publications are not treated as measured gains on this machine.')
sections.append('## Dependencies and combinations')
sections.append(tbl(['Complete combination','First / repeat TPS'],[[n,pair(n)] for n in ['graphs','graph-batch','graph-components','graph-placement','confirm-graphs']]))
sections.append('D1 changes residency and therefore CPU placement; D6 models CPU completion versus added transfer rather than treating every CPU wait as removable. D3 needs low submission overhead, so D3+D5 includes a captured cross-stream event fork/join. D4+D5 tests whether the batched copy mapping becomes useful once host launches are reduced. D1+D5 barely changes the graph-only result. D7 starts predicted movement after demand/CPU work and restricts rank, but still adds uploads. No exhaustive parameter sweep was used.')
sections.append('## Capacity and transfer accounting')
cap=j('route-capacity-study.json')
sections.append(tbl(['Fixed-route replay model','GPU demand MB/token','Missed experts/token'],[[k,f'{v["mean_fetch_MB"]:.1f}',f'{v["misses_per_token"]:.1f}'] for k,v in cap['replay'].items()]))
sections.append('Replay includes 316 prompt+continuation histories, simplified spares/admission and a global CPU split. Future-oracle numbers assume inaccessible future knowledge and no migration scheduling costs. They are an **optimistic opportunity estimate, not a rigorous hardware/engine lower bound or achieved TPS**. The remaining traffic is not demonstrably all unavoidable, but the tested causal allocation policy removes only a small fraction. Actual placement changes can alter routing numerically; the replay deliberately fixes routes and is kept separate from real measurements.')
traffic=[]
for n in ['baseline','confirm-graphs','trained-placement','lru','cpu-plan','selective-predict']:
    a=stats(n,1);traffic.append([n,*[f'{a[k]/1e6:.2f}' for k in ['fetch_bytes','predicted_bytes','predicted_used_bytes','demotion_bytes']],f'{a["hits"]:.1f}',f'{a["cpu_experts"]:.1f}',f'{a["pool_gib"]:.2f}',f'{a["ram_gib"]:.2f}'])
sections.append(tbl(['Repeat','Demand MB/token','Predicted MB','Selected prediction MB','Demotion MB','GPU hits/token','CPU experts/token','Pool GiB','Pinned tier GiB'],traffic))
sections.append('Prediction engagement distinguishes selected bytes from counterfactual demand saved: an expert otherwise assigned to CPU can be predicted correctly but add a GPU upload. Exact per-expert displaced traffic and prediction deadline distributions were not instrumented after the aggregate screen rejected this variant; do not claim those measurements. Warm demotions remain almost zero in the primary candidate. Graph replay does **not** reduce the ~414 MB/token demand.')
mem=[]
for n in ['baseline','confirm-graphs','long-baseline','long-graphs','trace-baseline','trace-graphs']:
    samples=j(n+'/memory.json');mem.append([n,f'{min(x["available_physical"] for x in samples)/2**30:.2f}',f'{min(x["available_commit"] for x in samples)/2**30:.2f}'])
sections.append(tbl(['Run','Minimum available physical GiB','Minimum available commit GiB'],mem))
sections.append('No extra expert RAM/pool capacity is credited to graphs. Hardware samples are in each fixed-run gpu.csv (including loading/prefill/idle). VRAM clocks/thermals were not locked or changed by the campaign; these are bounded trials, not statistical confidence intervals. Profiler overhead and cache capacity are recorded separately.')
sections.append('## New diagnostic timelines\n\nNsight node-level graph tracing is explicitly enabled; otherwise graph launches could hide their constituent kernels. Stage intervals overlap and must not be summed. CPU-wait kernel time includes polling/overhead; it is not all removable stall. No-GPU gaps are measured from the union of GPU activity within token ranges.')
trs=[]
for n in ['trace-baseline','trace-graphs','trace-components']:
    q=j(n+'/trace-repeats.json')
    for rep in ['0','1']:trs.append((n+' / '+rep,q[rep]))
keys=['wall_ms','fetch_ms','arithmetic_ms','cpu_wait_ms','no_gpu_ms','fetch_arithmetic_overlap_ms','cpu_arithmetic_overlap_ms']
sections.append(tbl(['Trace / repeat','Wall ms','Pull ms','Arithmetic ms','CPU-wait kernel ms','No-GPU ms','Pull∩arithmetic ms','CPU∩arithmetic ms'],[[n,*[f'{q[k]:.3f}' for k in keys]] for n,q in trs]))
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
names=['Retained R1','Graphs','Graphs + components'];vals=[q for n,q in trs if n.endswith('/ 1')]
axes[0].barh(names,[q['wall_ms'] for q in vals],color=['#67798d','#208d75','#d3942f']);axes[0].set_xlabel('Profiled repeated token wall time (ms)')
x=np.arange(3)
for i,(k,label) in enumerate([('fetch_ms','Fetch'),('cpu_wait_ms','CPU wait'),('no_gpu_ms','GPU inactive')]):axes[1].bar(x+(i-1)*.25,[q[k] for q in vals],.25,label=label)
axes[1].set_xticks(x,names);axes[1].set_ylabel('Active stage/gap ms; not additive');axes[1].legend()
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
sections.append('![Mean diagnostics](comparison.png)\n\n![Retained token](../trace-baseline/timeline-median.png)\n\n![Graph token](../trace-graphs/timeline-median.png)\n\n![Component-streaming token](../trace-components/timeline-median.png)')
sections.append('## Prefill, numerical and lifecycle qualification')
pf=[]
for n in ['long-baseline','long-graphs']:
    values=re.findall(r'prefill repeat=\d+ tokens=8192 ms=([\d.]+)',(ROOT/n/'engine.log').read_text());pf.append([n,*values])
sections.append(tbl(['8K prefill','First ms','Repeated ms'],pf))
parity=j('final-parity.json');sections.append(tbl(['Comparison','Logits','Exact','Max absolute difference'],[[r['name'],r['logits'],r['exact'],r['max_abs']] for r in parity]))
sections.append('Scheduling/transport-only screens also compare all logits and actual ordered route IDs. Placement/CPU/prediction variants show hybrid differences; those are recorded in parity.json, not declared bit-exact or excused by weaker thresholds. The final graph candidate preserves backend assignment. Eight experts remain required per route; mixed Q2_K/Q3_K/Q4_K weights are unchanged.')
life=j('lifecycle-graphs/requests.json');sections.append(tbl(['Request','Engine completion'],[[r['task'],r['done']] for r in life]))
sections.append('STOP/recovery and normal teardown completed, with live dual-owner checks and sampled immutable bytes per quant class. Long repeated requests exercise prefill lending. Qualification is c=1 on this NVIDIA/Windows setup, not AMD/multi-GPU certification.')
sections.append('''## Retention, rollback and remaining bottlenecks

Research branch: `research/maya-demand-campaign`. Primary rollback: `20af539`
on `research/maya-transfer-campaign`. Prototypes are preserved separately.
The graph candidate adds `STRATA_GLM_KDA_GRAPH=1` to the previous retained
settings; it does not replace the model or arithmetic kernels. The daily
launcher is not silently changed. Nothing is published externally.

Remaining bottlenecks: substantial demand uploads, required projections and
expert computation, and a CPU/GPU balancing problem where removing waits can
cost more transfer time than it saves. The overclocked GPU's VRAM bandwidth
does not remove the RAM→GPU link limit. Bounded arithmetic inspection found
no established large redundant-work opportunity, so no generic kernel rewrite
was undertaken. The oracle reuse opportunity warrants future work, but this
campaign does not demonstrate a deployable policy achieving it.

Reproduction: run_demand_lab.py, analyze_demand_routes.py,
qualify_demand_graphs.ps1, analyze_transfer_trace.py, summarize_demand.py and
report_demand_campaign.py. Large raw logits/traces remain local. The research
log records negative results, methodology corrections and uncertainty.''')
text='\n\n'.join(sections)+'\n';(OUT/'RESULTS.md').write_text(text,encoding='utf-8')
import markdown
body=markdown.markdown(text,extensions=['tables','fenced_code'])
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Maya demand campaign</title><style>body{max-width:1250px;margin:40px auto;font:16px/1.5 system-ui;padding:0 20px;color:#182c3d}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{border:1px solid #ccd5df;padding:8px}th{background:#eaf3f0}img{max-width:100%}h1,h2{color:#126c5b}</style>'+body,encoding='utf-8')
print(OUT/'RESULTS.md')
