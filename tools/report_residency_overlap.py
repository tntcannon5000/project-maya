"""Focused campaign report from saved evidence; no inference."""
import csv,json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'docs/evidence/maya-residency-overlap'
out=R/'report';out.mkdir(exist_ok=True)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def read(p):return json.loads(p.read_text())
measure=read(R/'measurements.json')
def row(name,repeat):return next(r for r in measure if r['name']==name and r['repeat']==repeat)
s=['# Maya: focused residency and overlap results',
'''Starting point: retained graph-enabled Maya (`3a4f906`), keeping the prior
RAM-backing/demotion improvements. Same GSQ-RCO 3.0bit Q4-attention packed
weights, c=1, 22 GiB expert pool, no MTP, speculative tokens, K4V, expert
skipping or further quantization. Nothing published. **60 TPS not reached.**

Retained as an opt-in research candidate: causal next-route reuse for eviction,
scored after the CPU answer is published, combined with 32-block down-component
copies restricted to captured recurrent layers. Ordinary fetch remains on
uncaptured layers. Defaults and the daily configuration are unchanged.

## Unprofiled results

Short fixed-input continuation: 30 prompt tokens and 128 scored tokens per
request, first then repeated. Comparison uses a fresh same-day graph baseline.
The experiment sequence was adaptive, one or two targeted cases at a time.''']
rows=[]
for n in ['baseline','transition','transition-late','transition-overlap','overlap-captured','combined-captured']:
    a,b=row(n,0),row(n,1)
    rows.append([n,f'{a["TPS"]:.2f}',f'{b["TPS"]:.2f}',f'{b["ms"]:.3f}',f'{b["fetch_bytes"]/1e6:.2f}',f'{b["demotion_bytes"]/1e6:.2f}'])
s.append(table(['Path','First TPS','Repeat TPS','Repeat ms/token','Demand MB/token','Demotion MB/token'],rows))
s.append('The final combination improves repeated TPS by about 4.4% and reduces repeated demand by 7.4%. Individual mechanisms show much smaller or no reliable throughput gains. Prior run variation was about 4%; these bounded results support a modest candidate, not a precise universal speedup or a large breakthrough.')
s.append('## Held-out web generation and lifecycle')
free=[read(R/n/'requests.json') for n in ['web-baseline','web-candidate']]
rows=[]
for a,b in zip(*free):
    assert a['task']==b['task'] and a['repeat']==b['repeat']
    rows.append([a['task'],a['repeat'],a['tokens'],f'{a["tokens"]*1000/a["decode_ms"]:.2f}',f'{b["tokens"]*1000/b["decode_ms"]:.2f}',a['output_ids']==b['output_ids']])
s.append(table(['Request','Repeat','Tokens','Baseline TPS','Candidate TPS','Same output IDs'],rows))
s.append('Both sides ran the same warmup, STOP after eight generated tokens, recovery, then two web requests. All generated token IDs matched. The web prompt asks for an HTML/JavaScript bouncing-ball page; generation is capped at 128 tokens, not a completed webapp or quality benchmark. STOP and subsequent request/teardown succeeded. Pooled web throughput improves about 3.3%.')
s.append('## What was learned and retained')
s.append(table(['Mechanism','Evidence','Decision'],[
 ['Recent-32 frequency','Offline CSV traffic worsened3.4%; writing improved1.2%.','Reject; no production implementation.'],
 ['Causal transition reuse','Live repeat demand413.79→383.00MB; 7,815 changed victim choices in13,272 routes.','Retain opt-in as part of combination.'],
 ['Move reuse scoring after CPU publication','Same measured placement/outputs, repeat42.72→43.54TPS.','Keep dependency; isolate bookkeeping from required answer.'],
 ['Limit overlapping copy blocks to32','Arithmetic stays near9.1ms instead of prior>11ms; real overlap~0.65ms.','Resource cap useful; all-layer version still pays gaps.'],
 ['Captured-layer-only overlap','Prior added gaps were almost entirely uncaptured work; final combination45.20TPS.','Retain opt-in; ordinary fetch elsewhere.'],
 ['Strata-style admission margin','Source uses gain-filtered/budgeted replacement; Maya admits fetched misses opportunistically.','Future hypothesis, not implemented/measured in this round.']]))
s.append('The transition predictor only ranks eviction victims. It never proactively transfers predicted weights. Training uses observations available so far, with frequency smoothing 16 and per-row aging at 2048 observations. The history persists across requests. This is a limited causal policy, not the future-knowing oracle. Repeated fixtures can reward learned transitions; the held-out web continuation checks some generalization, not broad workload coverage.')
s.append('## Diagnostic time and overlap')
traces=[('Prior graph baseline',R.parent/'maya-demand/trace-graphs'),('Transition before deferred scoring',R/'trace-transition'),('All-layer overlap32',R/'trace-overlap32'),('Final combination',R/'trace-final')]
keys=['wall_ms','fetch_ms','arithmetic_ms','cpu_wait_ms','no_gpu_ms','fetch_arithmetic_overlap_ms']
rows=[]
for name,p in traces:
    # Final capture deliberately uses one request only. Compare first requests.
    v=read(p/'trace-repeats.json')['0'];rows.append([name,*[f'{v[k]:.3f}' for k in keys]])
s.append(table(['Profiled FIRST request','Wall ms','Fetch ms','Arithmetic ms','CPU-wait ms','GPU idle ms','Fetch/math overlap ms'],rows))
s.append('Intervals overlap and are not additive. Profiles are separate from headline TPS; the final capture is one 128-token request, not a repeated warm run. The historical graph trace was recorded earlier and is not a contemporaneous significance test. CPU-wait kernel time is not all removable stall.')
engagement=read(R/'trace-final/component-engagement.json')
s.append(table(['Observed copy grid blocks','Inside captured graph','Kernel executions, including prompt'],[[r['blocks'],r['captured'],r['launches']] for r in engagement]))
s.append('Final capture verifies the intended restriction: all component kernels ran as graph nodes, with 32-block down copies and 170-block prefix copies. Measured first-request fetch/arithmetic overlap is 0.683 ms; GPU inactivity returns to 1.147 ms, near the prior graph-only 1.156 ms. Arithmetic stays near 9.24 ms. The earlier added gap has been removed without reviving the >11 ms arithmetic slowdown.')
g=read(R/'gap-attribution.json')
s.append(table(['Repeated diagnostic','Uncaptured gaps ms','Graph-boundary gaps ms','Inside-graph gaps ms'],[[k,*[f'{v[x]:.3f}' for x in ['uncaptured','graph_boundary','inside_graph']]] for k,v in g.items()]))
s.append('![Final individual token](../trace-final/timeline-median.png)\n\n![Resource-limited all-layer overlap](../trace-overlap32/timeline-median.png)')
s.append('## Numerical, prefill and ownership checks')
parity=read(R/'parity.json')
s.append(table(['Case','Logits compared','Exact','Maximum absolute difference','Top1 agreement'],[[p['name'],p['logits'],p['exact'],f'{p["max_abs"]:.6f}',f'{100*p["top1"]:.2f}%'] for p in parity if p['name'] in ['overlap-captured','combined-captured','gpu8k-candidate']]))
pf=[]
for n in ['gpu8k-baseline','gpu8k-candidate']:
    log=(R/n/'engine.log').read_text();ms=float(re.search(r'prefill repeat=0 tokens=8192 ms=([\d.]+)',log)[1])
    pf.append([n,f'{ms:.2f}',f'{8192*1000/ms:.1f}', 'verified all live dual owners' in log])
s.append(table(['Controlled GPU-only, 8K occupied','Prefill ms','Input tokens/s','Ownership check'],pf))
s.append('GPU-only comparison uses the same backend for every required expert and matches every compared logit exactly. Q2_K/Q3_K/Q4_K packed expert bytes are unchanged. Normal hybrid placement differs numerically: the residency policy changes CPU/GPU assignment and can change subsequent routes even with fixed input tokens. All scheduling variants of that same residency policy produced identical compared logits to one another. No parity threshold was weakened. This is not a claim of exact hybrid parity. The targeted 8K check found no prefill regression; its single timing pair is not evidence for a prefill optimization.')
s.append('## Memory, limitations and PR scope')
mem=[]
for n in ['baseline','combined-captured','gpu8k-candidate']:
    z=read(R/n/'memory.json');m=row(n,0)
    mem.append([n,f'{m["pool_gib"]:.2f}',f'{m["ram_gib"]:.2f}',f'{min(x["available_physical"] for x in z)/2**30:.2f}'])
s.append(table(['Case','Expert pool GiB','Pinned tier GiB','Minimum free physical GiB'],mem))
s.append('The transition tables add roughly 15 MiB of ordinary host memory, with no extra expert slots or background uploads. GPU clocks/thermal samples are stored alongside fixed runs; clocks were not locked and minor differences remain uncertain. CUDA/NVIDIA c=1 was built and exercised. AMD, multi-GPU and speculative/concurrent integration are unqualified. The history policy should be reviewed for those future paths separately.')
s.append('''Research branch: `research/maya-residency-overlap`. Checkpoints:
`85385a4` initial prototypes, `e765b0c` CPU-answer ordering,
`19dd065` captured-layer restriction, `276a0aa` combined candidate/teardown.
Baseline rollback: `3a4f906`. Implementation remains opt-in in
[maya-residency-overlap.json](../../../../configs/maya-residency-overlap.json).
Retained implementation/configuration checkpoint: `32821bc`.

PR-facing changes are the reuse policy helper and integration, bounded
component-copy kernel/fork-join, and teardown ordering. Keep experimental
tools/evidence separate during PR preparation. No PR was opened or published.
The detailed [audit](../../../MAYA-RESIDENCY-OVERLAP.md) preserves failed ideas,
dependency findings and source references. Remaining demand is still 383 MB/token;
neither smarter eviction nor overlap has removed the dominant transfer cost.''')
text='\n\n'.join(s)+'\n';(out/'RESULTS.md').write_text(text,encoding='utf-8')
import markdown
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Maya residency and overlap</title><style>body{max-width:1200px;margin:36px auto;padding:0 20px;font:16px/1.5 system-ui}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:8px;border:1px solid #ddd}img{max-width:100%}</style>'+markdown.markdown(text,extensions=['tables']),encoding='utf-8')
print(out/'RESULTS.md')
