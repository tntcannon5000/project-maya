"""Summarize integration evidence without treating profiled runs as performance."""
from pathlib import Path
import csv,json,re,statistics,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/'docs/evidence/maya-release-integration'
rows=[];parity=[]
for p in D.glob('*/fixed.csv'):
 data=list(csv.DictReader(p.open()));log=(p.parent/'engine.log').read_text()
 for rep in sorted({int(x['repeat']) for x in data}):
  ds=[x for x in data if int(x['repeat'])==rep];ms=[float(x['ms']) for x in ds]
  pf=re.search(r'prefill repeat='+str(rep)+r' tokens=(\d+) ms=([\d.]+)',log)
  rows.append(dict(test=p.parent.name,repeat=rep,tokens=len(ds),tps=1000/statistics.mean(ms),mean_ms=statistics.mean(ms),prefill_ms=float(pf[2]) if pf else None,**{k:statistics.mean(float(x[k]) for x in ds) for k in ['pool_slots','ram_slots','pool_gib','ram_gib','disk_reads','cpu_experts']}))
for left,right in [('before-short','merged-short'),('before-8k','merged-8k'),('gpu-before','gpu-merged'),('gpu-before','gpu-merged-glu-off'),('gpu-stock117','gpu-merged'),('gpu-short-before','gpu-short-merged'),('confirm-before','matched-capacity')]:
 paths=[D/x/'fixed.logits.f32' for x in [left,right]]
 if not all(p.exists() for p in paths):continue
 manifests=[json.loads((D/x/'manifest.json').read_text()) for x in [left,right]]
 assert manifests[0]['tokens_sha256']==manifests[1]['tokens_sha256']
 assert manifests[0]['weights']==manifests[1]['weights']
 a,b=[np.memmap(p,dtype='float32',mode='r') for p in paths]
 if a.shape!=b.shape:continue
 n=len(list(csv.DictReader((D/left/'fixed.csv').open())));a=a.reshape(n,-1);b=b.reshape(n,-1)
 maxima=np.array([np.max(np.abs(x-y)) for x,y in zip(a,b)])
 diff=np.flatnonzero(maxima)
 parity.append(dict(reference=left,candidate=right,logits=a.size,finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),exact=bool(np.array_equal(a,b)),max_abs=float(maxima.max()),first_differing_row=int(diff[0]) if len(diff) else None,matching_argmax=int(np.sum(a.argmax(1)==b.argmax(1))),rows=n))
self_checks=[]
for name in ['gpu-before','gpu-merged','gpu-merged-glu-off','gpu-stock117','gpu-short-before','gpu-short-merged']:
 p=D/name/'fixed.logits.f32'
 if not p.exists() or p.stat().st_size!=39649280*4:continue
 a=np.memmap(p,dtype='float32',mode='r').reshape(256,-1)
 self_checks.append(dict(test=name,exact=bool(np.array_equal(a[:128],a[128:])),max_abs=float(np.max(np.abs(a[:128]-a[128:]))),matching_argmax=int(np.sum(a[:128].argmax(1)==a[128:].argmax(1)))))
(D/'summary.json').write_text(json.dumps(dict(timing=rows,parity=parity,self_repeat=self_checks),indent=2))
lines=['# Maya v1.1.7 integration qualification','','- Date: 14 October 2026. Upstream release `1e225b9`, integration `736ac46`, pre-merge retained source `80d4abb`.','- Same GSQ-RCO 3.0bit Q4-attention pack/profile and retained switches; no MTP, K4V, token speculation or concurrent-graph experiment.','- c=1: 192K allocated capacity, short/8192-token occupied prompts, 256 forced continuation tokens, two requests. Fixed token/weight hashes verified. TPS is tokens / summed forward latency; logits writes are outside timed regions, no Nsight.','- Run order: before short, merged short, merged 8K, before 8K. GPU-only runs are correctness diagnostics, not headline TPS.','- This is a bounded non-regression screen, not a guarantee on all workloads or hardware. Historical TPS is not substituted for the fresh baseline.','','| Context | Request | Before TPS | Merged TPS | Change | Before prefill ms | Merged prefill ms |','|---|---|---:|---:|---:|---:|---:|']
by={(x['test'],x['repeat']):x for x in rows}
for task in ['short','8k']:
 for rep in [0,1]:
  a=by.get(('before-'+task,rep));b=by.get(('merged-'+task,rep))
  if a and b:lines.append(f"| {task} | {rep+1} | {a['tps']:.2f} | {b['tps']:.2f} | {(b['tps']/a['tps']-1)*100:+.2f}% | {a['prefill_ms']:.1f} | {b['prefill_ms']:.1f} |")
lines+=['','## Reversed-order short confirmation','','| Request | Before TPS | Merged TPS | Change |','|---|---:|---:|---:|']
for rep in [0,1]:
 a=by.get(('confirm-before',rep));b=by.get(('confirm-merged',rep))
 if a and b:lines.append(f"| {rep+1} | {a['tps']:.2f} | {b['tps']:.2f} | {(b['tps']/a['tps']-1)*100:+.2f}% |")
lines+=['','## Capacity-matched short check','','| Request | Before TPS | Merged TPS | Change |','|---|---:|---:|---:|']
for rep in [0,1]:
 a=by.get(('confirm-before',rep));b=by.get(('matched-capacity',rep))
 if a and b:lines.append(f"| {rep+1} | {a['tps']:.2f} | {b['tps']:.2f} | {(b['tps']/a['tps']-1)*100:+.2f}% |")
lines+=['','- This ablation requests 97.822 GiB instead of 101 GiB to approximately match the old binary physical RAM tier; per-class slot counts still differ, so this is a capacity sensitivity check, not an exact placement control or portable default.','', '## Numerical checks','','| Comparison | Logits | Exact | Max absolute difference | Matching argmax rows |','|---|---:|---|---:|---:|']
for p in parity:lines.append(f"| {p['reference']} -> {p['candidate']} | {p['logits']:,} | {p['exact']} | {p['max_abs']:.6g} | {p['matching_argmax']}/{p['rows']} |")
lines+=['','### Same-binary GPU-only repeat controls','','| Binary/fixture | Repeat exact | Max difference |','|---|---|---:|']
for x in self_checks:lines.append(f"| {x['test']} | {x['exact']} | {x['max_abs']:.6g} |")
lines+=['','- Hybrid comparisons can change CPU/GPU assignments; a fixed input history alone does not establish identical routing or arithmetic. No thresholds are relaxed.','- The first 73 short-fixture rows matched exactly. The first differing row also changed hit/fetch/CPU-expert counts. GPU-only comparison isolates backend placement; it does not make hybrid outputs exact.','','## Memory and execution','','| Run | RAM tier GiB | GPU pool GiB | Slots | Minimum free RAM GiB |','|---|---:|---:|---:|---:|']
for label in ['before-short','merged-short','before-8k','merged-8k']:
 x=by.get((label,0));p=D/label/'memory.json'
 if x and p.exists():
  mem=json.loads(p.read_text());lines.append(f"| {label} | {x['ram_gib']:.2f} | {x['pool_gib']:.2f} | {x['pool_slots']:.0f} | {min(q['available_physical'] for q in mem)/2**30:.2f} |")
lines+=['','- Upstream pinning-limit sizing can allocate more RAM backing despite identical requested budgets. This is a real integration effect and a capacity confound, not a per-kernel attribution.','- Transfer-byte fields in this stock-compatible driver are placeholders and must not be read as zero traffic.','','## Concurrent server smoke/performance','','| Build | First aggregate TPS | Repeat aggregate TPS |','|---|---:|---:|']
for label in ['before','merged']:
 p=D/('c4-'+label)/'retained/results.json'
 if p.exists():
  v=json.loads(p.read_text())
  if len(v)==2:lines.append(f"| {label} | {v[0]['aggregate_active_tps']:.2f} | {v[1]['aggregate_active_tps']:.2f} |")
lines+=['','- c=4 uses four real coding generations at 32K per slot, 256 output tokens each, two cohorts; aggregate TPS covers the interval all four lanes emit. It is not comparable directly with c=1 replay or proof of numerical parity.','- Focused native split/rows exact CPU tests passed; lifecycle and GPU ownership results are recorded in their logs.','- Full hardware audit and PR-ready work bullets: ../../PR-WORK-LOG.md. Raw logits remain local and ignored.']
lines+=['','## Verdict','','- Integration is built and checkpointed. No broad throughput loss is established: the normal-budget short warm case is 1.6-1.8% lower in two pairs, the repeated 8K case is flat, and repeated c=4 is 3.8% higher. Do not claim literally zero loss at every setting.','- At approximately matched RAM capacity, short warm decode is 32.23 TPS versus 31.37 TPS for the confirmed old binary (+2.8%). This supports memory/placement sensitivity; it is not proof of a universally faster allocator or code path. The ordinary 101 GiB preset remains available; the 97.822 GiB setting is a local ablation, not a new default.','- Short serial-prefill GPU-only decode is exact across 39,649,280 logits. Long batched-prefill repeat controls differ even within the old and stock upstream binaries; exact long-prefill parity remains unresolved. Disabling upstream fused SwiGLU did not resolve it.','- Merged lifecycle: warmup, STOP after eight tokens, recovery and QUIT passed. Native split and multi-row exact tests passed. Ownership checks passed for live owners and sampled weight bytes per mixed quant class.','- Prepared an ignored local preset, maya-retained-v117.local.json, with the retained settings and original 128K capacity; no server is left running. This is not a new product-wide default.']
(D/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
