from pathlib import Path
import json,csv,statistics
r=Path(__file__).resolve().parents[1]; d=r/'docs/evidence/maya-upstream-117-ab'
a=json.loads((d/'summary.json').read_text()); assert len(a)==8 and all(x['tokens']==512 for x in a)
by={(x['test'],x['repeat']):x for x in a}
for task in ['short','8k']:
 m=[json.loads((d/(b+'-'+task)/'manifest.json').read_text()) for b in ['upstream','retained']]
 assert m[0]['tokens_sha256']==m[1]['tokens_sha256']
 assert m[0]['weights']==m[1]['weights']
lines=['# Maya v1.1.7 versus retained fork at 192K','', '2026-10-13. Upstream `1e225b9`; retained implementation `57b4d14`, all latest experimental switches off. No engine implementation changed for this comparison. Upstream checkout: `../project-maya-upstream-v1.1.7`. Both full upstream engine and common benchmark driver built successfully.', '', '## Method', '', 'c=1, 196608-token allocated context, same GSQ-RCO 3.0bit Q4-attention packed weights/profile, fixed CPU plan 011223344, requested 22 GiB GPU pool, 101 GiB RAM budget. Both actually fitted 2674 GPU slots / 21.837 GiB. Upstream retained its normal exclusive tier ownership; our fork enabled its retained RAM shadow and scheduling configuration. KV streaming and all speculative token decoding disabled. This is a configuration A/B, not an ablation of one change.', '', 'Two workloads: 30-token prompt and 8192-token occupied source-code prompt. Each scores 512 forced-input tokens from the first 512 tokenizer tokens of serve/server.py at 57b4d14, twice after tier warming. This is a code replay fixture, not independently generated output or a task-quality benchmark. Occupancy grows by 512; neither fills the 192K window. Each pair has identical token-file and packed-weight hashes. Fixed history does not imply identical floating-point routes across implementations.', '', 'Order: upstream short, retained short, retained 8K, upstream 8K. TPS is 512 divided by summed forward() latency; logits writes are outside timed regions. No Nsight profiling. Normal tier diagnostics are enabled equally. First means first scored request after loading/warming, not cold disk startup. Only two requests per condition: no confidence intervals and no broad workload claim.', '', '## Decode', '', '| Occupied prompt | Request | Upstream TPS | Retained TPS | Retained gain |', '|---|---|---:|---:|---:|']
for task in ['short','8k']:
 for rep in [0,1]:
  u=by[('upstream-'+task,rep)]; c=by[('retained-'+task,rep)]
  lines.append(f"| {u['prefill_tokens']} | {'First' if rep==0 else 'Repeat'} | {u['tps']:.2f} | {c['tps']:.2f} | {(c['tps']/u['tps']-1)*100:.1f}% |")
lines+=['','## 8K prefill','', '| Request | Upstream seconds | Retained seconds |','|---|---:|---:|']
for rep in [0,1]: lines.append(f"| {rep+1} | {by[('upstream-8k',rep)]['prefill_ms']/1000:.3f} | {by[('retained-8k',rep)]['prefill_ms']/1000:.3f} |")
lines+=['','## Capacity and limits','','| Build/workload | RAM tier GiB | Minimum free system RAM GiB | Peak GPU MiB | Mean disk reads/token (repeat) |','|---|---:|---:|---:|---:|']
for name in ['upstream-short','retained-short','retained-8k','upstream-8k']:
 mem=json.loads((d/name/'memory.json').read_text()); g=list(csv.DictReader((d/name/'gpu.csv').open())); g=[{k.strip():v.strip() for k,v in x.items()} for x in g]
 peak=max(float(x['memory.used [MiB]'].split()[0]) for x in g)
 x=by[(name,1)]
 lines.append(f"| {name} | {x['ram_gib']:.2f} | {min(x['available_physical'] for x in mem)/2**30:.2f} | {peak:.0f} | {x['disk_reads']:.3f} |")
lines+=['', 'Our retained RAM backing is about 10 GiB larger; the GPU pool matches. This comparison includes that deliberate memory/performance tradeoff. VRAM is nearly full: 192K fits these controlled runs, but this is not foreground-use headroom qualification. CPU-expert and residency counts are recorded in summary.json; custom transfer counters are zero placeholders in the shared stock-compatible driver and must not be interpreted as zero traffic.', '', 'All calls completed with finite logits; repeated requests and process teardown succeeded. Numerical parity has not been qualified against new upstream, and no per-feature speed attribution is claimed. Current results support preserving the retained work while integrating upstream, not immediate PR sign-off. Raw logits stay local and ignored; manifests, timing CSVs, hardware logs and reproducible drivers are retained. No external publication.']
(d/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
