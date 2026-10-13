import json,statistics,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]; D=R/'docs/evidence/maya-upstream-117-c4'
data={b:json.loads((D/b/'results.json').read_text()) for b in ['upstream','retained']}
assert all(len(x)==2 for x in data.values())
rows=[]
for b,runs in data.items():
 for x in runs:
  q=list(x['requests'].values()); assert len(q)==4 and all(z['tokens']==512 for z in q)
  rows.append(dict(build=b,repeat=x['repeat'],aggregate_tps=x['aggregate_active_tps'],per_request_tps=[z['intertoken_tps'] for z in q],end_to_end_tps=x['end_to_end_tps'],active_seconds=x['all_four_active_ms']/1000,tokens=sum(z['tokens'] for z in q)))
(D/'summary.json').write_text(json.dumps(rows,indent=2))
s=['# C=4: upstream Maya v1.1.7 versus retained fork','','c=4, 32768-token capacity per slot; same GSQ-RCO 3.0bit Q4-attention pack and expert profile; no MTP, token speculation, K4V or KV streaming. Unprofiled server runs with identical four coding prompts, greedy sampling, up to 512 tokens each, two cohorts per build. Upstream runs first; retained second. No inference source changes. Allocated capacity is not occupied context: these are short prompts with 512 generated tokens.','', 'Aggregate decode TPS counts streamed tokens strictly after the last lane first emits through the earliest lane last emission: an interval when all four lanes are generating. Per-request TPS uses (tokens-1)/(last-first arrival). End-to-end TPS includes prompt processing from batch submission to final DONE. These are wall-clock server measures, not the previous c=1 fixed-input forward-call timer; do not infer concurrency scaling from those different fixtures. Outputs can differ between builds, and this is not a parity or quality test.','', '| Build | Cohort | Aggregate decode TPS | Per-request TPS range | End-to-end TPS | Four-lane interval seconds |','|---|---|---:|---:|---:|---:|']
for x in rows:
 s.append(f"| {x['build']} | {x['repeat']+1} | {x['aggregate_tps']:.2f} | {min(x['per_request_tps']):.2f}-{max(x['per_request_tps']):.2f} | {x['end_to_end_tps']:.2f} | {x['active_seconds']:.2f} |")
s+=['','## Capacity and qualification','']
for b in data:
 log=(D/b/'engine.log').read_text(); mem=json.loads((D/b/'memory.json').read_text())
 s.append(f"- {b}: minimum system RAM available {min(x['available_physical'] for x in mem)/2**30:.2f} GiB.")
 s.extend('- '+x for x in log.splitlines() if ('expert pool ' in x and 'slots/layer' in x) or 'RAM tier ' in x and 'pinned' in x or 'sequence states:' in x)
 s.extend('- '+x for x in log.splitlines() if 'graphs:' in x)
s+=['','Both returned READY multi=4, completed eight 512-token requests (4096 tokens per build), and exited normally. Prompts and environments are preserved in manifests, responses and timing in results.json, and wire/hardware logs remain local. Small differences need confirmation; this is one two-cohort comparison, not a broad concurrency qualification. Retained flags are configured but their presence alone is not proof each mechanism engages in the multi-request path.']
(D/'RESULTS.md').write_text('\n'.join(s)+'\n'); print('\n'.join(s))
