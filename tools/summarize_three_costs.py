"""Summarize one experiment and compare its logits against a named reference."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'docs/evidence/maya-three-costs'
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--reference',default='baseline');a=p.parse_args()
out=ROOT/a.name;rows=list(csv.DictReader((out/'fixed.csv').open()))
result={}
for rep in sorted({int(r['repeat']) for r in rows}):
    rs=[r for r in rows if int(r['repeat'])==rep];ms=[float(r['ms']) for r in rs]
    result[str(rep)]=dict(tokens=len(rs),TPS=1000/statistics.mean(ms),mean_ms=statistics.mean(ms),
      p95_ms=float(np.percentile(ms,95)),**{k:statistics.mean(float(r[k]) for r in rs) for k in
      ('fetch_bytes','demotion_bytes','hits','cpu_experts','promotions','pool_slots','ram_slots','pool_gib','ram_gib')})
reference=ROOT/a.reference/'fixed.logits.f32'
if reference.exists():
    x=np.fromfile(out/'fixed.logits.f32',dtype=np.float32);y=np.fromfile(reference,dtype=np.float32)
    result['parity']=dict(reference=a.reference,count=x.size,finite=bool(np.isfinite(x).all()),same_size=x.size==y.size)
    if x.size==y.size:
        delta=np.abs(x-y)
        result['parity'].update(exact=bool(np.array_equal(x,y)),max_abs=float(delta.max()),mean_abs=float(delta.mean()),changed=int(np.count_nonzero(delta)))
        tokens=len(rows)
        if x.size%tokens==0:
            result['parity']['greedy_agreement']=float(np.mean(x.reshape(tokens,-1).argmax(1)==y.reshape(tokens,-1).argmax(1)))
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
