import csv,json,re,statistics,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[1]; root=r/'docs/evidence/maya-upstream-117-ab'
rows=[]
for p in root.glob('*/fixed.csv'):
 data=list(csv.DictReader(p.open())); log=(p.parent/'engine.log').read_text()
 for rep in [0,1]:
  ds=[x for x in data if int(x['repeat'])==rep]
  if not ds: continue
  ms=[float(x['ms']) for x in ds]
  pf=re.search(r'prefill repeat='+str(rep)+r' tokens=(\d+) ms=([\d.]+)',log)
  rows.append(dict(test=p.parent.name,repeat=rep,tokens=len(ds),tps=1000/statistics.mean(ms),mean_ms=statistics.mean(ms),p95_ms=sorted(ms)[int(.95*(len(ms)-1))],prefill_tokens=int(pf[1]) if pf else None,prefill_ms=float(pf[2]) if pf else None,**{k:statistics.mean(float(x[k]) for x in ds) for k in ['pool_slots','ram_slots','pool_gib','ram_gib','disk_reads','ram_fetches','cpu_experts']}))
(root/'summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
