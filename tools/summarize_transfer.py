import csv,json
from pathlib import Path
import numpy as np
ROOT=Path('docs/evidence/maya-transfer')
rows=[]
for folder in sorted(ROOT.iterdir()):
    if not (folder/'fixed.csv').exists():continue
    data=list(csv.DictReader((folder/'fixed.csv').open()))
    if not data:continue
    for repeat in sorted({int(x['repeat']) for x in data}):
        r=[x for x in data if int(x['repeat'])==repeat]
        ms=np.array([float(x['ms']) for x in r])
        row=dict(experiment=folder.name,repeat=repeat,tokens=len(r),TPS=1000/float(ms.mean()),mean_ms=float(ms.mean()),p95_ms=float(np.percentile(ms,95)))
        for key in ('fetch_bytes','demotion_bytes','shadow_avoided_bytes','predicted_bytes','predicted_used_bytes','dma_bytes','hits','cpu_experts','promotions','disk_reads','pool_gib','ram_gib'):
            row[key]=float(np.mean([float(x[key]) for x in r])) if key in r[0] else None
        rows.append(row)
with (ROOT/'screens.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for row in rows: print(row['experiment'],row['repeat'],row['tokens'],round(row['TPS'],2), 'fetch MB',None if row['fetch_bytes'] is None else round(row['fetch_bytes']/1e6,1),'D2H MB',None if row['demotion_bytes'] is None else round(row['demotion_bytes']/1e6,1))
baseline=np.fromfile(ROOT/'fixed-baseline/fixed.logits.f32',dtype='float32').reshape(-1,154880)
parity=[]
for folder in sorted(ROOT.iterdir()):
    path=folder/'fixed.logits.f32'
    if not path.exists() or path.stat().st_size%(154880*4):continue
    b=np.fromfile(path,dtype='float32').reshape(-1,154880)
    if not len(b) or len(b)>len(baseline):continue
    a=baseline[:len(b)];delta=b.astype('float64')-a
    parity.append(dict(experiment=folder.name,positions=len(b),finite=bool(np.isfinite(b).all()),exact=bool(np.array_equal(a,b)),max_abs=float(np.abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))),top1_agreement=float(np.mean(a.argmax(1)==b.argmax(1)))))
(ROOT/'screen-logit-comparisons.json').write_text(json.dumps(parity,indent=2))
