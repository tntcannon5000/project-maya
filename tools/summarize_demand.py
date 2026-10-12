import csv,json,statistics,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'docs/evidence/maya-demand'
out=[];parity=[]
reference=np.fromfile(ROOT/'baseline/fixed.logits.f32',dtype='float32').reshape(-1,154880)
route_ref=[tuple(map(int,x.split()[:9])) for x in (ROOT/'retained-routes/routes.0').read_text().splitlines()]
for p in sorted(ROOT.iterdir()):
    if not(p/'fixed.csv').exists():continue
    rows=list(csv.DictReader((p/'fixed.csv').open()))
    if not rows:continue
    manifest=json.loads((p/'manifest.json').read_text())
    for rep in sorted({r['repeat'] for r in rows}):
        r=[x for x in rows if x['repeat']==rep];ms=[float(x['ms']) for x in r]
        row=dict(name=p.name,repeat=int(rep),n=len(r),TPS=1000/statistics.mean(ms),ms=statistics.mean(ms),p95=float(np.percentile(ms,95)))
        for k in ['fetch_bytes','demotion_bytes','predicted_bytes','predicted_used_bytes','hits','cpu_experts','promotions','disk_reads','pool_gib','ram_gib']:row[k]=statistics.mean(float(x[k]) for x in r)
        out.append(row)
    if (p/'fixed.logits.f32').exists() and manifest['tokens_sha256']==json.loads((ROOT/'baseline/manifest.json').read_text())['tokens_sha256']:
        z=np.fromfile(p/'fixed.logits.f32',dtype='float32').reshape(-1,154880)
        if len(z)<=len(reference):
            ref=reference[:len(z)];delta=z.astype('float64')-ref
            q=dict(name=p.name,positions=len(z),finite=bool(np.isfinite(z).all()),exact=bool(np.array_equal(z,ref)),max_abs=float(abs(delta).max()),rms=float(np.sqrt((delta*delta).mean())),top1=float((ref.argmax(1)==z.argmax(1)).mean()))
            if(p/'routes.0').exists():
                rr=[tuple(map(int(x) for x in line.split()[:9])) for line in (p/'routes.0').read_text().splitlines()]
                if len(rr)==len(route_ref):q['route_top8_exact_fraction']=sum(a==b for a,b in zip(rr,route_ref))/len(rr)
            parity.append(q)
with (ROOT/'screens.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(ROOT/'parity.json').write_text(json.dumps(parity,indent=2))
for r in out:print(r['name'],r['repeat'],round(r['TPS'],2),round(r['fetch_bytes']/1e6,1),'MB demand',round(r['predicted_bytes']/1e6,1),'MB predicted')
