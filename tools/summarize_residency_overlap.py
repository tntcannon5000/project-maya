"""Offline audit of completed focused experiments; no model execution."""
import csv,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'docs/evidence/maya-residency-overlap'
rows=[];parity=[]
baseline=ROOT/'baseline/fixed.logits.f32'
for p in sorted(ROOT.iterdir()):
    if not (p/'fixed.csv').exists():continue
    data=list(csv.DictReader((p/'fixed.csv').open()))
    if not data:continue
    for repeat in sorted({r['repeat'] for r in data}):
        group=[r for r in data if r['repeat']==repeat]
        result=dict(name=p.name,repeat=int(repeat),tokens=len(group))
        result.update({k:statistics.mean(float(r[k]) for r in group) for k in ['ms','fetch_bytes','demotion_bytes','hits','cpu_experts','promotions','pool_gib','ram_gib']})
        result['TPS']=1000/result['ms'];rows.append(result)
    manifest=json.loads((p/'manifest.json').read_text())
    reference=baseline
    if manifest['env'].get('STRATA_GLM_CPU_PLAN')=='000000000':
        reference=ROOT.parent/'maya-transfer/gpu-clean-cold/fixed.logits.f32'
    elif manifest['tokens_sha256']!=json.loads((ROOT/'baseline/manifest.json').read_text())['tokens_sha256']:
        continue
    ref_manifest=json.loads((reference.parent/'manifest.json').read_text())
    assert manifest['tokens_sha256']==ref_manifest['tokens_sha256']
    a=np.memmap(reference,dtype='float32',mode='r');b=np.memmap(p/'fixed.logits.f32',dtype='float32',mode='r')
    count=min(len(a),len(b));maximum=0.;sq=0.;equal=True;finite=True;top=0;positions=0
    for start in range(0,count,154880):
        x=a[start:start+154880];y=b[start:start+154880]
        finite &= bool(np.isfinite(y).all());equal &= bool(np.array_equal(x,y))
        delta=x.astype('float64')-y;maximum=max(maximum,float(abs(delta).max()));sq+=float(delta@delta)
        top+=int(x.argmax()==y.argmax());positions+=1
    parity.append(dict(name=p.name,reference=str(reference),positions=positions,logits=count,finite=finite,exact=equal,max_abs=maximum,rms=(sq/count)**.5,top1=top/positions))
    del a,b
(ROOT/'measurements.json').write_text(json.dumps(rows,indent=2))
(ROOT/'parity.json').write_text(json.dumps(parity,indent=2))
for r in rows:print(r['name'],r['repeat'],round(r['TPS'],2),round(r['fetch_bytes']/1e6,2),'MB')
for p in parity:print('parity',p['name'],p['exact'],p['max_abs'],p['top1'])
