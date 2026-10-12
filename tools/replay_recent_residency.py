"""Causal recent-frequency feasibility screen, no inference or future reads.

Same simplified CPU-count model as campaign two. Not a predictor of exact TPS:
all missing experts are admitted here, unlike Maya's GPU-only/spare admission.
"""
import collections, json, re, struct
import argparse
import numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old=root/'docs/evidence/maya-demand'
out=root/'docs/evidence/maya-residency-overlap';out.mkdir(exist_ok=True)
cap=json.loads((old/'route-capacity-study.json').read_text())['baseline_slots']
cap={int(k):v-3 for k,v in cap.items()}
meta={int(r[0]):int(r[4]) for line in Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya/native_experts.txt').read_text().splitlines() if not line.startswith('#') and (r:=line.split())}
prior=collections.defaultdict(list)
for l,e in struct.iter_unpack('<HH',(root/'data/expert-profile-glm.bin').read_bytes()[24:]):prior[l].append(e)
parser=argparse.ArgumentParser();parser.add_argument('--transition',action='store_true');args=parser.parse_args()
results=[]
for name in ['train-writing','retained-routes']:
    rows=[(int(r[0]),list(map(int,r[1:9]))) for line in (old/name/'routes.0').read_text().splitlines() if (r:=line.split())]
    for mode in ['aged','transition' if args.transition else 'recent32']:
        cache={l:set(prior[l][:cap[l]]) for l in cap};freq=collections.Counter();last={};hist={};epoch=collections.Counter();events=0;total=0
        transitions={l:np.zeros((288,288),dtype=np.float32) for l in cap};seen={l:np.zeros(288,dtype=np.float32) for l in cap};prev={}
        for l,ids in rows:
            epoch[l]+=1;t=epoch[l]
            missing=[e for e in ids if e not in cache[l]];n=len(missing)
            total+=(n-[0,1,1,2,2,3,3,4,4][n])*meta[l]
            for e in ids:
                key=(l,e);delta=t-last.get(key,0)
                hist[key]=(((hist.get(key,0)<<delta) if delta<32 else 0)|1)&0xffffffff
                last[key]=t;freq[key]+=1
            events+=len(ids)
            if events>=32768:
                freq=collections.Counter({k:v//2 for k,v in freq.items()});events=0
            if mode=='transition':
                if l in prev:
                    transitions[l][np.ix_(prev[l],ids)]+=1
                    seen[l][prev[l]]+=1
                prev[l]=ids
                # Smoothed conditional next-route reuse. Update uses only the
                # just-observed transition; score predicts the next route.
                prior_rate=np.array([freq[l,e] for e in range(288)],dtype=np.float32)
                prior_rate/=max(float(prior_rate.sum()/8),1)
                prediction=((transitions[l][ids]+16*prior_rate)/(seen[l][ids,None]+16)).mean(axis=0)
            def score(e):
                key=(l,e);delta=t-last.get(key,0)
                recent=(hist.get(key,0)>>delta).bit_count() if delta<32 else 0
                value=float(prediction[e]) if mode=='transition' else recent if mode=='recent32' and t>=32 else freq[key]
                return (value,freq[key],last.get(key,0))
            for e in missing:
                victims=cache[l]-set(ids)
                if len(cache[l])>=cap[l]:
                    if not victims:continue
                    cache[l].remove(min(victims,key=score))
                cache[l].add(e)
        results.append(dict(task=name,policy=mode,MB_per_token=total/(len(rows)/42)/1e6))
(out/('transition-replay.json' if args.transition else 'recent-replay.json')).write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
