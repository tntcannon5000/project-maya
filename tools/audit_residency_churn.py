"""Observed residency loss and bounded, causal admission screening.

No inference is launched. Admission is known only in new four-mask traces.
Replay is approximate: no asynchronous ownership, prefill scheduling or exact
device count aging. Its error against actual masks is reported, not hidden.
"""
import argparse, collections, csv, json, struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = [0, 1, 1, 2, 2, 3, 3, 4, 4]

def load(path):
    rows = []
    token = -1
    previous = 999
    for line in path.read_text().splitlines():
        fields = line.split('|')[0].split()
        l = int(fields[0])
        if l <= previous: token += 1
        previous = l
        rows.append(dict(t=token, l=l, ids=list(map(int, fields[1:9])),
                         fetch=int(fields[9]), cpu=int(fields[11]), off=int(fields[9]) | int(fields[10]) | int(fields[11]),
                         promo=int(fields[12]) if len(fields)>12 else None))
    return rows

def observed(rows, sizes, phases):
    previous = {}
    pending = {}
    stats = collections.defaultdict(collections.Counter)
    for r in rows:
        s = stats[phases[r['t']]]
        s['routes'] += 1
        for i,e in enumerate(r['ids']):
            key = r['l'],e
            off = bool(r['off'] >> i & 1)
            fetch = bool(r['fetch'] >> i & 1)
            s['uses'] += 1
            s['hits'] += not off
            s['fetch_bytes'] += fetch*sizes[r['l']]
            s['cpu_weight_bytes'] += bool(r['cpu'] >> i & 1)*sizes[r['l']]
            if key in previous:
                t, resident = previous[key]
                gap = r['t']-t
                if resident and off:
                    s['observed_residency_losses'] += 1
                    s['loss_fetch_bytes'] += fetch*sizes[r['l']]
                    for n in (1,4,16):
                        if gap <= n:
                            s[f'loss_within_{n}'] += 1
                            s[f'loss_fetch_bytes_within_{n}'] += fetch*sizes[r['l']]
            if key in pending:
                start, phase = pending.pop(key)
                p = stats[phase]
                p['admissions_with_next_use'] += 1
                p['admissions_surviving_next_use'] += not off
                p['admissions_lost_before_next_use'] += off
            admitted = r['promo'] is not None and bool(r['promo'] >> i & 1)
            if admitted:
                s['admissions'] += 1
                pending[key] = r['t'], phases[r['t']]
            previous[key] = r['t'], not off or admitted
    for _,phase in pending.values(): stats[phase]['admissions_right_censored'] += 1
    for s in stats.values():
        tokens=s['routes']/42
        s['tokens']=tokens
        s['fetch_MB_per_token']=s['fetch_bytes']/tokens/1e6
        s['cpu_weight_MB_per_token']=s['cpu_weight_bytes']/tokens/1e6
        s['hit_percent']=100*s['hits']/s['uses']
        for n in (1,4,16):
            s[f'loss_fetch_MB_per_token_within_{n}']=s[f'loss_fetch_bytes_within_{n}']/tokens/1e6
    return dict(stats)

def replay(rows, sizes, caps, prior, mode, phases):
    cache={l:set(prior[l][:caps[l]-3]) for l in caps}
    freq=np.zeros((46,288),dtype=np.float64)
    trans=np.zeros((46,288,288),dtype=np.float64)
    seen=np.zeros((46,288),dtype=np.float64)
    prev={}; last={}; stats=collections.defaultdict(collections.Counter); events=0
    for r in rows:
        l,ids,t=r['l'],r['ids'],r['t']
        s=stats[phases[t]]; s['routes']+=1
        missing=[(i,e) for i,e in enumerate(ids) if e not in cache[l]]
        for e in ids: freq[l,e]+=1; last[l,e]=t
        cpu={e for i,e in sorted(missing,key=lambda ie:(freq[l,ie[1]],-ie[0]))[:PLAN[len(missing)]]}
        fetched=[e for _,e in missing if e not in cpu]
        s['fetch_bytes']+=len(fetched)*sizes[l]
        s['cpu_weight_bytes']+=len(cpu)*sizes[l]
        s['hits']+=8-len(missing)
        for i,e in enumerate(ids): s['mask_disagreements']+=((e not in cache[l]) != bool(r['off']>>i&1))
        events+=8
        if events>=32768: freq=np.floor(freq/2);events=0
        if l in prev:
            for e in prev[l]:
                if seen[l,e]>=2048: seen[l,e]*=.5; trans[l,e]*=.5
                seen[l,e]+=1;trans[l,e,ids]+=1
        prev[l]=ids
        scores=((trans[l,ids]+16*freq[l]*(8/max(freq[l].sum(),1)))/(seen[l,ids,None]+16)).mean(axis=0)
        def rank(e): return (scores[e],freq[l,e],last.get((l,e),-1),e)
        # Fetched experts only; at most three admissions into spare slots.
        # Admission margin is predetermined, not fitted to either trace.
        for e in fetched[:3]:
            victim=min(cache[l],key=rank)
            if mode=='margin' and freq[l,e] < 1.5*freq[l,victim]:
                s['rejected_admissions']+=1
                continue
            cache[l].add(e); s['admissions']+=1
        while len(cache[l])>caps[l]-3: cache[l].remove(min(cache[l],key=rank))
    for s in stats.values():
        n=s['routes']/42
        s['fetch_MB_per_token']=s['fetch_bytes']/n/1e6
        s['cpu_weight_MB_per_token']=s['cpu_weight_bytes']/n/1e6
        s['mask_disagreement_percent']=100*s['mask_disagreements']/(s['routes']*8)
    return dict(stats)

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args()
    out=ROOT/'docs/evidence/maya-churn';out.mkdir(exist_ok=True)
    meta=Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya/native_experts.txt')
    sizes={int(f[0]):int(f[4]) for line in meta.read_text().splitlines() if not line.startswith('#') and (f:=line.split())}
    caps={int(k):v for k,v in json.loads((ROOT/'docs/evidence/maya-demand/route-capacity-study.json').read_text())['baseline_slots'].items()}
    prior=collections.defaultdict(list)
    for l,e in struct.iter_unpack('<HH',(ROOT/'data/expert-profile-glm.bin').read_bytes()[24:]):prior[l].append(e)
    results={}
    for name in a.runs:
        folder=ROOT/name; rows=load(folder/'routes.0')
        manifest=json.loads((folder/'manifest.json').read_text())
        prompt=int(manifest['command'][-3]);repeats=int(manifest['command'][-1])
        nt=len((folder/'tokens.txt').read_text().split())
        assert len(rows)==nt*repeats*42, (len(rows),nt,repeats)
        phases={t:f"r{t//nt}-"+('prompt' if t%nt<prompt else 'decode') for t in range(nt*repeats)}
        results[folder.name]={'observed':observed(rows,sizes,phases),
            'replay_transition':replay(rows,sizes,caps,prior,'base',phases),
            'replay_admission_margin':replay(rows,sizes,caps,prior,'margin',phases)}
    (out/'audit.json').write_text(json.dumps(results,indent=2))
    for name,res in results.items():
        for phase,s in res['observed'].items():
            if 'decode' not in phase:continue
            b=res['replay_transition'][phase];m=res['replay_admission_margin'][phase]
            print(name,phase,'observed MB/t',round(s['fetch_MB_per_token'],2),'hit%',round(s['hit_percent'],1),
                  'loss <=1/4/16 MB/t',[round(s[f'loss_fetch_MB_per_token_within_{n}'],2) for n in (1,4,16)],
                  'replay',round(b['fetch_MB_per_token'],2),'margin',round(m['fetch_MB_per_token'],2),
                  'mask error%',round(b['mask_disagreement_percent'],1))

if __name__=='__main__':main()
