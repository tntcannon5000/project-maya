"""Held-out residency allocation and explicit offline capacity bounds.

Replay is NOT engine TPS: no CPU arithmetic route changes or transfer scheduling.
Oracle has perfect future information and no cost for changing ownership.
"""
import collections,json,re,struct,heapq,sqlite3,statistics,bisect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/evidence/maya-demand'
meta={int(r[0]):int(r[4]) for line in Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya/native_experts.txt').read_text().splitlines() if not line.startswith('#') and (r:=line.split())}
def routes(name):
    return [(int(r[0]),list(map(int,r[1:9]))) for line in (OUT/name/'routes.0').read_text().splitlines() if (r:=line.split())]
training=routes('train-writing')+routes('train-web');test=routes('retained-routes')
counts={l:collections.Counter() for l in meta}
for l,ids in training:counts[l].update(ids)
scores={l:sorted((counts[l][e] for e in range(288)),reverse=True) for l in meta}
slots={int(a):int(b) for a,b in re.findall(r'(\d+):(\d+)',next(x for x in (OUT/'baseline/engine.log').read_text().splitlines() if 'slots per layer' in x))}
budget=sum(meta[l]*slots[l] for l in slots);alloc={l:49 for l in slots};used=sum(meta[l]*alloc[l] for l in alloc)
# Marginal avoided transfer bytes per occupied byte: frequency * blob / slot.
# All routed experts within a layer use equal blob sizes; do not reward cheap
# experts twice or use raw hit counts as if quant types had equal cost.
while True:
    choices=[l for l in alloc if alloc[l]<291 and used+meta[l]<=budget]
    if not choices:break
    l=max(choices,key=lambda l:scores[l][alloc[l]-3] if alloc[l]-3<288 else 0)
    alloc[l]+=1;used+=meta[l]
(OUT/'trained-slots.txt').write_text(''.join(f'{l} {n}\n' for l,n in sorted(alloc.items())))
ranked=sorted(((l,e) for l in range(46) for e in range(288)),key=lambda le:counts.get(le[0],{}).get(le[1],0),reverse=True)
(OUT/'trained-profile.bin').write_bytes(struct.pack('<4sIIIII',b'STRP',1,46,288,len(ranked),len(ranked))+b''.join(struct.pack('<HH',l,e) for l,e in ranked))
old=(ROOT/'data/expert-profile-glm.bin').read_bytes();prior=collections.defaultdict(list)
for l,e in struct.iter_unpack('<HH',old[24:]):prior[l].append(e)

def replay(policy,capacity,seed,rows):
    cache={l:set(seed[l][:capacity[l]-3]) for l in capacity};freq={l:collections.Counter() for l in capacity};last={l:{} for l in capacity}
    future=collections.defaultdict(collections.deque)
    for i,(l,ids) in enumerate(rows):
        for e in ids:future[l,e].append(i)
    values=[];reused=0;ever=set()
    for i,(l,ids) in enumerate(rows):
        for e in ids:future[l,e].popleft()
        missing=[e for e in ids if e not in cache[l]];n=len(missing)
        # Same split counts, not necessarily the same expert IDs assigned CPU.
        take=[0,1,1,2,2,3,3,4,4][n]
        values.append(dict(layer=l,miss=n,gpu_fetch=n-take,fetch_bytes=(n-take)*meta[l]))
        for e in ids:freq[l][e]+=1;last[l][e]=i
        for e in missing:
            reused+=(l,e) in ever;ever.add((l,e))
            if len(cache[l])>=capacity[l]-3:
                victims=cache[l]-set(ids)
                if not victims:continue
                if policy=='oracle':v=max(victims,key=lambda x:future[l,x][0] if future[l,x] else float('inf'))
                elif policy=='lru':v=min(victims,key=lambda x:last[l].get(x,-1))
                else:v=min(victims,key=lambda x:(freq[l][x],last[l].get(x,-1)))
                cache[l].remove(v)
            cache[l].add(e)
    # 42 routed layers per token, including prompt in this descriptive replay.
    return dict(mean_fetch_MB=sum(v['fetch_bytes'] for v in values)/(len(values)/42)/1e6,
                misses_per_token=sum(v['miss'] for v in values)/(len(values)/42),refetches=reused,routes=len(rows))
trained={l:sorted(range(288),key=lambda e:counts[l][e],reverse=True) for l in meta}
result=dict(training=['writing','web'],holdout='CSV',pool_bytes=budget,new_pool_bytes=used,baseline_slots=slots,trained_slots=alloc,
    replay={f'{name}-{policy}':replay(policy,cap,seed,test) for name,cap,seed in [('original',slots,prior),('trained',alloc,trained)] for policy in ['lfu','lru','oracle']})
(OUT/'route-capacity-study.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['replay'],indent=2))

# D6: derive a bounded per-layer plan from the already captured retained CSV.
# GPU CPU-wait order follows routed layers 3..44. Test writing/web separately
# before retention: a calibrated CSV improvement alone may be overfitting.
db=sqlite3.connect(ROOT/'docs/evidence/maya-transfer/trace-retained/timeline.sqlite');strings=dict(db.execute('select id,value from StringIds'))
wait_ids=[i for i,s in strings.items() if 'moe_cpu_wait_kernel' in s]
windows=[]
for a,b,text,i in db.execute('select start,end,text,textId from NVTX_EVENTS'):
    m=re.match(r'token seq=(\d+) pos=(\d+)',text or strings.get(i,''))
    if m and int(m[2])>=30:windows.append((a,b))
windows.sort();wait=collections.defaultdict(list)
for a,b in windows[128:]:
    ks=list(db.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? and demangledName in ('+','.join('?' for _ in wait_ids)+') order by start',(a,b,*wait_ids)))
    assert len(ks)==42,len(ks)
    for l,(s,e) in zip(range(3,45),ks):wait[l].append((e-s)/1e6)
cpu_cost=collections.defaultdict(list);pull_cost=collections.defaultdict(list);compute=collections.defaultdict(list)
starts=[a for a,b in windows]
marks=collections.defaultdict(dict);cpu_ranges=collections.defaultdict(dict)
for a,b,text,i in db.execute('select start,end,text,textId from NVTX_EVENTS'):
    txt=text or strings.get(i,'');ti=bisect.bisect_right(starts,a)-1
    if ti<128 or ti>=len(windows) or a>=windows[ti][1]:continue
    m=re.match(r'route=\d+ layer=(\d+) fetch=(\d+) cpu=(\d+) miss=(\d+)',txt)
    if m:marks[ti][int(m[1])]=(int(m[2]).bit_count(),int(m[3]).bit_count())
    m=re.match(r'CPU experts layer=(\d+)',txt)
    if m and b:cpu_ranges[ti][int(m[1])]=(b-a)/1e6
for ti,(a,b) in enumerate(windows):
    if ti<128:continue
    layer=3;fetch=0.;arith=0.
    for start,end,name in db.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start',(a,b)):
        n=strings[name];ms=(end-start)/1e6
        if 'moe_fetch_kernel' in n:fetch=ms
        if 'moe_gate_up_kernel' in n or 'moe_down_kernel' in n:arith+=ms
        if 'moe_cpu_wait_kernel' in n:
            nf,nc=marks[ti].get(layer,(0,0))
            if nf:pull_cost[layer].append(fetch/nf)
            if nc and layer in cpu_ranges[ti]:cpu_cost[layer].append(cpu_ranges[ti][layer]/nc)
            compute[layer].append(arith);layer+=1;arith=0.
plans={};measure={};costs={}
for l in sorted(meta):
    mean=statistics.mean(wait[l]);measure[l]=mean
    # Minimise the slower lane, not the wait kernel alone. A CPU wait can be
    # cheaper than the extra transfer needed to remove it. Medians reduce
    # outlier leverage; hold-out requests must still qualify these estimates.
    c=statistics.median(cpu_cost[l]);p=statistics.median(pull_cost[l]);g=statistics.median(compute[l])
    costs[l]=dict(cpu_ms_per_expert=c,pull_ms_per_expert=p,compute_ms=g)
    plans[l]=''.join(str(min(range(f+1),key=lambda k:max(c*k,p*(f-k)+g))) for f in range(9))
(OUT/'cpu-wait-plan.txt').write_text(''.join(f'{l} {v}\n' for l,v in plans.items()))
(OUT/'cpu-wait-calibration.json').write_text(json.dumps(dict(wait_ms=measure,plans=plans,costs=costs,training='retained CSV repeat; not held-out'),indent=2))
