"""Verify actual kernel variants, phase ordering, helper counters and prefix logits."""
import csv,json,re,sqlite3,bisect
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'docs/evidence/maya-missing-costs';D=P/'engagement'
db=sqlite3.connect(D/'timeline.sqlite');strings=dict(db.execute('select id,value from StringIds'))
fixed=list(csv.DictReader((D/'fixed.csv').open()));low=min(int(r['position']) for r in fixed)
tokens=[]
for s,e,t,i in db.execute('select start,end,text,textId from NVTX_EVENTS'):
    m=re.match(r'token seq=(\d+) pos=(\d+)',t or strings.get(i,''))
    if m and int(m[2])>=low:tokens.append((s,e))
tokens.sort();starts=[x[0] for x in tokens];ks=[[] for _ in tokens]
for s,e,stream,n in db.execute('select start,end,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
    i=bisect.bisect_right(starts,s)-1
    if i>=0 and s<tokens[i][1]:ks[i].append(dict(start=s,end=e,stream=stream,name=strings[n]))
counts=dict(route=0,paired_prefix4=0,component4=0,late_phase_publications=0,gu=0,down=0)
for rows in ks:
    routes=[r for r in rows if 'moe_route_kernel' in r['name']];assert len(routes)==42
    for i,route in enumerate(routes):
        end=routes[i+1]['start'] if i+1<len(routes) else float('inf')
        layer=[r for r in rows if route['start']<=r['start']<end]
        prefix=[r for r in layer if 'moe_gu_band_fetch_kernel' in r['name']]
        down=[r for r in layer if 'moe_component_kernel' in r['name']]
        phase=[r for r in layer if 'moe_prefix_ready_kernel' in r['name']]
        gu=[r for r in layer if 'moe_gate_up_kernel' in r['name']]
        dc=[r for r in layer if 'moe_down_kernel' in r['name']]
        assert len(prefix)==len(down)==len(gu)==len(dc)==2 and len(phase)==1
        assert all('<(int)4>' in r['name'] or '<4>' in r['name'] for r in prefix+down),[r['name'] for r in prefix+down]
        assert phase[0]['start']>=max(r['end'] for r in prefix+down)
        assert phase[0]['stream']==down[-1]['stream']
        for key,n in [('route',1),('paired_prefix4',2),('component4',2),('late_phase_publications',1),('gu',2),('down',2)]:counts[key]+=n
log=(P/'engagement-counts/engine.log').read_text(errors='replace')
pair=re.search(r'maya Q2 pair: GU rows (\d+), down paired rows (\d+), down fallback tail rows (\d+)',log)
assert pair and all(int(pair[i])>0 for i in (1,2))
coarse=re.search(r'maya CPU single coarse: batches=(\d+)',log);assert coarse and int(coarse[1])>0
# A16-token second request has different residency than the128-token fixture.
# Only first-request prefix logits are a controlled comparison here.
ref=np.fromfile(P/'baseline/fixed.logits.f32',dtype=np.float32)
got=np.fromfile(D/'fixed.logits.f32',dtype=np.float32)
vocab=ref.size//256;assert got.size==len(fixed)*vocab
prefix_n=16*vocab
assert np.array_equal(ref[:prefix_n],got[:prefix_n]) and np.isfinite(got).all()
result=dict(tokens=len(tokens),actual_launches=counts,
    Q2_rows_including_setup=dict(GU=int(pair[1]),down_pair=int(pair[2]),down_tail=int(pair[3])),
    coarse_batches_including_setup=int(coarse[1]),
    full_fetch_before_phase_release=True,prefix_logit_count=prefix_n,prefix_exact=True,
    limitation='Second request has different cache history from full128-token reference; no second-request numerical claim. Diagnostic flags include counters and profiler, not a TPS qualification.')
(P/'engagement-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
