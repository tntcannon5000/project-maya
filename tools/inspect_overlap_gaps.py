"""Attribute idle gaps to graph-internal, graph-boundary or uncaptured work."""
import bisect,collections,json,re,sqlite3
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'docs/evidence'
out={}
for name in ['maya-demand/trace-graphs','maya-residency-overlap/trace-overlap32']:
    db=sqlite3.connect(root/name/'timeline.sqlite');strings=dict(db.execute('select id,value from StringIds'))
    windows=[]
    for a,b,s,k in db.execute('select start,end,text,textId from NVTX_EVENTS'):
        m=re.match(r'token seq=(\d+) pos=(\d+)',s or strings.get(k,''))
        if m and int(m[2])>=30:windows.append((a,b))
    windows.sort();windows=windows[128:];starts=[a for a,b in windows]
    events=[[] for _ in windows]
    for a,b,g in db.execute('select start,end,graphId from CUPTI_ACTIVITY_KIND_KERNEL'):
        i=bisect.bisect_right(starts,a)-1
        if i>=0 and a<windows[i][1]:events[i].append((a,b,g or 0))
    for a,b in db.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY'):
        i=bisect.bisect_right(starts,a)-1
        if i>=0 and a<windows[i][1]:events[i].append((a,b,0))
    totals=collections.Counter()
    for (a,b),es in zip(windows,events):
        end=a;previous=0
        for s,e,g in sorted(es):
            if s>end:
                kind='inside_graph' if g and g==previous else 'graph_boundary' if g or previous else 'uncaptured'
                totals[kind]+=(s-end)/1e6
            if e>end:end=e;previous=g
        totals['token_tail']+=max(0,b-end)/1e6
    out[name]={k:v/len(windows) for k,v in totals.items()}
(root/'maya-residency-overlap/gap-attribution.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
