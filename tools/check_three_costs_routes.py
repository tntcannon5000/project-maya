"""Check every per-layer route mask in two fixed-input diagnostic captures."""
import bisect,csv,json,re,sqlite3
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def routes(p):
    d=sqlite3.connect(f'file:{(p/"timeline.sqlite").as_posix()}?mode=ro',uri=True)
    strings=dict(d.execute('select id,value from StringIds'))
    lo=min(int(x['position']) for x in csv.DictReader((p/'fixed.csv').open()))
    tokens=[];rs=[]
    for start,end,text,tid in d.execute('select start,end,text,textId from NVTX_EVENTS'):
        text=text or strings.get(tid,'')
        m=re.match(r'token seq=(\d+) pos=(\d+)',text)
        if m and int(m[2])>=lo:tokens.append((start,end,int(m[2])))
        m=re.fullmatch(r'route=(\d+) layer=(\d+) fetch=(\d+) cpu=(\d+) miss=(\d+) promo=(\d+) blob=(\d+)',text)
        if m:rs.append((start,tuple(map(int,m.groups()))))
    tokens.sort();starts=[t[0] for t in tokens];out=[[] for t in tokens]
    for start,row in rs:
        i=bisect.bisect_right(starts,start)-1
        if i>=0 and start<tokens[i][1]:out[i].append(row)
    result=[]
    for i,rows in enumerate(out):
        rows.sort()
        assert len(rows)==42
        result.extend((i,tokens[i][2],*row[1:]) for row in rows)
    return result
b=routes(R/'docs/evidence/maya-decode-latest/latest')
c=routes(R/'docs/evidence/maya-three-costs/retained-trace')
result=dict(routes=len(c),reference_routes=len(b),exact_ordered_layer_masks=b==c,
            fields=['token','position','layer','fetch_mask','cpu_mask','miss_mask','promo_mask','blob_bytes'],
            limitation='Route IDs are not present in these NVTX marks; exact logits complement this mask comparison.')
(R/'docs/evidence/maya-three-costs/route-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert b==c
