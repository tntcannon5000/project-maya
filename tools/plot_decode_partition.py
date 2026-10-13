"""Disjoint GPU-activity accounting; sums exactly to token wall time.

This is an observed time partition, not a counterfactual critical-path model.
CPU execution overlaps these lanes and must not be added to their durations.
"""
import argparse,bisect,collections,csv,json,re,sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_decode_timeline import category,write_csv

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
d=sqlite3.connect(out/'timeline.sqlite');strings=dict(d.execute('select id,value from StringIds'))
fixed=list(csv.DictReader((out/'fixed.csv').open()));lo=min(int(r['position']) for r in fixed)
tokens=[]
for s,e,text,tid in d.execute('select start,end,text,textId from NVTX_EVENTS'):
    m=re.match(r'token seq=(\d+) pos=(\d+)',text or strings.get(tid,''))
    if m and int(m[2])>=lo:tokens.append(dict(start=s,end=e,pos=int(m[2]),events=[]))
tokens.sort(key=lambda t:t['start']);starts=[t['start'] for t in tokens]
assert len(tokens)==len(fixed)
def attach(s,e,kind):
    i=max(0,bisect.bisect_right(starts,s)-1)
    while i<len(tokens) and tokens[i]['start']<e:
        t=tokens[i];left=max(t['start'],s);right=min(t['end'],e)
        if right>left:t['events'] += [(left,kind,1),(right,kind,-1)]
        i+=1
for s,e,n in d.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
    name=strings[n];cat=category(name,0)
    kind='fetch' if cat=='Expert fetch' else 'cpuwait' if cat=='CPU answer wait' else 'wait' if cat=='Host/disk wait' or 'moe_dma_wait' in name else 'compute'
    attach(s,e,kind)
for s,e in d.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY'):attach(s,e,'copy')
labels=['GPU computation only','Fetch + GPU computation','Fetch without GPU computation','CPU-answer wait (remaining)','Other GPU waits / copies','No recorded GPU activity']
colors=['#247da4','#6baf76','#e59639','#cc5a65','#9985b4','#cbd0d7']
rows=[]
for t,f in zip(tokens,fixed):
    assert t['pos']==int(f['position'])
    active=collections.Counter();totals=collections.Counter();previous=t['start']
    for time,kind,delta in sorted(t['events']+[(t['end'],'end',0)]):
        if active['compute']:label=labels[1] if active['fetch'] else labels[0]
        elif active['fetch']:label=labels[2]
        elif active['cpuwait']:label=labels[3]
        elif active['wait'] or active['copy']:label=labels[4]
        else:label=labels[5]
        totals[label]+=(time-previous)/1e6;previous=time;active[kind]+=delta
    wall=(t['end']-t['start'])/1e6
    assert abs(sum(totals.values())-wall)<1e-6
    rows.append(dict(repeat=int(f['repeat']),position=t['pos'],wall_ms=wall,**{k:totals[k] for k in labels}))
write_csv(out/'partition-tokens.csv',rows)
means={str(r):{k:sum(t[k] for t in rows if t['repeat']==r)/sum(t['repeat']==r for t in rows) for k in ['wall_ms']+labels} for r in sorted({t['repeat'] for t in rows})}
(out/'partition-means.json').write_text(json.dumps(means,indent=2))
fig,(ax,detail)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
for y,(rep,s) in enumerate(means.items()):
    left=0
    for k,c in zip(labels,colors):
        ax.barh(y,s[k],left=left,color=c,label=k if y==0 else None)
        if s[k]>.65:ax.text(left+s[k]/2,y,f'{s[k]:.2f}',ha='center',va='center',color='white' if k==labels[0] else 'black')
        left+=s[k]
    ax.text(left+.2,y,f'{left:.2f} ms',va='center')
ax.set_yticks(range(len(means)),['First request','Repeated request'][:len(means)]);ax.invert_yaxis()
ax.set_xlim(0,max(s['wall_ms'] for s in means.values())*1.16)
ax.set_xlabel('Mean ms/token — disjoint intervals, sum equals measured token loop')
warm=means[str(max(map(int,means)))];ys=list(range(len(labels)))
detail.barh(ys,[warm[k] for k in labels],color=colors)
detail.set_yticks(ys,labels);detail.invert_yaxis()
for y,k in enumerate(labels):detail.text(warm[k]+.07,y,f'{warm[k]:.3f} ms',va='center')
detail.set_xlim(0,max(warm[k] for k in labels)*1.22);detail.set_xlabel('Repeated-request mean ms/token')
fig.suptitle('Current Maya decode: where each token’s wall time goes\nOne CSV coding continuation, 128 tokens per request; Nsight diagnostic timings')
fig.savefig(out/'decode-partition.png',dpi=160);plt.close(fig)
print(json.dumps(means,indent=2))
