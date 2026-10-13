"""Per-layer GPU readiness and transport accounting from CUDA/NVTX evidence.

No inference or engine modifications. Physical PCIe occupancy cannot be derived
from kernel duration; bandwidth is effective useful-payload throughput only.
"""
import argparse, bisect, collections, csv, json, re, sqlite3, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_decode_timeline import category, duration, overlap, write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
    d=sqlite3.connect(p/'timeline.sqlite');d.row_factory=sqlite3.Row
    strings=dict(d.execute('select id,value from StringIds'))
    scored=list(csv.DictReader((p/'fixed.csv').open()))
    manifest=json.loads((p/'manifest.json').read_text())
    native=Path(manifest['weights']['path']).parent/'pack-maya/native_experts.txt'
    text=native.read_text();layout={}
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        il,gu,dt,off,blob,gate,up,down=map(int,line.split()[:8])
        # For this model's K-block formats both matrix dimensions are multiples
        # of 256, and gate/up/down contain the same number of elements. Tensor
        # file offsets are NOT component lengths (intervening tensors/padding).
        sizes={10:84,11:110,12:144}
        assert gu in sizes and dt in sizes,'Unsupported quant layout'
        units,rem=divmod(blob,2*sizes[gu]+sizes[dt]);assert rem==0
        layout[il]=dict(blob=blob,prefix=2*units*sizes[gu],gu=gu,down=dt)
    lo=min(int(r['position']) for r in scored);tokens=[];routes=[]
    for r in d.execute('select start,end,text,textId from NVTX_EVENTS'):
        s=r['text'] or strings.get(r['textId'],'')
        m=re.match(r'token seq=(\d+) pos=(\d+)',s)
        if m and int(m[2])>=lo:tokens.append(dict(start=r['start'],end=r['end'],pos=int(m[2]),kernels=[],routes=[]))
        m=re.match(r'route=(\d+) layer=(\d+) fetch=(\d+) cpu=(\d+) miss=(\d+) promo=(\d+) blob=(\d+)',s)
        if m:routes.append(dict(start=r['start'],route=int(m[1]),layer=int(m[2]),bytes=int(m[3]).bit_count()*int(m[7]),miss=int(m[5])))
    tokens.sort(key=lambda t:t['start']);starts=[t['start'] for t in tokens]
    assert len(tokens)==len(scored)
    def owner(s):
        i=bisect.bisect_right(starts,s)-1
        return tokens[i] if i>=0 and s<tokens[i]['end'] else None
    for r in routes:
        t=owner(r['start'])
        if t is not None:t['routes'].append(r)
    for r in d.execute('select start,end,streamId,gridX,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
        t=owner(r['start'])
        if t is not None:
            assert r['end']<=t['end'], 'Kernel crosses token boundary; extend clipping before use'
            t['kernels'].append(dict(r,name=strings[r['demangledName']]))
    layer_rows=[];stage_rows=[];kernel_rows=[];token_rows=[];transfer_rows=[]
    for t,f in zip(tokens,scored):
        assert t['pos']==int(f['position'])
        rep=int(f['repeat']);ks=t['kernels'];rs=sorted(t['routes'],key=lambda r:r['route'])
        routing=[k for k in ks if 'moe_route_kernel' in k['name']]
        assert len(rs)==len(routing)==42,(len(rs),len(routing))
        assert [r['layer'] for r in rs]==list(range(3,45)),[r['layer'] for r in rs]
        assert sum(r['bytes'] for r in rs)==int(f['fetch_bytes']), 'NVTX bytes do not reconcile'
        # Known captured topology: prefix first, then one/two ordered down
        # copies. Grid size is NOT a phase identifier: prefix may also use32.
        for j,route in enumerate(routing):
            end=routing[j+1]['start'] if j+1<len(routing) else t['end']
            fs=[k for k in ks if route['start']<=k['start']<end and any(s in k['name'] for s in ('moe_fetch_kernel','moe_component_kernel','moe_gu_band_fetch_kernel'))]
            assert len(fs) in (1,2,3,4)
            paired=any('moe_gu_band_fetch_kernel' in k['name'] for k in fs)
            for i,k in enumerate(fs):k['part']='prefix' if 'moe_gu_band_fetch_kernel' in k['name'] else 'down' if paired else 'whole' if len(fs)==1 else 'prefix' if i==0 else 'down'
        groups=collections.defaultdict(list);names=collections.defaultdict(list)
        for k in ks:
            name=k['name'];cat=category(name,0)
            if cat=='Dense projections':cat='Dense GEMV / projections / output head'
            elif cat=='Routing':cat='Routing / shared SwiGLU / publication'
            elif cat=='Expert gate/up':cat='Expert gate/up + shared down'
            if 'moe_component' in name or 'moe_gu_band_fetch' in name:cat='Fetch: down component' if k['part']=='down' else 'Fetch: gate/up prefix'
            elif 'moe_fetch' in name:cat='Fetch: whole expert'
            elif 'quantize' in name or 'swiglu_q8' in name:cat='Activation quantization / SwiGLU'
            groups[cat].append((k['start'],k['end']));names[name].append((k['start'],k['end']))
        for cat,ivs in groups.items():stage_rows.append(dict(repeat=rep,position=t['pos'],stage=cat,ms=duration(ivs)/1e6))
        for name,ivs in names.items():kernel_rows.append(dict(repeat=rep,position=t['pos'],kernel=name,count=len(ivs),ms=duration(ivs)/1e6))
        for j,(r,route) in enumerate(zip(rs,routing)):
            end=routing[j+1]['start'] if j+1<len(routing) else t['end']
            es=[k for k in ks if route['start']<=k['start']<end]
            fetch=[k for k in es if any(s in k['name'] for s in ('moe_fetch_kernel','moe_component_kernel','moe_gu_band_fetch_kernel'))]
            gus=[k for k in es if 'moe_gate_up_kernel' in k['name']];downs=[k for k in es if 'moe_down_kernel' in k['name']]
            gu=gus[-1];down=downs[0]
            assert len(fetch) in (1,2,3,4) and len(gus) in (1,2) and len(downs) in (1,2)
            prefixes=[k for k in fetch if k['part'] in ('prefix','whole')]
            down_fetches=[k for k in fetch if k['part']=='down']
            prefix=prefixes[-1]
            ready=max(k['end'] for k in fetch)
            lay=layout[r['layer']];count=r['bytes']//lay['blob']
            assert r['bytes']%lay['blob']==0
            for k in fetch:
                part=k['part']
                size=lay['blob'] if part=='whole' else lay['prefix']//len(prefixes) if part=='prefix' else (lay['blob']-lay['prefix'])//len(down_fetches)
                transfer_rows.append(dict(repeat=rep,position=t['pos'],layer=r['layer'],part=part,blocks=k['gridX'],bytes=count*size,ms=(k['end']-k['start'])/1e6))
            iv=[(k['start'],k['end']) for k in fetch]
            math=[(k['start'],k['end']) for k in gus+downs]
            layer_rows.append(dict(repeat=rep,position=t['pos'],layer=r['layer'],bytes=r['bytes'],disk_mask=r['miss'],
                fetch_ms=duration(iv)/1e6,route_to_fetch_ms=(min(k['start'] for k in fetch)-route['end'])/1e6,
                prefix_ready_to_gu_us=(gu['start']-prefix['end'])/1e3,
                down_dependencies_to_start_us=(down['start']-max(gu['end'],down_fetches[0]['end'] if down_fetches else ready))/1e3,
                down_copy_remaining_after_gu_ms=max(0,ready-gu['end'])/1e6,
                overlap_ms=overlap(iv,math)/1e6,component=len(fetch)>1,
                fetch_start_ms=(min(k['start'] for k in fetch)-t['start'])/1e6,
                fetch_end_ms=(ready-t['start'])/1e6,gu_start_ms=(gu['start']-t['start'])/1e6,
                gu_end_ms=(gu['end']-t['start'])/1e6,down_start_ms=(down['start']-t['start'])/1e6,
                down_end_ms=(downs[-1]['end']-t['start'])/1e6,
                fetch_intervals=[((k['start']-t['start'])/1e6,(k['end']-k['start'])/1e6) for k in fetch],
                gu_intervals=[((k['start']-t['start'])/1e6,(k['end']-k['start'])/1e6) for k in gus],
                down_intervals=[((k['start']-t['start'])/1e6,(k['end']-k['start'])/1e6) for k in downs]))
        token_rows.append(dict(repeat=rep,position=t['pos'],wall_ms=(t['end']-t['start'])/1e6,bytes=int(f['fetch_bytes'])))
    write_csv(p/'deep-layers.csv',layer_rows);write_csv(p/'deep-stages.csv',stage_rows);write_csv(p/'deep-kernels.csv',kernel_rows);write_csv(p/'deep-transfers.csv',transfer_rows)
    summary={}
    for rep in sorted({r['repeat'] for r in token_rows}):
        ts=[r for r in token_rows if r['repeat']==rep];ls=[r for r in layer_rows if r['repeat']==rep];n=len(ts)
        totals=collections.Counter()
        for r in stage_rows:
            if r['repeat']==rep:totals[r['stage']]+=r['ms']/n
        b=sum(r['bytes'] for r in ts)/n;fetch=sum(r['fetch_ms'] for r in ls)/n
        summary[rep]=dict(tokens=n,stages_ms=dict(totals),bytes_per_token=b,fetch_ms=fetch,
            effective_GB_s=b/1e6/fetch,ideal_gen5_x16_ms=b/(64e9*128/130)*1000,
            mean_prefix_ready_to_gu_us=statistics.mean(r['prefix_ready_to_gu_us'] for r in ls),
            mean_down_dependencies_to_start_us=statistics.mean(r['down_dependencies_to_start_us'] for r in ls),
            route_to_fetch_ms_per_token=sum(r['route_to_fetch_ms'] for r in ls)/n,
            down_copy_remaining_ms_per_token=sum(r['down_copy_remaining_after_gu_ms'] for r in ls)/n,
            overlap_ms=sum(r['overlap_ms'] for r in ls)/n,wall_ms=statistics.mean(r['wall_ms'] for r in ts))
        summary[rep]['transfers']={}
        for part in ('whole','prefix','down'):
            rows=[r for r in transfer_rows if r['repeat']==rep and r['part']==part]
            nb=sum(r['bytes'] for r in rows);ms=sum(r['ms'] for r in rows)
            summary[rep]['transfers'][part]=dict(MB_per_token=nb/n/1e6,ms_per_token=ms/n,effective_GB_s=nb/ms/1e6 if ms else None)
        for field in ('prefix_ready_to_gu_us','down_dependencies_to_start_us'):
            vals=sorted(r[field] for r in ls)
            assert min(vals)>=0,(field,min(vals))
            summary[rep][field+'_p95']=vals[int(.95*(len(vals)-1))]
    (p/'deep-summary.json').write_text(json.dumps(summary,indent=2))
    aggregates=collections.defaultdict(lambda: [0.0,0])
    for r in kernel_rows:
        v=aggregates[(r['repeat'],r['kernel'])];v[0]+=r['ms'];v[1]+=r['count']
    write_csv(p/'deep-kernel-means.csv',[dict(repeat=rep,kernel=name,ms_per_token=v[0]/summary[rep]['tokens'],launches_per_token=v[1]/summary[rep]['tokens']) for (rep,name),v in sorted(aggregates.items(),key=lambda x:-x[1][0])])
    rep=max(summary);s=summary[rep];items=sorted(s['stages_ms'].items(),key=lambda x:-x[1]);phase='Warm repeated' if rep else 'First-request'
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained');ys=list(range(len(items)))
    ax.barh(ys,[v for k,v in items],color=['#d68736' if 'Fetch' in k else '#237e9e' for k,v in items])
    ax.set_yticks(ys,[k for k,v in items]);ax.invert_yaxis()
    for y,(k,v) in zip(ys,items):ax.text(v+.03,y,f'{v:.3f}',va='center')
    ax.set_xlim(0,max(v for k,v in items)*1.16);ax.set_xlabel('ms/token; overlapping stage intervals, not additive')
    ax.set_title(f'{phase} decode GPU substages | {s["tokens"]} tokens | loop {s["wall_ms"]:.2f} ms')
    fig.savefig(p/'deep-stages.png',dpi=150);plt.close(fig)
    chosen=sorted([r for r in token_rows if r['repeat']==rep],key=lambda r:r['wall_ms'])[s['tokens']//2]
    ls=[r for r in layer_rows if r['repeat']==rep and r['position']==chosen['position']]
    fig,ax=plt.subplots(figsize=(13,8),layout='constrained')
    for r in ls:
        for start,end,color,label in [('fetch_start_ms','fetch_end_ms','#d68736','Fetch span'),('gu_start_ms','gu_end_ms','#237e9e','Gate/up'),('down_start_ms','down_end_ms','#59a67c','Down')]:
            intervals=r['gu_intervals'] if label=='Gate/up' else r['down_intervals'] if label=='Down' else r['fetch_intervals']
            ax.broken_barh(intervals,(r['layer']-.35,.7),facecolors=color,label=label if r==ls[0] else None)
    ax.invert_yaxis();ax.set_ylabel('MoE layer');ax.set_xlabel('ms from token entry');ax.legend()
    ax.set_title(f'{phase} token {chosen["position"]}: layer-local fetch and compute\nAttention and other work occupy gaps; fetch spans may overlap computation')
    fig.savefig(p/'deep-layer-timeline.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
