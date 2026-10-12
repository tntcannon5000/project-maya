"""Fixed-input Nsight capture: interval-union accounting, no CPU/GPU double counting."""
import argparse,bisect,collections,csv,json,re,sqlite3,statistics
from pathlib import Path
from analyze_decode_timeline import category,union,duration,overlap,write_csv,COLORS
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
    db=sqlite3.connect(out/'timeline.sqlite');db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'))
    scored=list(csv.DictReader((out/'fixed.csv').open()));lo=min(int(r['position']) for r in scored)
    tokens=[];cpu=[]
    for r in db.execute('select start,end,text,textId from NVTX_EVENTS'):
        text=r['text'] or strings.get(r['textId'],'')
        if text.startswith('token seq='):
            m=re.search(r'seq=(\d+) pos=(\d+) input=(\d+)',text)
            if m and int(m[2])>=lo:tokens.append(dict(start=r['start'],end=r['end'],seq=int(m[1]),pos=int(m[2]),events=[],cpu=[]))
        elif text.startswith('CPU experts') and r['end']:cpu.append((r['start'],r['end']))
    tokens.sort(key=lambda r:r['start']);starts=[x['start'] for x in tokens]
    def find(s):
        i=bisect.bisect_right(starts,s)-1
        return tokens[i] if i>=0 and s<tokens[i]['end'] else None
    for r in db.execute('select start,end,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
        t=find(r['start'])
        if t is None:continue
        name=strings[r['demangledName']];cat='DMA readiness wait' if 'moe_dma_wait' in name else category(name,0)
        t['events'].append(dict(start=r['start'],end=r['end'],stage=cat,stream=r['streamId'],bytes=0))
    for r in db.execute('select start,end,streamId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start'):
        t=find(r['start'])
        if t is not None:t['events'].append(dict(start=r['start'],end=r['end'],stage='D2H' if r['copyKind']==2 else 'H2D/other',stream=r['streamId'],bytes=r['bytes']))
    for s,e in cpu:
        t=find(s)
        if t is not None:t['cpu'].append((s,min(e,t['end'])))
    rows=[];stages=[]
    excluded={'Expert fetch','CPU answer wait','Host/disk wait','DMA readiness wait','D2H','H2D/other'}
    for t in tokens:
        groups=collections.defaultdict(list)
        for e in t['events']:groups[e['stage']].append((max(e['start'],t['start']),min(e['end'],t['end'])))
        arithmetic=[iv for cat,ivs in groups.items() if cat not in excluded for iv in ivs]
        gpu=[iv for ivs in groups.values() for iv in ivs]
        row=dict(seq=t['seq'],position=t['pos'],wall_ms=(t['end']-t['start'])/1e6,
            fetch_ms=duration(groups['Expert fetch'])/1e6,d2h_ms=duration(groups['D2H'])/1e6,
            dma_h2d_ms=duration(groups['H2D/other'])/1e6,arithmetic_ms=duration(arithmetic)/1e6,
            fetch_arithmetic_overlap_ms=overlap(groups['Expert fetch'],arithmetic)/1e6,
            d2h_arithmetic_overlap_ms=overlap(groups['D2H'],arithmetic)/1e6,
            cpu_arithmetic_overlap_ms=overlap(t['cpu'],arithmetic)/1e6,
            no_gpu_ms=((t['end']-t['start'])-duration(gpu))/1e6,
            d2h_bytes=sum(e['bytes'] for e in t['events'] if e['stage']=='D2H'),
            h2d_bytes=sum(e['bytes'] for e in t['events'] if e['stage']=='H2D/other'))
        rows.append(row);t['row']=row
        stages.extend(dict(seq=t['seq'],stage=cat,ms=duration(ivs)/1e6) for cat,ivs in groups.items())
    assert len(rows)==len(scored),(len(rows),len(scored))
    write_csv(out/'trace-tokens.csv',rows);write_csv(out/'trace-stages.csv',stages)
    summary={k:statistics.mean(r[k] for r in rows) for k in rows[0] if k not in ('seq','position')}
    summary['tokens']=len(rows);(out/'trace-summary.json').write_text(json.dumps(summary,indent=2))
    ordered=sorted(tokens,key=lambda t:t['row']['wall_ms'])
    for label,idx in [('fast',0),('median',len(ordered)//2),('slow',len(ordered)-1)]:
        t=ordered[idx];base=t['start'];fig,ax=plt.subplots(figsize=(16,5),layout='constrained')
        for e in t['events']:
            cat=e['stage'];lane=0 if cat=='D2H' else 1 if cat=='H2D/other' else 2 if cat=='Expert fetch' else 3 if cat in excluded else 4
            ax.broken_barh([((e['start']-base)/1e6,(e['end']-e['start'])/1e6)],(lane-.3,.6),facecolors=COLORS.get(cat,'#688f91'))
        for s,e in t['cpu']:ax.broken_barh([((s-base)/1e6,(e-s)/1e6)],(4.7,.6),facecolors='#7549a5')
        ax.set_yticks(range(6),['D2H','H2D/D2D','Mapped expert pull','GPU waits','GPU arithmetic','CPU service'])
        ax.set_xlim(0,t['row']['wall_ms']);ax.set_xlabel('ms from token entry');ax.set_title(f'{out.name}: {label}, position {t["pos"]}, {t["row"]["wall_ms"]:.2f} ms (profiled)')
        fig.savefig(out/f'timeline-{label}.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
