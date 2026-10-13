"""Nsight CUDA/NVTX timeline -> reconciled token tables and a local visual report."""
import argparse, bisect, collections, csv, html, json, re, sqlite3, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'Expert fetch':'#e28532','Expert gate/up':'#1769aa','Expert down':'#4098ce',
        'CPU answer wait':'#d1495b','Host/disk wait':'#962d44','Routing':'#9970ab',
        'Dense projections':'#4b7b56','KDA recurrence/prep':'#66a182','DSA / MLA':'#427d9d',
        'Residual/norm':'#bca55a','Embedding/head':'#715b93','Other kernel':'#8b8b8b',
        'D2H copies (VRAM to RAM)':'#56b4a9','H2D/D2D copies':'#9ed4d0','No recorded GPU activity':'#dedede'}

def category(name, layer):
    for key,cat in [('moe_cpu_wait','CPU answer wait'),('moe_gate_up','Expert gate/up'),
                    ('moe_down','Expert down'),('moe_fetch','Expert fetch'),('moe_gu_band_fetch','Expert fetch'),('moe_component','Expert fetch'),('moe_prefetch','Expert fetch'),
                    ('moe_wait','Host/disk wait'),('moe_route','Routing'),('kda_','KDA recurrence/prep'),
                    ('dsa_','DSA / MLA'),('mla_','DSA / MLA'),('headwise_','DSA / MLA'),
                    ('hc','Residual/norm'),('embed_','Embedding/head'),('head_prep','Embedding/head'),
                    ('argmax','Embedding/head')]:
        if key in name: return cat
    if any(x in name for x in ('mv_kernel','mv_q4_rows_kernel','mvc_kernel','quantize_kernel')):
        return 'Embedding/head' if layer < 0 else 'Dense projections'
    return 'Other kernel'

def union(intervals):
    merged=[]
    for s,e in sorted(intervals):
        if not merged or s>merged[-1][1]: merged.append([s,e])
        else: merged[-1][1]=max(e,merged[-1][1])
    return merged

def duration(intervals): return sum(e-s for s,e in union(intervals))

def overlap(a,b):
    a,b=union(a),union(b); i=j=0; total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return total

def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def table(rows):
    if not rows:return ''
    keys=list(rows[0]); s='<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'
    for r in rows:
        s+='<tr>'+''.join('<td>'+html.escape(f'{r[k]:.3f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>'
    return s+'</tbody></table>'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); a=ap.parse_args()
    root=a.root.resolve(); trace=root/'trace-v2'; out=root/'report'; out.mkdir(exist_ok=True)
    db=sqlite3.connect(trace/'timeline.sqlite'); db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'))
    nv=[]
    for r in db.execute('select * from NVTX_EVENTS'):
        d=dict(r); text=d.get('text') or strings.get(d.get('textId'),'')
        if text:nv.append(dict(start=d['start'],end=d.get('end') or d['start'],text=text,tid=d['globalTid']))
    tokens=[]; layers=[]; cpus=[]; routes=[]
    for r in nv:
        text=r['text']
        if text.startswith('token seq='):
            m=re.search(r'seq=(\d+) pos=(\d+) input=(\d+)',text)
            if m: tokens.append(dict(r,seq=int(m[1]),pos=int(m[2]),input=int(m[3]),kernels=[],copies=[],cpu=[],routes=[],apis=[]))
        elif text.startswith('layer='):
            m=re.search(r'layer=(\d+) pos=(\d+)',text)
            layers.append(dict(r,layer=int(m[1]),pos=int(m[2])))
        elif text.startswith('CPU experts'): cpus.append(r)
        elif text.startswith('route='): routes.append(r)
    tokens.sort(key=lambda t:t['start']); layers.sort(key=lambda t:t['start'])
    if not tokens:raise RuntimeError('No NVTX token ranges captured')
    ts=[t['start'] for t in tokens]; ls=[t['start'] for t in layers]
    def find_token(start):
        i=bisect.bisect_right(ts,start)-1
        return tokens[i] if i>=0 and start<=tokens[i]['end'] else None
    # One target CUDA process, one device. Correlation joins identify the host layer
    # that launched each kernel, not the layer the CPU happened to be enqueuing later.
    launch_ids=[i for i,s in strings.items() if 'cudaLaunch' in s or s=='cudaGraphLaunch']
    db.execute('create temp table launch_api as select * from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in ('+','.join(map(str,launch_ids))+')')
    db.execute('create index temp.launch_corr on launch_api(correlationId)')
    query='''select k.start,k.end,k.streamId,k.correlationId,k.demangledName,k.shortName,
        r.start as api_start,r.end as api_end,r.globalTid
        from CUPTI_ACTIVITY_KIND_KERNEL k left join launch_api r
        on k.correlationId=r.correlationId order by k.start'''
    missing_layer=0
    for r in db.execute(query):
        t=find_token(r['start'])
        if t is None:continue
        layer=-1
        if r['api_start'] is not None:
            i=bisect.bisect_right(ls,r['api_start'])-1
            if i>=0 and r['api_start']<=layers[i]['end'] and layers[i]['tid']==r['globalTid']:layer=layers[i]['layer']
        name=strings[r['demangledName']]
        t['kernels'].append(dict(start=r['start'],end=r['end'],layer=layer,name=name,
                                  stage=category(name,layer),stream=r['streamId'],api_start=r['api_start']))
    for r in db.execute('select start,end,streamId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start'):
        t=find_token(r['start'])
        if t is not None:t['copies'].append(dict(r))
    for r in cpus:
        t=find_token(r['start'])
        if t is not None:t['cpu'].append(r)
    for r in routes:
        t=find_token(r['start'])
        if t is not None:t['routes'].append(r)
    # CUDA API durations are host-call time, not device execution time.
    query_ids=[i for i,s in strings.items() if s.startswith('cudaEventQuery')]
    excluded=','.join(map(str,query_ids)) or '-1'
    for r in db.execute('select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where nameId not in ('+excluded+') order by start'):
        t=find_token(r['start'])
        if t is not None and r['globalTid']==t['tid']:
            t['apis'].append(dict(start=r['start'],end=r['end'],name=strings[r['nameId']]))
    requests=json.loads((trace/'requests.json').read_text())
    groups=[]
    for t in tokens:
        if not groups or t['pos']<=groups[-1][-1]['pos']:groups.append([])
        groups[-1].append(t)
    if len(groups)!=len(requests):
        raise RuntimeError(f'Expected {len(requests)} request groups, got {len(groups)}: '+str([(g[0]['pos'],g[-1]['pos']) for g in groups]))
    scored=[]
    for group,req in zip(groups,requests):
        for t in group:
            t['task']=req['task']; t['index']=t['pos']-len(req['prompt_ids'])+2
            # p=prompt_length-1 belongs to prefill logits; decode forwards start at p=prompt_length.
            if req['task']!='warmup' and t['pos']>=len(req['prompt_ids']):scored.append(t)
    if not scored:raise RuntimeError('No scored decode tokens')
    stage_rows=[]; token_rows=[]; kernel_rows=[]; layer_rows=[]; copy_rows=[]; cpu_rows=[]
    for t in scored:
        start=t['start']; end=t['end']; wall=(end-start)/1e6
        sums=collections.Counter()
        for k in t['kernels']:
            sums[k['stage']]+=(k['end']-k['start'])/1e6
            kernel_rows.append(dict(task=t['task'],seq=t['seq'],output_index=t['index'],layer=k['layer'],
                start_ms=(k['start']-start)/1e6,end_ms=(k['end']-start)/1e6,
                duration_ms=(k['end']-k['start'])/1e6,cumulative_wall_ms=(k['end']-start)/1e6,stage=k['stage'],kernel=k['name']))
        gpu=[(k['start'],k['end']) for k in t['kernels']]+[(c['start'],c['end']) for c in t['copies']]
        cpu=[(c['start'],c['end']) for c in t['cpu']]
        for c in t['copies']:
            copy_rows.append(dict(task=t['task'],seq=t['seq'],direction={1:'H2D',2:'D2H',8:'D2D'}.get(c['copyKind'],str(c['copyKind'])),
                start_ms=(c['start']-start)/1e6,end_ms=(c['end']-start)/1e6,duration_ms=(c['end']-c['start'])/1e6,bytes=c['bytes']))
        for c in t['cpu']:
            cpu_rows.append(dict(task=t['task'],seq=t['seq'],start_ms=(c['start']-start)/1e6,end_ms=(c['end']-start)/1e6,
                duration_ms=(c['end']-c['start'])/1e6,label=c['text']))
        busy=duration(gpu)/1e6; cpu_ms=duration(cpu)/1e6; both=overlap(gpu,cpu)/1e6
        fetch_bytes=fetch_count=cpu_count=disk_count=0
        for r in t['routes']:
            fields=dict(re.findall(r'(\w+)=(\d+)',r['text']))
            count=int(fields['fetch']).bit_count(); fetch_count+=count; fetch_bytes+=count*int(fields['blob'])
            cpu_count+=int(fields['cpu']).bit_count(); disk_count+=int(fields['miss']).bit_count()
        row=dict(task=t['task'],seq=t['seq'],output_index=t['index'],context_position=t['pos'],
                 wall_ms=wall,gpu_busy_ms=busy,no_gpu_activity_ms=wall-busy,cpu_service_ms=cpu_ms,
                 cpu_gpu_overlap_ms=both,cpu_only_ms=cpu_ms-both,neither_ms=wall-busy-cpu_ms+both,
                 fetch_MB=fetch_bytes/1e6,fetch_experts=fetch_count,cpu_experts=cpu_count,host_miss_experts=disk_count,
                 d2h_MB=sum(c['bytes'] for c in t['copies'] if c['copyKind']==2)/1e6,
                 d2h_ms=duration([(c['start'],c['end']) for c in t['copies'] if c['copyKind']==2])/1e6,
                 host_cuda_nonpoll_ms=duration([(c['start'],c['end']) for c in t['apis']])/1e6,
                 cpu_during_math_ms=overlap(cpu,[(k['start'],k['end']) for k in t['kernels'] if k['stage'] not in ['Expert fetch','CPU answer wait','Host/disk wait']])/1e6,
                 cpu_during_fetch_ms=overlap(cpu,[(k['start'],k['end']) for k in t['kernels'] if k['stage']=='Expert fetch'])/1e6,
                 cpu_during_wait_ms=overlap(cpu,[(k['start'],k['end']) for k in t['kernels'] if k['stage'] in ['CPU answer wait','Host/disk wait']])/1e6,
                 **{s:sums[s] for s in COLORS if s not in ['D2H copies (VRAM to RAM)','H2D/D2D copies','No recorded GPU activity']})
        token_rows.append(row); t['summary']=row
        for layer in range(45):
            ks=[k for k in t['kernels'] if k['layer']==layer]
            if not ks:continue
            s=min(k['start'] for k in ks); e=max(k['end'] for k in ks)
            layer_rows.append(dict(task=t['task'],seq=t['seq'],output_index=t['index'],layer=layer,
                start_ms=(s-start)/1e6,end_ms=(e-start)/1e6,span_ms=(e-s)/1e6,
                kernel_ms=sum(k['end']-k['start'] for k in ks)/1e6,cumulative_wall_ms=(e-start)/1e6,
                fetch_ms=sum(k['end']-k['start'] for k in ks if k['stage']=='Expert fetch')/1e6,
                cpu_wait_ms=sum(k['end']-k['start'] for k in ks if k['stage']=='CPU answer wait')/1e6))
    for name,rows in [('tokens',token_rows),('kernels',kernel_rows),('layers',layer_rows),('copies',copy_rows),('cpu',cpu_rows)]:write_csv(out/(name+'.csv'),rows)
    stock=json.loads((root/'stock'/'requests.json').read_text()); run_rows=[]
    for r in stock:
        if r['task']=='warmup':continue
        intervals=np.diff(r['arrivals_ms'])
        run_rows.append(dict(task=r['task'],repeat=r['repeat']+1,tokens=r['tokens'],
            decode_TPS=r['tokens']/(r['decode_ms']/1000),decode_ms=r['decode_ms'],
            mean_interval_ms=float(np.mean(intervals)),median_interval_ms=float(np.median(intervals)),
            p95_interval_ms=float(np.percentile(intervals,95)),max_interval_ms=float(max(intervals))))
    write_csv(out/'runs.csv',run_rows)
    tasks=['csv','explanation','web']; overview=[]; averages=[]
    for task in tasks:
        rr=[r for r in run_rows if r['task']==task]; tt=[t for t in token_rows if t['task']==task]
        rq=next(r for r in requests if r['task']==task)
        stock_first=next(r for r in stock if r['task']==task)
        overview.append(dict(task=task,stock_TPS=sum(r['tokens'] for r in rr)/(sum(r['decode_ms'] for r in rr)/1000),
            stock_TPS_min=min(r['decode_TPS'] for r in rr),stock_TPS_max=max(r['decode_TPS'] for r in rr),
            trace_TPS=rq['tokens']/(rq['decode_ms']/1000),trace_tokens=len(tt),
            trace_mean_token_ms=statistics.mean(t['wall_ms'] for t in tt),
            output_ids_equal=rq['output_ids']==stock_first['output_ids'],
            stock_repeats_identical=all(r['output_ids']==stock_first['output_ids'] for r in stock if r['task']==task)))
        for stage in COLORS:
            vals=[t.get(stage,0) for t in tt]
            if stage=='No recorded GPU activity':vals=[t['no_gpu_activity_ms'] for t in tt]
            if stage in ['D2H copies (VRAM to RAM)','H2D/D2D copies']:
                vals=[duration([(c['start'],c['end']) for c in t['copies'] if (c['copyKind']==2)==(stage=='D2H copies (VRAM to RAM)')])/1e6 for t in scored if t['task']==task]
            averages.append(dict(task=task,stage=stage,mean_ms=statistics.mean(vals),p95_ms=float(np.percentile(vals,95)),
                                 max_ms=max(vals)))
    write_csv(out/'stage_averages.csv',averages); write_csv(out/'overview.csv',overview)
    # Charts: real unprofiled observations and diagnostic timing are kept separate.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for i,task in enumerate(tasks):
        rr=[r for r in run_rows if r['task']==task]
        axs[0].bar(np.arange(3)+i*4,[r['decode_TPS'] for r in rr],label=task)
        for r in [r for r in stock if r['task']==task]:axs[1].plot(np.arange(1,len(r['arrivals_ms'])),np.diff(r['arrivals_ms']),alpha=.55,label=task if r['repeat']==0 else None)
    axs[0].set_title('Unprofiled stock engine: three repetitions per workload'); axs[0].set_ylabel('Decode tokens/s'); axs[0].set_xticks([1,5,9],tasks); axs[0].legend()
    axs[1].set_title('Observed output-token intervals (first token excluded)'); axs[1].set_ylabel('ms'); axs[1].set_xlabel('Output interval');axs[1].legend()
    fig.savefig(out/'01-baseline.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    means={s:statistics.mean(r['mean_ms'] for r in averages if r['stage']==s) for s in COLORS}
    order=sorted(means,key=means.get,reverse=True)
    axs[0].barh(order[::-1],[means[s] for s in order[::-1]],color=[COLORS[s] for s in order[::-1]])
    axs[0].set_xlabel('Average ms per traced decode token');axs[0].set_title('Kernel time by stage + gaps (instrumented)')
    for task in tasks:
        tt=[r for r in token_rows if r['task']==task];axs[1].plot([r['output_index'] for r in tt],[r['wall_ms'] for r in tt],label=task)
    axs[1].set_title('Every captured decode loop');axs[1].set_ylabel('Host token-loop elapsed ms');axs[1].set_xlabel('Output index');axs[1].legend()
    fig.savefig(out/'02-breakdown.png',dpi=150);plt.close(fig)
    selected=[]
    for task in tasks:
        tt=sorted([t for t in scored if t['task']==task],key=lambda t:t['summary']['wall_ms'])
        for label,i in [('fast',0),('median',len(tt)//2),('slow',len(tt)-1)]:
            t=tt[i];selected.append((label,t))
    details=[]
    for label,t in selected:
        base=t['start']; wall=(t['end']-base)/1e6; name=f"{t['task']}-{label}"
        fig,axs=plt.subplots(2,1,figsize=(16,7),layout='constrained',gridspec_kw={'height_ratios':[2,1]})
        ax=axs[0]
        for k in t['kernels']:
            y=3 if k['stage']=='Expert fetch' else 2 if k['stage'] in ['CPU answer wait','Host/disk wait'] else 4
            ax.broken_barh([((k['start']-base)/1e6,(k['end']-k['start'])/1e6)],(y-.32,.64),facecolors=COLORS[k['stage']])
        for c in t['cpu']:ax.broken_barh([((c['start']-base)/1e6,(c['end']-c['start'])/1e6)],(.68,.64),facecolors='#7549a5')
        for c in t['copies']:ax.broken_barh([((c['start']-base)/1e6,(c['end']-c['start'])/1e6)],(-.32,.64),facecolors=COLORS['D2H copies (VRAM to RAM)' if c['copyKind']==2 else 'H2D/D2D copies'])
        for c in t['apis']:ax.broken_barh([((c['start']-base)/1e6,(c['end']-c['start'])/1e6)],(4.68,.64),facecolors='#5b6576')
        ax.set_yticks([0,1,2,3,4,5],['CUDA copies (mostly D2H)','CPU expert service','GPU waits','GPU expert fetch','GPU arithmetic','Host CUDA calls (no polling)'])
        ax.set_ylim(-.6,6.0);ax.set_xlim(0,wall);ax.set_xlabel('Elapsed ms from host token entry');ax.set_title(f"{t['task']} / {label} token / seq {t['seq']} / position {t['pos']} / {wall:.2f} ms",pad=18)
        lr=[r for r in layer_rows if r['seq']==t['seq']]
        for r in lr:
            if r['layer']%5==0:ax.axvline(r['start_ms'],color='#999',lw=.5,alpha=.6);ax.text(r['start_ms'],5.6,'L'+str(r['layer']),fontsize=8)
        axs[1].plot([r['layer'] for r in lr],[r['end_ms'] for r in lr],marker='.',label='Cumulative elapsed to layer completion')
        axs[1].bar([r['layer'] for r in lr],[r['span_ms'] for r in lr],alpha=.5,label='Individual layer span')
        axs[1].set_xlabel('Layer');axs[1].set_ylabel('ms');axs[1].legend()
        fig.savefig(out/(name+'.png'),dpi=130);plt.close(fig)
        ev=[r.copy() for r in kernel_rows if r['seq']==t['seq']]
        for c in [r for r in copy_rows if r['seq']==t['seq']]:
            ev.append(dict(task=t['task'],seq=t['seq'],output_index=t['index'],layer=-1,start_ms=c['start_ms'],
                end_ms=c['end_ms'],duration_ms=c['duration_ms'],cumulative_wall_ms=c['end_ms'],
                stage=c['direction']+' copy',kernel=f"{c['bytes']} bytes"))
        ev.sort(key=lambda r:r['start_ms'])
        details.append(dict(name=name,summary=t['summary'],layers=lr,events=ev))
    # The median coding token's slowest layer: raw kernels and actual cumulative offsets.
    median=next(t for label,t in selected if label=='median' and t['task']=='csv')
    lr=[r for r in layer_rows if r['seq']==median['seq']]; worst=max(lr,key=lambda r:r['span_ms'])
    ks=[k for k in median['kernels'] if k['layer']==worst['layer']]; base=min(k['start'] for k in ks)
    fig,ax=plt.subplots(figsize=(14,7),layout='constrained')
    for i,k in enumerate(ks):
        ax.barh(i,(k['end']-k['start'])/1e6,left=(k['start']-base)/1e6,color=COLORS[k['stage']])
        ax.text((k['end']-base)/1e6,i,f" {(k['end']-k['start'])/1e6:.3f} ms",va='center',fontsize=8)
    ax.set_yticks(range(len(ks)),[k['stage'] for k in ks]);ax.invert_yaxis();ax.set_xlabel('Elapsed ms within layer');ax.set_title(f"Median coding token: layer {worst['layer']} (slowest layer in this token)")
    fig.savefig(out/'03-layer-zoom.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(15,9),layout='constrained')
    for ax,task in zip(axs,tasks):
        tt=[t for t in scored if t['task']==task]; index={t['seq']:i for i,t in enumerate(tt)}; arr=np.full((45,len(tt)),np.nan)
        for r in layer_rows:
            if r['task']==task:arr[r['layer'],index[r['seq']]]=r['span_ms']
        im=ax.imshow(arr,aspect='auto',origin='lower',cmap='magma');ax.set_title(task);ax.set_ylabel('Layer');fig.colorbar(im,ax=ax,label='Layer span ms')
    axs[-1].set_xlabel('Captured decode token');fig.savefig(out/'04-heatmap.png',dpi=150);plt.close(fig)
    # Explicit reconciliation validates scope attribution and no overlapping sums mistaken for wall time.
    checks=dict(token_count=len(scored),kernel_count=len(kernel_rows),layer_rows=len(layer_rows),
        expected_layers=len(scored)*45,all_tokens_45_layers=len(layer_rows)==len(scored)*45,
        min_unattributed_ms=min(r['no_gpu_activity_ms'] for r in token_rows),
        max_partition_error_ms=max(abs(r['wall_ms']-(r['gpu_busy_ms']-r['cpu_gpu_overlap_ms']+r['cpu_only_ms']+r['cpu_gpu_overlap_ms']+r['neither_ms'])) for r in token_rows))
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    layer_averages=[]
    for task in tasks:
        for layer in range(45):
            rr=[r for r in layer_rows if r['task']==task and r['layer']==layer]
            layer_averages.append(dict(task=task,layer=layer,mean_span_ms=statistics.mean(r['span_ms'] for r in rr),
                mean_cumulative_ms=statistics.mean(r['end_ms'] for r in rr),mean_fetch_ms=statistics.mean(r['fetch_ms'] for r in rr),
                mean_cpu_wait_ms=statistics.mean(r['cpu_wait_ms'] for r in rr)))
    write_csv(out/'layer_averages.csv',layer_averages)
    traffic=[]
    for task in tasks:
        rr=[r for r in token_rows if r['task']==task]
        traffic.append(dict(task=task,fetch_MB=statistics.mean(r['fetch_MB'] for r in rr),
            d2h_MB=statistics.mean(r['d2h_MB'] for r in rr),d2h_ms=statistics.mean(r['d2h_ms'] for r in rr),
            fetch_payload_GB_s=sum(r['fetch_MB'] for r in rr)/sum(r['Expert fetch'] for r in rr),
            d2h_payload_GB_s=sum(r['d2h_MB'] for r in rr)/sum(r['d2h_ms'] for r in rr),
            cpu_service_ms=statistics.mean(r['cpu_service_ms'] for r in rr),
            cpu_during_fetch_ms=statistics.mean(r['cpu_during_fetch_ms'] for r in rr),
            cpu_during_math_ms=statistics.mean(r['cpu_during_math_ms'] for r in rr),
            cpu_during_wait_ms=statistics.mean(r['cpu_during_wait_ms'] for r in rr)))
    write_csv(out/'traffic.csv',traffic)
    data=json.dumps(details).replace('</','<\\/')
    doc='''<!doctype html><meta charset="utf-8"><title>Maya decode microscope</title>
<style>body{font:16px system-ui;background:#f5f6f8;color:#182333;max-width:1500px;margin:30px auto;padding:0 24px}h1{font-size:34px}h2{margin-top:44px}p{max-width:1100px;line-height:1.6}.card{background:white;padding:22px;border-radius:12px;margin:20px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}thead{position:sticky;top:0;background:#eaf0f6}.scroll{max-height:520px;overflow:auto}select{padding:10px;font-size:16px}a{color:#1769aa}code{background:#eaf0f6;padding:2px 5px}</style>
<h1>Maya decode microscope</h1><p>GLM GSQ-RCO 3.0bit Q4-attention · RTX 5090 / 9950X3D · c=1 · 131,072-token capacity · no MTP or suffix speculation. Three unprofiled repetitions per task; one CUDA/NVTX capture spanning three task runs. 128 output tokens per request. Warm-up excluded.</p>
<div class="card"><b>Read this timeline correctly</b><p>GPU bars are actual execution, attributed to the launching layer through CUDA correlation IDs. CPU bars are elapsed expert-service calls (including worker coordination), not summed CPU-core execution. Rows overlap: do not add CPU time to GPU time. Grey/unoccupied time means no recorded GPU kernel/copy, not proof that the CPU caused the gap. Mapped-RAM expert fetches appear as GPU kernels, not memcpy events. Nsight changes timing; use stock runs for throughput and trace runs for diagnosis. CUDA API tracing is enabled; CPU instruction sampling and OS context switches are not captured.</p></div>
<p><b>Capture caveats:</b> profiler residency reduces expert slots from 2731 to 2722. Coding output differs between the first stock run and trace; explanation and web output agree. Stock repetitions also expose any hybrid run-to-run differences in the table. These are workload measurements, not a fixed-token causal estimate of profiler overhead. Capacity is 128K, but these short prompts occupy only tens to hundreds of tokens. The decode TPS counter includes the first token sampled from prefill; token timelines cover the subsequent 127 forward passes per task.</p>
<h2>1. Normal execution, repeated</h2>'''+table(overview)+'<img src="01-baseline.png"><div class="scroll">'+table(run_rows)+'</div>'+'''
<h2>2. Captured execution breakdown</h2><img src="02-breakdown.png"><p>Stage averages include every scored decode forward pass, excluding prompt processing and warm-up. Kernel categories are exclusive; DMA may overlap kernels. The token CSV separately reconciles the union of device activity and CPU service against elapsed wall time.</p><div class="scroll">'''+table(averages)+'''</div>
<h3>Traffic and CPU overlap</h3>'''+table(traffic)+'''<p>D2H is primarily whole-expert demotion at the token boundary: 8.3 / 10.8 / 14.2 MB blobs. This is a different transfer from the GPU fetch kernel pulling RAM experts. CPU-service time overlaps fetch, arithmetic and wait kernels; those columns must not be added to token duration. Host miss masks are requests for service, not proof of SSD reads (the engine logged zero disk reads).</p>
<h2>3. Individual token timelines</h2><p>Select fastest, median or slowest captured token for each task. Vertical labels mark layer numbers. Full per-token, per-layer and per-kernel CSVs below cover every captured token.</p><select id="pick"></select><div id="detail"></div>
<h2>4. Within one layer</h2><img src="03-layer-zoom.png"><p>Bars retain their actual offsets and gaps; this is not a synthetic stacked schedule.</p>
<h2>5. All layers across all captured tokens</h2><img src="04-heatmap.png"><div class="scroll">'''+table(layer_averages)+'''</div>
<h2>Raw tables</h2><p><a href="tokens.csv">Every token</a> · <a href="layers.csv">Every layer, with cumulative elapsed time</a> · <a href="kernels.csv">Every kernel, sequential offsets and durations</a> · <a href="copies.csv">Every copy</a> · <a href="cpu.csv">CPU service spans</a> · <a href="stage_averages.csv">Stage averages</a> · <a href="runs.csv">Normal runs</a> · <a href="checks.json">Reconciliation checks</a></p>
<script>const data='''+data+''';let sel=document.getElementById('pick');data.forEach((d,i)=>sel.add(new Option(d.name,i)));function tab(rows){let keys=Object.keys(rows[0]);return '<div class="scroll"><table><thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+String(typeof r[k]==='number'?r[k].toFixed(3):r[k]).replaceAll('<','&lt;')+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}function show(){let d=data[+sel.value];document.getElementById('detail').innerHTML='<img src="'+d.name+'.png"><h3>Token accounting</h3>'+tab([d.summary])+'<h3>Layer sequence</h3>'+tab(d.layers)+'<h3>Kernel sequence</h3>'+tab(d.events)}sel.onchange=show;show();</script>'''
    (out/'index.html').write_text(doc,encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(overview=overview,checks=checks,stage_means=means,traffic=traffic),indent=2))
    print(json.dumps(dict(overview=overview,checks=checks,stage_means=means),indent=2))

if __name__=='__main__':main()
