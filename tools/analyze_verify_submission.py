"""Separate within-window traced GPU gaps from fixture readback/commit gaps."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from trace_verify_windows import union


def analyze(path):
    conn=sqlite3.connect(path)
    kernels=conn.execute('SELECT k.start,k.end,k.correlationId,k.streamId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
    activities=[(a,b,c,stream,name) for a,b,c,stream,name in kernels]
    for table in ('MEMCPY','MEMSET'):
        activities.extend((a,b,c,s,table) for a,b,c,s in conn.execute(f'SELECT start,end,correlationId,streamId FROM CUPTI_ACTIVITY_KIND_{table}'))
    activities.sort()
    runtime={c:(a,b,name) for a,b,c,name in conn.execute('SELECT r.start,r.end,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId')}
    conn.close()
    activity_correlation_matches=sum(r[2] in runtime for r in activities)
    windows=[];current=None
    for a,b,c,stream,name in kernels:
        if 'embed_streams_kernel' in name:
            if current is None:current=dict(start=a,width=0,picks=0)
            current['width']+=1
        elif 'argmax_kernel' in name:
            assert current is not None
            current['picks']+=1
            if current['picks']==current['width']:
                current['end']=b;windows.append(current);current=None
    assert current is None and len(windows)==4, 'Require four complete marked-by-kernel windows'

    def intervals(start,end):
        rows=[r for r in activities if r[0]<end and r[1]>start]
        busy=union([(max(start,r[0]),min(end,r[1])) for r in rows])
        gaps=[]
        for prev,nxt in zip(busy,busy[1:]):
            ga,gb=prev[1],nxt[0]
            first=next(r for r in rows if r[0]==gb)
            api=runtime.get(first[2])
            late=max(0,min(gb,api[0])-ga) if api else None
            pending=max(0,min(gb,api[1])-ga) if api else None
            gaps.append(dict(start_ns=ga,end_ns=gb,us=(gb-ga)/1e3,
                next_name=first[4],next_stream=first[3],api_name=api[2] if api else None,
                api_start_ns=api[0] if api else None,api_end_ns=api[1] if api else None,
                late_api_start_us=late/1e3 if late is not None else None,
                api_not_returned_us=pending/1e3 if pending is not None else None))
        return dict(span_ms=(end-start)/1e6,busy_union_ms=sum(b-a for a,b in busy)/1e6,
                    gpu_gap_ms=sum(r['us'] for r in gaps)/1e3,
                    gap_count=len(gaps),matched_gap_count=sum(r['api_name'] is not None for r in gaps),
                    unmatched_gap_ms=sum(r['us'] for r in gaps if r['api_name'] is None)/1e3,
                    late_api_start_gap_ms=sum(r['late_api_start_us'] or 0 for r in gaps)/1e3,
                    api_not_returned_gap_ms=sum(r['api_not_returned_us'] or 0 for r in gaps)/1e3,
                    gaps=gaps)
    result=[]
    for w in windows:
        summary=intervals(w['start'],w['end'])
        selected=[r for r in kernels if w['start']<=r[0]<w['end']]
        regions=[];region=None
        for a,b,c,stream,name in selected:
            if 'moe_route_kernel' in name:
                if region is None:region=dict(start=a,routes=0,waits=0)
                region['routes']+=1
            elif 'moe_cpu_wait_kernel' in name:
                assert region is not None
                region['waits']+=1
                if region['waits']==w['width']:
                    assert region['routes']==w['width']
                    regions.append(intervals(region['start'],b));region=None
        assert region is None and len(regions)==42
        waits=[(a,b) for a,b,_,_,name in selected if 'moe_cpu_wait_kernel' in name]
        summary.update(width=w['width'],kernel_count=len(selected),moe_regions=len(regions),
                       cpu_wait_kernel_ms=sum(b-a for a,b in union(waits))/1e6,
                       moe_gpu_gap_ms=sum(r['gpu_gap_ms'] for r in regions),
                       moe_late_api_start_gap_ms=sum(r['late_api_start_gap_ms'] for r in regions))
        summary['largest_gaps']=sorted(summary.pop('gaps'),key=lambda r:r['us'],reverse=True)[:12]
        result.append(summary)
    return dict(windows=result,activity_count=len(activities),
        activity_correlation_matches=activity_correlation_matches,database_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Four warmed diagnostic windows, embed-to-final-argmax boundaries. Excludes between-window fixture readback/filesystem/commit. GPU busy includes wait kernels and copies, not just useful arithmetic. Late API start is evidence of late submission, not a guaranteed graph saving. API-not-returned time is an upper bound, not proof of launch starvation. Missing correlations are explicitly counted. Nsight overhead remains.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('database',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=analyze(a.database);a.out.write_text(json.dumps(result,indent=2))
    for row in result['windows']:print({k:v for k,v in row.items() if k!='largest_gaps'})
