"""Summarize unprofiled fixed-input call latency and explicit graph engagement."""
import argparse
import json
import re
import statistics
from pathlib import Path

CALL=re.compile(r'verify call: repeat=(\d+) step=(\d+) width=(\d+) position=(\d+) ms=([\d.]+)')
COUNTS=re.compile(r'MoE graph CUDA\d+: (\d+) eligible calls, (\d+) captures / (\d+) replays, (\d+) route credits, (\d+) nodes built, (\d+) cached keys; (\d+) capture failures / (\d+) disabled-key fallbacks / (\d+) launch failures / (\d+) invalidations')

def summarize(directory):
    result={}
    for label in ('graph-off','graph-on'):
        text=(directory/(label+'.log')).read_text(errors='replace')
        calls=[dict(repeat=int(r),step=int(s),width=int(w),position=int(p),ms=float(t)) for r,s,w,p,t in CALL.findall(text)]
        if not calls:raise ValueError('No measured calls: '+label)
        groups=[]
        for repeat,width in sorted({(r['repeat'],r['width']) for r in calls}):
            values=[r['ms'] for r in calls if r['repeat']==repeat and r['width']==width]
            groups.append(dict(repeat=repeat,width=width,n=len(values),mean_ms=statistics.mean(values),min_ms=min(values),max_ms=max(values)))
        keys=('eligible','captures','replays','route_credits','nodes_built','cached_keys','capture_failures','disabled_fallbacks','launch_failures','invalidations')
        counts=[dict(zip(keys,map(int,row))) for row in COUNTS.findall(text)]
        result[label]=dict(calls=calls,groups=groups,engagement_reports=counts,
            memory_and_guard_lines=[line for line in text.splitlines() if any(s in line for s in ('expert pool ','RAM tier ','MoE graph CUDA0 host totals','MoE graph CUDA0 bypass reasons','routing ring CUDA'))])
        if label=='graph-on' and (not counts or sum(r['replays'] for r in counts)==0):raise ValueError('Graph replay did not engage')
    result['parity']=json.loads((directory/'graph-off-vs-graph-on.json').read_text())
    result['scope']='Fixed-input call wall latency including window synchronization; excludes fixture logit readback and file writes. First calls include graph creation. Only repeated warm calls assess reuse. Not generated-token TPS.'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    result=summarize(a.directory);(a.directory/'summary.json').write_text(json.dumps(result,indent=2))
    for label in ('graph-off','graph-on'):print(label,result[label]['groups'],result[label]['engagement_reports'])
    print(result['parity'])
