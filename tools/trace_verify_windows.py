"""Capture four warmed fixed-input verify windows from a saved parity manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NSYS = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe')


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def overlap(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(database):
    with closing(sqlite3.connect(database)) as conn:
        rows = conn.execute('SELECT k.start,k.end,k.streamId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
    if not rows:
        raise RuntimeError('Capture contains no GPU kernels; cannot establish engagement or overlap')
    totals = {}
    for start, end, stream, name in rows:
        entry = totals.setdefault(name, dict(name=name, calls=0, sum_ms=0.0, max_ms=0.0))
        duration = (end-start)/1e6
        entry['calls'] += 1
        entry['sum_ms'] += duration
        entry['max_ms'] = max(entry['max_ms'], duration)
    for entry in totals.values():
        entry['mean_ms'] = entry['sum_ms']/entry['calls']
    def is_fetch(name):
        return 'moe_fetch_kernel' in name or 'moe_component_kernel' in name
    fetch = union([(a, b) for a, b, _, name in rows if is_fetch(name)])
    other = union([(a, b) for a, b, _, name in rows if not is_fetch(name)])
    route = union([(a, b) for a, b, _, name in rows if 'moe_route_kernel' in name])
    gu = union([(a, b) for a, b, _, name in rows if 'moe_win_gu_' in name])
    down = union([(a, b) for a, b, _, name in rows if 'moe_win_down_' in name])
    elapsed = max(r[1] for r in rows) - min(r[0] for r in rows)
    return dict(kernel_count=len(rows), stream_ids=sorted({r[2] for r in rows}),
                captured_kernel_span_ms=elapsed / 1e6,
                fetch_union_ms=sum(b-a for a,b in fetch)/1e6,
                fetch_overlap_any_kernel_ms=overlap(fetch, other)/1e6,
                fetch_overlap_route_ms=overlap(fetch, route)/1e6,
                fetch_overlap_window_gu_ms=overlap(fetch, gu)/1e6,
                fetch_overlap_window_down_ms=overlap(fetch, down)/1e6,
                window_gu_union_ms=sum(b-a for a,b in gu)/1e6,
                window_down_union_ms=sum(b-a for a,b in down)/1e6,
                kernel_totals=sorted(totals.values(), key=lambda x:x['sum_ms'], reverse=True),
                note='Diagnostic four-window capture; interval unions avoid double counting. Other kernels include waits. No CPU overlap or throughput claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = json.loads(a.manifest.read_text())
    env = {k:v for k,v in os.environ.items() if not k.startswith(('MAYA_', 'STRATA_'))}
    env.update(source['env']);env['MAYA_VERIFY_TRACE']='1'
    cfg=json.loads((ROOT/'maya-retained-v117.local.json').read_text())
    env['PATH']=';'.join(cfg['lib_dirs'])+';'+env['PATH']
    exe=ROOT/'build/maya_verify_scratch_test.exe'
    cmd=[str(NSYS),'profile','--show-output=true','--trace=cuda,nvtx','--cuda-graph-trace=node',
         '--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop',
         '-o',str(out/'timeline'),str(exe),'G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya',
         str(a.manifest.resolve().parent/'tokens.txt'),str(out/'logits.f32')]
    (out/'manifest.json').write_text(json.dumps(dict(command=cmd,env=source['env'],
        trace_env={'MAYA_VERIFY_TRACE':'1'},binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2))
    with (out/'engine.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    subprocess.run([str(NSYS),'export','--type','sqlite','--output',str(out/'timeline.sqlite'),str(out/'timeline.nsys-rep')],check=True,timeout=120)
    result=analyze(out/'timeline.sqlite')
    (out/'overlap.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
