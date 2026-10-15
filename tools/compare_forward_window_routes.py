"""Align fixed-forward and window-first route logs from check_window_forward.py."""
import argparse,json
from pathlib import Path


def read_routes(path):
    rows=[]
    for line in path.read_text().splitlines():
        data=list(map(int,line.split('|')[0].split()))
        assert len(data)==13, (path,line)
        rows.append(dict(layer=data[0],ids=data[1:9],fetch=data[9],miss=data[10],cpu=data[11],promo=data[12]))
    return rows


def align(directory):
    a=read_routes(directory/'forward-routes.0');b=read_routes(directory/'window_first-routes.0')
    manifest=json.loads((directory/'window_first-manifest.json').read_text())
    env=manifest['env'];steps=int(env['MAYA_VERIFY_STEPS']);widths=[int(env['MAYA_VERIFY_WIDTH_A']),int(env['MAYA_VERIFY_WIDTH_B'])]
    assert len(a)==42*(steps+1)
    assert len(b)==42*(1+sum(widths[i%2] for i in range(steps)))
    left=[a[i*42:(i+1)*42] for i in range(steps+1)]
    right=[b[:42]];at=42
    for i in range(steps):
        width=widths[i%2]
        block=b[at:at+42*width];at+=42*width
        for n,layer in enumerate(range(3,45)):
            assert all(r['layer']==layer for r in block[n*width:(n+1)*width])
        right.append(block[::width])
    result=[]
    for row,(x,y) in enumerate(zip(left,right)):
        assert [r['layer'] for r in x]==[r['layer'] for r in y]==list(range(3,45))
        mismatches=[]
        for l,r in zip(x,y):
            cpu_l={e for rank,e in enumerate(l['ids']) if l['cpu']>>rank&1}
            cpu_r={e for rank,e in enumerate(r['ids']) if r['cpu']>>rank&1}
            if l!=r:
                mismatches.append(dict(layer=l['layer'],ordered_ids_equal=l['ids']==r['ids'],
                    cpu_experts_equal=cpu_l==cpu_r,forward_cpu=sorted(cpu_l),window_cpu=sorted(cpu_r),forward=l,window=r))
        result.append(dict(row=row,route_records_equal=not mismatches,
            changed_layers=len(mismatches),first_difference=mismatches[0] if mismatches else None,
            first_cpu_difference=next((r for r in mismatches if not r['cpu_experts_equal']),None),
            first_id_difference=next((r for r in mismatches if not r['ordered_ids_equal']),None)))
    return dict(scope='Route IDs and masks only; input floats, weights and source bytes are not logged.',rows=result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    result=align(a.directory)
    (a.directory/'route-comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
