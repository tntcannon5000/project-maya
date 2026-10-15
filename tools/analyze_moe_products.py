"""Compare bounded selected-row MMQ intermediates; does not simulate arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_moe_combine_capture import read


def products(path):
    data=path.read_bytes();offset=0;result={}
    while offset<len(data):
        if len(data)-offset<40: raise ValueError('truncated product header')
        h=np.frombuffer(data,dtype='<i4',count=10,offset=offset).tolist();offset+=40
        j,pos,rows,max_rows,experts,gu,down,fused,nff,e=h
        if not (0<=j<8 and j not in result and 0<nff<=65536 and 0<e<=65536
                and 0<max_rows<=rows and experts>0): raise ValueError('bad product header')
        size=(e+nff)*4
        if offset+size>len(data): raise ValueError('truncated product values')
        values=np.frombuffer(data,dtype='<f4',count=e+nff,offset=offset).copy();offset+=size
        if not np.isfinite(values).all(): raise ValueError('nonfinite product')
        result[j]={'shape':h[2:], 'position':pos, 'input':values[:e], 'hidden':values[e:]}
    if set(result)!=set(range(8)): raise ValueError('expected all eight selected MMQ routes')
    return result,hashlib.sha256(data).hexdigest()


def delta(a,b):
    if a.shape!=b.shape: raise ValueError('shape mismatch')
    return {'max_abs':float(np.abs(a-b).max()),
            'different_bits':int(np.count_nonzero(a.view('u4')!=b.view('u4')))}


def main():
    p=argparse.ArgumentParser();p.add_argument('cold',type=Path);p.add_argument('warm',type=Path)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    ca,cb=read(args.cold),read(args.warm)
    if ca['k']!=8 or cb['k']!=8 or not np.array_equal(ca['ids'],cb['ids']): raise ValueError('route mismatch')
    a,ha=products(Path(str(args.cold)+'.products'));b,hb=products(Path(str(args.warm)+'.products'))
    result={'cold_sha256':ha,'warm_sha256':hb,
      'shape_fields':['total_rows','max_rows','experts','gu_type','down_type','fused_glu','nff','embedding'], 'routes':[]}
    for j in range(8):
        for capture, product in ((ca,a[j]),(cb,b[j])):
            if product['position']!=int(capture['positions'][j]) or product['shape'][-1]!=capture['e']:
                raise ValueError('product/combined capture mismatch')
        result['routes'].append({'route':j,'expert':int(ca['ids'][j]),'cold_shape':a[j]['shape'],
            'warm_shape':b[j]['shape'],'input':delta(a[j]['input'],b[j]['input']),
            'post_swiglu':delta(a[j]['hidden'],b[j]['hidden']),
            'down_output':delta(ca['outputs'][j],cb['outputs'][j])})
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
