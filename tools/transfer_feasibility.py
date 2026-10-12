"""Lossless compression feasibility on actual expert components, never synthetic zeros."""
import argparse, hashlib, json, time, zlib
import lz4.block, zstandard
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pack',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    rows=[]
    with a.model.open('rb') as f:
        layers=[list(map(int,x.split()[:8])) for x in (a.pack/'native_experts.txt').read_text().splitlines() if x and not x.startswith('#')]
        # Three distributed layers per class, three experts per layer, every component.
        for typ in sorted({x[1] for x in layers}):
            group=[x for x in layers if x[1]==typ]
            for v in [group[0],group[len(group)//2],group[-1]]:
                il,gu,down,off,blob,gate,up,d=v
                # Native manifest component starts; each contiguous matrix holds 288 experts.
                gs=(up-gate)//288
                ds=blob-2*gs
                for expert in (0,137,287):
                    for comp,start,size in [('gate',gate,gs),('up',up,gs),('down',d,ds)]:
                        f.seek(start+expert*size); raw=f.read(size)
                        assert len(raw)==size and size>0
                        codecs=[('lz4',lz4.block.compress,lz4.block.decompress),
                                ('zstd1',zstandard.ZstdCompressor(level=1).compress,zstandard.ZstdDecompressor().decompress)]
                        for level,compress,decompress in codecs:
                            t=time.perf_counter(); packed=compress(raw); enc=time.perf_counter()-t
                            t=time.perf_counter(); decoded=decompress(packed); dec=time.perf_counter()-t
                            assert decoded==raw
                            rows.append(dict(layer=il,gu_type=gu,down_type=down,expert=expert,component=comp,
                                bytes=size,compressed=len(packed),level=level,encode_s=enc,decode_s=dec,
                                sha256=hashlib.sha256(raw).hexdigest()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(rows,indent=2))
    for level in ('lz4','zstd1'):
        r=[x for x in rows if x['level']==level]; n=sum(x['bytes'] for x in r)
        print(level,'ratio',sum(x['compressed'] for x in r)/n,'decode GB/s',n/sum(x['decode_s'] for x in r)/1e9,flush=True)

if __name__=='__main__': main()
