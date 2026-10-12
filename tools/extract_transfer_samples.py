"""Three actual packed expert blobs, preserving native gate/up/down bytes."""
from pathlib import Path
pack=Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya')
out=Path('docs/evidence/maya-transfer/transport');out.mkdir(exist_ok=True)
seen=set()
with (pack.parent/'GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf').open('rb') as f:
    for line in (pack/'native_experts.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        il,gu,dt,off,blob,gate,up,down=map(int,line.split()[:8])
        if gu in seen:continue
        seen.add(gu);gs=(up-gate)//288;ds=blob-2*gs
        raw=[]
        for pos,size in [(gate,gs),(up,gs),(down,ds)]:f.seek(pos);raw.append(f.read(size))
        result=b''.join(raw);assert len(result)==blob
        (out/f'type{gu}.bin').write_bytes(result)
