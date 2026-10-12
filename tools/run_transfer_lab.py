"""One controlled fixed-input experiment, with immutable manifest and stdout/stderr."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--exe',default='maya_transfer_lab.exe')
    p.add_argument('--env',action='append',default=[]);p.add_argument('--repeats',default=2,type=int)
    p.add_argument('--occupied',default=0,type=int)
    a=p.parse_args();out=ROOT/'docs/evidence/maya-transfer'/a.name;out.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'docs/evidence/maya-timeline/stock/requests.json').read_text())
    row=next(x for x in source if x['task']=='csv')
    prompt=row['prompt_ids']; continuation=row['output_ids']
    if a.occupied:
        # Reproducible code tokens as filler; natural short prompt follows. This is an occupied-context fixture.
        prompt=(continuation*((a.occupied//len(continuation))+1))[:a.occupied-len(prompt)]+prompt
    ids=prompt+continuation
    (out/'tokens.txt').write_text(' '.join(map(str,ids)))
    env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','MAYA_'))}
    env.update(STRATA_GLM_NO_MTP='1',STRATA_GLM_USAGE='0',STRATA_GLM_CPU_PLAN='011223344',
        STRATA_GLM_POOL_GB='22',STRATA_GLM_RAM_GB='87.4',STRATA_GLM_PREFILL='auto',STRATA_GLM_TIER_DIAG='1')
    env['PATH']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;'+env['PATH']
    for v in a.env:
        k,val=v.split('=',1);env[k]=val
    exe=ROOT/'build'/a.exe
    cmd=[str(exe),'G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya',str(out/'tokens.txt'),str(len(prompt)),str(out/'fixed'),str(a.repeats)]
    (out/'manifest.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('STRATA_','MAYA_'))},
        binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),tokens_sha256=hashlib.sha256((out/'tokens.txt').read_bytes()).hexdigest()),indent=2))
    with (out/'engine.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,env=env,stderr=log,stdout=log,check=True)
    print('complete',a.name,flush=True)
if __name__=='__main__':main()
