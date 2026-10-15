"""Bounded ordinary-forward vs window-first-row fixed-input numerical diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokens-file',type=Path,required=True)
    p.add_argument('--prefix',type=int,required=True)
    p.add_argument('--backend',choices=('gpu','hybrid'),default='gpu')
    p.add_argument('--steps',type=int,default=6)
    p.add_argument('--route-log',action='store_true')
    p.add_argument('--env',action='append',default=[])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'maya-retained-v117.local.json').read_text())
    exe=ROOT/'build/maya_verify_scratch_test.exe'
    for mode in ('forward','window_first'):
        env={k:v for k,v in os.environ.items() if not k.startswith(('MAYA_','STRATA_'))}
        env.update(cfg['env'])
        env.update(STRATA_GLM_CPU_PLAN='000000000' if a.backend=='gpu' else '011223344',
                   STRATA_GLM_POOL_GB='16',STRATA_GLM_RAM_GB='94',MAYA_WIN_SCRATCH='64',MAYA_WIN_REUSE='1',
                   STRATA_GLM_WIN_STAGGER='0',STRATA_GLM_WIN_ALLCPU='0',MAYA_WIN_FETCH_STAGE='1',MAYA_WIN_GROUPED='2',
                   MAYA_WIN_HC_BATCH='1',MAYA_WIN_COMPONENT_FETCH='1',MAYA_MV_EXACT_COLS='1',
                   MAYA_VERIFY_PREFIX_TOKENS=str(a.prefix),MAYA_VERIFY_BATCH_PREFILL='1',
                   MAYA_VERIFY_STEPS=str(a.steps),MAYA_VERIFY_REPEATS='1',MAYA_VERIFY_WIDTH_A='3',
                   MAYA_VERIFY_WIDTH_B='5',MAYA_VERIFY_REFERENCE_MODE=mode)
        for setting in a.env:
            key,value=setting.split('=',1);env[key]=value
        if a.route_log:env['STRATA_GLM_ROUTE_LOG']=str((a.out/(mode+'-routes')).resolve())
        env['PATH']=';'.join(cfg['lib_dirs'])+';'+env['PATH']
        cmd=[str(exe),'G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya',str(a.tokens_file.resolve()),str((a.out/(mode+'.f32')).resolve())]
        manifest=dict(command=cmd,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
            token_sha256=hashlib.sha256(a.tokens_file.read_bytes()).hexdigest(),
            env={k:v for k,v in env.items() if k.startswith(('MAYA_','STRATA_'))},
            scope='One prefix logit row plus fixed-input rows; window T3/T5 commits exactly one input. Numerical diagnostic, not throughput.')
        with (a.out/(mode+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        manifest['returncode']=result.returncode
        (a.out/(mode+'-manifest.json')).write_text(json.dumps(manifest,indent=2))
        print(mode,result.returncode,flush=True)
        if result.returncode:raise RuntimeError(mode)
    x=np.fromfile(a.out/'forward.f32',dtype=np.float32).reshape(-1,154880)
    y=np.fromfile(a.out/'window_first.f32',dtype=np.float32).reshape(-1,154880)
    assert x.shape==y.shape==(a.steps+1,154880)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    rows=[]
    for i,(left,right) in enumerate(zip(x,y)):
        delta=np.abs(left-right)
        lt=np.argsort(left)[-2:][::-1];rt=np.argsort(right)[-2:][::-1]
        rows.append(dict(row=i,kind='prefix' if i==0 else 'fixed_input',
            input_position=a.prefix-1+i,exact=bool(np.array_equal(left,right)),
            max_abs=float(delta.max()),mean_abs=float(delta.mean()),different_values=int(np.count_nonzero(delta)),
            forward_top=int(lt[0]),window_top=int(rt[0]),greedy_match=bool(lt[0]==rt[0]),
            forward_margin=float(left[lt[0]]-left[lt[1]]),window_margin=float(right[rt[0]]-right[rt[1]])))
    summary=dict(backend=a.backend,values=int(x.size),exact=bool(np.array_equal(x,y)),
                 max_abs=float(np.abs(x-y).max()),rows=rows)
    (a.out/'comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
