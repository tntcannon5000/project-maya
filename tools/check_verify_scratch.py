"""Two controlled-GPU fixed-input window runs; record exact logits and lifecycle outcomes."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/dflash-investigation/parity'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default='legacy,candidate');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'maya-retained-v117.local.json').read_text())
    q=json.loads((ROOT/'docs/evidence/dflash-quant-width/Q4_K_M/requests.json').read_text())[1]
    ids=json.loads((ROOT/'docs/evidence/dflash-quant-width/Q4_K_M/manifest.json').read_text())['prompt_ids']+q['output_ids']
    tokenfile=OUT/'tokens.txt';tokenfile.write_text('\n'.join(map(str,ids)))
    records=[]
    for label,slots,reuse in [('legacy',8,0),('candidate',64,1),('no-reuse',64,0),('token-path',8,0),('cpu-serial',64,1),('cpu-merged',64,1)]:
        if label not in a.cases.split(','):continue
        env={k:v for k,v in os.environ.items() if not k.startswith(('MAYA_','STRATA_'))}
        env.update(cfg['env']);env.update(STRATA_GLM_CPU_PLAN='000000000',STRATA_GLM_POOL_GB='16',
           MAYA_WIN_SCRATCH=str(slots),MAYA_WIN_REUSE=str(reuse),STRATA_GLM_WIN_STAGGER='0')
        env['PATH']=';'.join(cfg['lib_dirs'])+';'+env['PATH']
        if label=='token-path':env['STRATA_GLM_WIN_BATCH']='0'
        if label.startswith('cpu-'):
            env.update(STRATA_GLM_CPU_PLAN='012345678',STRATA_GLM_WIN_ALLCPU='1',
              STRATA_GLM_DFLASH='G:/Models/GLM-5.3-Flash-DFlash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf',
              MAYA_WIN_SINGLE_NODE_BATCH='1' if label=='cpu-merged' else '0')
        cmd=[str(ROOT/'build/maya_verify_scratch_test.exe'),'G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya',str(tokenfile),str(OUT/(label+'.f32'))]
        with (OUT/(label+'.log')).open('w') as log:
            r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        records.append(dict(label=label,returncode=r.returncode,env={k:v for k,v in env.items() if k.startswith(('MAYA_','STRATA_'))}))
        (OUT/(label+'-manifest.json')).write_text(json.dumps(records[-1],indent=2))
        print(label,'exit',r.returncode,flush=True)
        if r.returncode:raise RuntimeError(label)
    cases=a.cases.split(',')
    if len(cases)!=2:return
    left,right=cases
    a=np.fromfile(OUT/(left+'.f32'),dtype=np.float32);b=np.fromfile(OUT/(right+'.f32'),dtype=np.float32)
    assert a.shape==b.shape
    delta=np.abs(a-b)
    result=dict(left=left,right=right,values=int(a.size),exact=bool(np.array_equal(a,b)),max_abs=float(delta.max()),mean_abs=float(delta.mean()),
      mismatched_values=int(np.count_nonzero(a!=b)),greedy_matches=int(np.sum(a.reshape(-1,154880).argmax(1)==b.reshape(-1,154880).argmax(1))),rows=int(a.size//154880))
    (OUT/(left+'-vs-'+right+'.json')).write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
