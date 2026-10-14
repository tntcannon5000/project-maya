"""Bounded independent-engine control/DFlash2/MTP comparison, sequential GPU use."""
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import strata_tokenizer as ST
from bench_dflash_matrix import PROMPT
OUT=ROOT/'docs/evidence/dflash-investigation/neurall'
EXE=ROOT.parent/'llama-glm-neurall/build-local/bin/Release/llama-server.exe'
MODEL=Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf')
def http(path, data=None):
    req=urllib.request.Request('http://127.0.0.1:8097'+path,
      data=json.dumps(data).encode() if data is not None else None,
      headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='none,dflash2,mtp1');a=ap.parse_args()
    tok=ST.Tokenizer.from_dir(MODEL.parent/'pack-maya/tokenizer')
    ids=tok.encode('[gMASK]<sop><|user|>\n'+PROMPT+'<|assistant|>\n</think>',parse_special=True)
    for label,kind,width,draft in [
      ('none','none',0,None),
      ('dflash2','draft-dflash',2,'G:/Models/GLM-5.3-Flash-DFlash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf'),
      ('mtp1','draft-mtp',1,'G:/Models/GLM-5.3-Flash-MTP/GLM-5.3-Flash-MTP-Q4_K.gguf')]:
        if label not in a.only.split(','):continue
        out=OUT/label;out.mkdir(parents=True,exist_ok=False)
        cmd=[str(EXE),'-m',str(MODEL),'--host','127.0.0.1','--port','8097','-np','1','-c','131072',
             '-ngl','all','--cpu-moe','--moe-expert-cache','-1','-nr','-t','16','-ub','2048','-b','2048',
             '--load-mode','auto','-ctk','q8_0','-ctv','q8_0','--spec-type',kind]
        if draft:cmd+=['-md',draft,'--spec-draft-ngl','all','--spec-draft-n-max',str(width),'--spec-draft-p-min','0']
        env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','MAYA_','STRATA_'))}
        env['PATH']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;'+env['PATH']
        (out/'manifest.json').write_text(json.dumps(dict(command=cmd,prompt_ids=ids,source='neurall local 9d35489'),indent=2))
        with (out/'engine.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=EXE.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                until=time.monotonic()+600
                while True:
                    if p.poll() is not None:raise RuntimeError(f'{label} load exit {p.returncode}')
                    try:
                        if http('/health').get('status')=='ok':break
                    except Exception:pass
                    if time.monotonic()>until:raise TimeoutError('load timeout')
                    time.sleep(1)
                print(label,'READY',flush=True)
                rows=[]
                for stage in ['warmup','measure']:
                    r=http('/completion',dict(prompt=ids,n_predict=192,temperature=0,seed=1,cache_prompt=False,repeat_penalty=1.0))
                    rows.append(dict(stage=stage,response=r));(out/'requests.json').write_text(json.dumps(rows,indent=2))
                    print(label,stage,r.get('timings'),flush=True)
            finally:
                if p.poll() is None:p.terminate();p.wait(timeout=60)
if __name__=='__main__':main()
