"""One controlled fixed-input experiment, with immutable manifest and stdout/stderr."""
import argparse,hashlib,json,os,subprocess,ctypes,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
        ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--exe',default='maya_transfer_lab.exe')
    p.add_argument('--env',action='append',default=[]);p.add_argument('--repeats',default=2,type=int)
    p.add_argument('--occupied',default=0,type=int)
    p.add_argument('--tokens',type=int,help='Bound scored continuation for mechanism-only diagnostics')
    p.add_argument('--corpus',action='store_true',help='Fill context with frozen baseline Python source, not repeated tokens')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--evidence-root',default='docs/evidence/maya-transfer')
    p.add_argument('--task',choices=['csv','explanation','web'],default='csv')
    p.add_argument('--route-log',action='store_true')
    p.add_argument('--hardware',action='store_true')
    a=p.parse_args();out=ROOT/a.evidence_root/a.name;out.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'docs/evidence/maya-timeline/stock/requests.json').read_text())
    row=next(x for x in source if x['task']==a.task)
    prompt=row['prompt_ids']; continuation=row['output_ids']
    if a.tokens is not None:
        if not 0<a.tokens<=len(continuation):p.error('tokens must fit the frozen continuation')
        continuation=continuation[:a.tokens]
    if a.occupied:
        # Reproducible code tokens as filler; natural short prompt follows. This is an occupied-context fixture.
        prompt=(continuation*((a.occupied//len(continuation))+1))[:a.occupied-len(prompt)]+prompt
        if a.corpus:
            sys.path.insert(0,str(ROOT));import strata_tokenizer as ST
            tok=ST.Tokenizer.from_dir(Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya/tokenizer'))
            names=['tools/analyze_decode_timeline.py','serve/server.py','tools/profile_decode.py']
            text='\n'.join(subprocess.check_output(['git','show','615973d:'+name],cwd=ROOT,text=True,encoding='utf-8') for name in names)
            head=tok.encode('[gMASK]<sop><|user|>\nHere are Python modules from a local inference project:\n',parse_special=True)
            tail=tok.encode('\nAdd a Python CSV-to-dictionary parsing helper, with type hints and a docstring.\n<|assistant|>\n</think>',parse_special=True)
            body=tok.encode(text,parse_special=False)
            prompt=head+body[:a.occupied-len(head)-len(tail)]+tail
            assert len(prompt)==a.occupied
    ids=prompt+continuation
    (out/'tokens.txt').write_text(' '.join(map(str,ids)))
    env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','MAYA_'))}
    env.update(STRATA_GLM_NO_MTP='1',STRATA_GLM_USAGE='0',STRATA_GLM_CPU_PLAN='011223344',
        STRATA_GLM_POOL_GB='22',STRATA_GLM_RAM_GB='87.4',STRATA_GLM_PREFILL='auto',STRATA_GLM_TIER_DIAG='1')
    env['PATH']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;'+env['PATH']
    for v in a.env:
        k,val=v.split('=',1);env[k]=val
    if a.trace:env['MAYA_TIMELINE']='1'
    if a.route_log:env['STRATA_GLM_ROUTE_LOG']=str(out/'routes')
    exe=ROOT/'build'/a.exe
    pack=Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya')
    model=pack.parent/'GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf'
    weights=dict(path=str(model),size=model.stat().st_size,mtime_ns=model.stat().st_mtime_ns,
        pack_manifests={n:hashlib.sha256((pack/n).read_bytes()).hexdigest() for n in ('index.txt','native_experts.txt')},
        expert_profile_sha256=hashlib.sha256((ROOT/'data/expert-profile-glm.bin').read_bytes()).hexdigest())
    weights['experiment_files']={key:dict(path=str(ROOT/env[key]),sha256=hashlib.sha256((ROOT/env[key]).read_bytes()).hexdigest())
        for key in ('MAYA_PROFILE_OVERRIDE','MAYA_SLOT_MAP','MAYA_CPU_PLAN_MAP') if key in env}
    cmd=[str(exe),'G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya',str(out/'tokens.txt'),str(len(prompt)),str(out/'fixed'),str(a.repeats)]
    nsys=r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe'
    if a.trace:cmd=[nsys,'profile','--show-output=true','--trace=cuda,nvtx','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
        '--capture-range-end=stop','-o',str(out/'timeline'),*cmd]
    (out/'manifest.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('STRATA_','MAYA_'))},
        binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),tokens_sha256=hashlib.sha256((out/'tokens.txt').read_bytes()).hexdigest(),weights=weights),indent=2))
    with (out/'engine.log').open('w') as log:
        gpu_log=(out/'gpu.csv').open('w') if a.hardware else None
        gpu=None
        if gpu_log:
            gpu=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,clocks.gr,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=gpu_log,stderr=subprocess.DEVNULL)
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stderr=log,stdout=log)
        memory=[];start=time.monotonic()
        while child.poll() is None:
            m=MemoryStatus();m.length=ctypes.sizeof(m)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                memory.append(dict(seconds=time.monotonic()-start,available_physical=m.avail_phys,available_commit=m.avail_page))
            time.sleep(1)
        (out/'memory.json').write_text(json.dumps(memory,indent=2))
        if gpu:gpu.terminate();gpu.wait(timeout=10);gpu_log.close()
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,cmd)
    if a.trace:
        subprocess.run([nsys,'export','--type','sqlite','--output',str(out/'timeline.sqlite'),str(out/'timeline.nsys-rep')],check=True)
    print('complete',a.name,flush=True)
if __name__=='__main__':main()
