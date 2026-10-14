"""Sequential c=4 A/B, identical prompts, 32K per slot, no speculative decode."""
import os,sys,json,time,subprocess,hashlib,threading,ctypes
from pathlib import Path
from run_transfer_lab import MemoryStatus
import strata_tokenizer as ST
R=Path(__file__).resolve().parents[1]; OUT=R/'docs/evidence/maya-upstream-117-c4'
PACK=Path('G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya')
TOK=ST.Tokenizer.from_dir(PACK/'tokenizer')
BASE=dict(STRATA_GLM_NO_MTP='1',STRATA_GLM_USAGE='0',STRATA_GLM_CPU_PLAN='011223344',STRATA_GLM_POOL_GB='22',STRATA_GLM_RAM_GB='101',STRATA_GLM_KDA_GRAPH='1',STRATA_GLM_SEQ_CTX='32768')
RETAINED=dict(MAYA_RAM_SHADOW_MB='24576',MAYA_RAM_SHADOW_EXPAND='1',MAYA_DEFER_DEMOTIONS='layers',MAYA_REUSE_POLICY='transition',MAYA_OVERLAP_BLOCKS='32',MAYA_OVERLAP_CAPTURED_ONLY='1',MAYA_CPU_EARLY='1',MAYA_CPU_PAUSE='1',MAYA_DOWN_BANDS='2',MAYA_MOE_SUFFIX_GRAPH='1',MAYA_ROUTE_WARP='1',MAYA_CPU_PIPELINE='1',MAYA_CPU_PHASE_LIMIT='8')
PROMPTS=[
 'Write a complete Python FastAPI backend for a task dashboard with projects, tasks, status filters, SQLite persistence, validation and tests. Include the code for every file.',
 'Write a complete React and TypeScript task dashboard with project navigation, task filters, editable status, loading states and responsive CSS. Include all component code.',
 'Write a complete Python CSV import service for a task dashboard, with schema validation, duplicate handling, error reporting, SQLite transactions and pytest tests. Include full implementation code.',
 'Write a complete TypeScript API client and React query hooks for a task dashboard, with authentication, retries, pagination, optimistic updates and tests. Include full implementation code.']
IDS=[TOK.encode('[gMASK]<sop><|user|>\n'+p+'<|assistant|>\n</think>',parse_special=True) for p in PROMPTS]
def run(branch, executable=None):
 out=OUT/branch; out.mkdir(parents=True,exist_ok=False)
 exe=Path(executable) if executable else ((R.parent/'project-maya-upstream-v1.1.7/build/strata.exe') if branch=='upstream' else R/'build/strata.exe')
 env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','MAYA_'))}; env.update(BASE)
 if branch=='retained':env.update(RETAINED)
 env['PATH']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;'+env['PATH']
 cmd=[str(exe),'--serve','--glm-pack',str(PACK),'--max-context','32768','--parallel','4','--prefill','auto','--expert-profile',str(R/'data/expert-profile-glm.bin'),'--short-read','0','--prompt-cache','0','--suffix-draft','0']
 (out/'manifest.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith(('STRATA_','MAYA_'))},binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),prompts=PROMPTS,prompt_ids=IDS,context_per_slot=32768,max_tokens=512,repeats=2),indent=2))
 results=[]; memory=[]; stopped=threading.Event()
 def monitor():
  start=time.monotonic()
  while not stopped.is_set():
   m=MemoryStatus();m.length=ctypes.sizeof(m)
   ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
   memory.append(dict(seconds=time.monotonic()-start,available_physical=m.avail_phys));stopped.wait(1)
 with (out/'engine.log').open('w') as log,(out/'gpu.csv').open('w') as glog,(out/'wire.log').open('w') as wire:
  gpu=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,clocks.gr,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=glog,stderr=subprocess.DEVNULL)
  p=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
  threading.Thread(target=monitor,daemon=True).start()
  watchdog=threading.Timer(900,lambda:p.kill() if p.poll() is None else None);watchdog.start()
  try:
   while True:
    line=p.stdout.readline();wire.write(line);wire.flush()
    if not line:raise RuntimeError('exit before READY')
    if line.startswith('READY'):
     assert 'multi=4' in line,line
     print(branch,line.strip(),flush=True);break
   for rep in range(2):
    start=time.perf_counter(); req={i:dict(prompt_ids=ids,output_ids=[],arrivals_ms=[]) for i,ids in enumerate(IDS)};done=set()
    p.stdin.write(''.join(f'GEN 512 temperature=0 rid={i} '+','.join(map(str,ids))+'\n' for i,ids in enumerate(IDS)));p.stdin.flush()
    while len(done)<4:
     line=p.stdout.readline();now=(time.perf_counter()-start)*1000;wire.write(line);wire.flush()
     if not line:raise RuntimeError('exit during requests')
     f=line.split()
     if not f:continue
     if f[0]=='ERR':raise RuntimeError(line)
     if f[0]=='T':q=req[int(f[1])];q['output_ids'].append(int(f[2]));q['arrivals_ms'].append(now)
     if f[0]=='DONE':
      i=int(f[1]);req[i].update(done=line.strip(),done_ms=now,prompt_ms=float(f[4]),decode_ms=float(f[5]),finish=f[6]);done.add(i)
    for q in req.values():
     a=q['arrivals_ms'];q['tokens']=len(a);q['intertoken_tps']=(len(a)-1)*1000/(a[-1]-a[0]);q['text']=TOK.decode(q['output_ids'])
    lo=max(q['arrivals_ms'][0] for q in req.values());hi=min(q['arrivals_ms'][-1] for q in req.values())
    assert hi>lo
    count=sum(sum(lo<t<=hi for t in q['arrivals_ms']) for q in req.values())
    row=dict(repeat=rep,requests=req,all_four_active_ms=hi-lo,all_four_active_tokens=count,aggregate_active_tps=count*1000/(hi-lo),end_to_end_tps=sum(q['tokens'] for q in req.values())*1000/max(q['done_ms'] for q in req.values()))
    results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2));print(branch,rep,row['aggregate_active_tps'],[q['tokens'] for q in req.values()],flush=True)
   p.stdin.write('QUIT\n');p.stdin.flush();wire.write(p.stdout.read());assert p.wait(timeout=60)==0
  finally:
   watchdog.cancel()
   if p.poll() is None:p.terminate();p.wait(timeout=30)
   stopped.set();gpu.terminate();gpu.wait(timeout=10);(out/'memory.json').write_text(json.dumps(memory,indent=2))
if __name__ == '__main__':
 for branch in ['upstream','retained']:run(branch)
