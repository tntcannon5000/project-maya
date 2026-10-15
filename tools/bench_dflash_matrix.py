"""Short c=1 drafter-quant/width screen; no persistent launcher changes."""
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST

OUT = ROOT / 'docs/evidence/dflash-quant-width'
DRAFT = Path('G:/Models/GLM-5.3-Flash-DFlash2')
PROMPT = ('Write a complete Python FastAPI backend for a task dashboard with projects, tasks, '
          'status filters, SQLite persistence, validation and tests. Include the code for every file.')

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--exe',type=Path,default=ROOT/'build/strata.exe')
    ap.add_argument('--stock',action='store_true')
    ap.add_argument('--widths',default='4,1,7,2,6,3,5', help='Comma-separated forced draft widths; empty skips forced policies')
    ap.add_argument('--quant',nargs='+',default=['Q4_K_M','Q5_K_M','Q8_0'])
    ap.add_argument('--env',action='append',default=[])
    ap.add_argument('--cancel-smoke',action='store_true')
    ap.add_argument('--prompt-file',type=Path)
    ap.add_argument('--tokens',type=int,default=192)
    ap.add_argument('--confidence-max-drafts',type=int,default=0,
                    help='Append the upstream single-GPU confidence policy (0.95 cut, no mandatory drafts)')
    a=ap.parse_args();OUT=a.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'maya-retained-v117.local.json').read_text())
    tok = ST.Tokenizer.from_dir(Path(cfg['tokenizer']))
    prompt=a.prompt_file.read_text(encoding='utf-8') if a.prompt_file else PROMPT
    if a.tokens < 1:ap.error('--tokens must be positive')
    if not 0 <= a.confidence_max_drafts <= 7:ap.error('--confidence-max-drafts must be 0..7')
    widths = [int(x) for x in a.widths.split(',') if x.strip()]
    if any(w < 1 or w > 7 for w in widths):ap.error('--widths entries must be 1..7')
    ids = tok.encode('[gMASK]<sop><|user|>\n'+prompt+'<|assistant|>\n</think>', parse_special=True)
    exe = a.exe.resolve()
    for quant in a.quant:
        out = OUT/quant
        out.mkdir(exist_ok=False)
        stats = out/'rounds.txt'
        env = {k:v for k,v in os.environ.items() if not k.startswith(('MAYA_', 'STRATA_'))}
        env.update(cfg['env'])
        if a.stock:
            env={k:v for k,v in env.items() if not k.startswith('MAYA_')}
        for setting in a.env:
            k,v=setting.split('=',1);env[k]=v
        env['STRATA_GLM_DFLASH_STATS'] = str(stats)
        env['STRATA_GLM_DFLASH_PROF'] = '1'
        env['PATH'] = ';'.join(cfg['lib_dirs'])+';'+env['PATH']
        draft = DRAFT/f'GLM-5.3-Flash-DFlash2-{quant}.gguf'
        cmd = [str(exe), '--serve', *cfg['args'], '--parallel', '1', '--dflash', str(draft),
               '--spec-min-p', '0', '--short-read', '0', '--prompt-cache', '0', '--kv-streaming', 'off']
        manifest = dict(command=cmd, environment={k:v for k,v in env.items() if k.startswith(('STRATA_', 'MAYA_'))}, prompt=prompt, prompt_ids=ids,
                        binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                        drafter_sha256=hashlib.file_digest(draft.open('rb'), 'sha256').hexdigest(),
                        context_capacity=131072, occupied_prompt_tokens=len(ids), output_cap=a.tokens,
                        width_order=widths, confidence_max_drafts=a.confidence_max_drafts,
                        stats_columns=['offered','accepted','draft_ms','verify_ms','cpu_experts','vram_misses','confidence...'])
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        rows=[]
        with (out/'engine.log').open('w') as log, (out/'wire.log').open('w') as wire, (out/'gpu.csv').open('w') as glog:
            gpu = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.gr,clocks.mem,temperature.gpu,power.draw','--format=csv','-l','1'], stdout=glog, stderr=subprocess.DEVNULL)
            p = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            watchdog=threading.Timer(900, p.kill); watchdog.start()
            def read():
                s=p.stdout.readline(); wire.write(s); wire.flush()
                if not s: raise RuntimeError(f'{quant}: engine exited {p.poll()}')
                if s.startswith('ERR'): raise RuntimeError(s)
                return s
            try:
                while not read().startswith('READY'): pass
                print(quant,'READY',flush=True)
                requests=[('warmup',0,a.tokens),('control',0,a.tokens)]+[('matrix',w,a.tokens) for w in manifest['width_order']]
                if a.confidence_max_drafts:requests.append(('confidence',a.confidence_max_drafts,a.tokens))
                if a.cancel_smoke:requests += [('cancel',4,192),('recovery',2,32)]
                for name,width,n in requests:
                    before=len(stats.read_text().splitlines()) if stats.exists() else 0
                    start=time.perf_counter(); output=[]; arrivals=[]
                    policy = 'spec_min_p=0.95 spec_n_min=0 spec_cum=0' if name=='confidence' else f'spec_min_p=0 spec_n_min={width} spec_cum=0'
                    p.stdin.write(f'GEN {n} temperature=0 spec={width+1} {policy} '+','.join(map(str,ids))+'\n');p.stdin.flush()
                    while True:
                        s=read()
                        if s.startswith('T '):
                            output.append(int(s.split()[1]));arrivals.append((time.perf_counter()-start)*1000)
                            if name=='cancel' and len(output)==8:
                                p.stdin.write('STOP\n');p.stdin.flush()
                        if s.startswith('DONE'): break
                    fields=s.split()
                    if name=='cancel' and 'cancel' not in fields:raise RuntimeError('STOP did not cancel')
                    rounds=[list(map(float,x.split())) for x in stats.read_text().splitlines()[before:]] if stats.exists() else []
                    offered=sum(r[0] for r in rounds); accepted=sum(r[1] for r in rounds)
                    if width and (not rounds or max(r[0] for r in rounds)>width or (name!='confidence' and max(r[0] for r in rounds)!=width)):
                        raise RuntimeError(f'Width {width} did not engage: {rounds[:2]}')
                    row=dict(kind=name,width=width,draft_policy=policy,tokens=int(fields[1]),prompt_ms=float(fields[3]),decode_ms=float(fields[4]),
                             done=s.strip(),tps=int(fields[1])*1000/float(fields[4]),output_ids=output,
                             text=tok.decode(output),arrivals_ms=arrivals,rounds=rounds,offered=offered,accepted=accepted,
                             acceptance=accepted/offered if offered else None,
                             draft_ms=sum(r[2] for r in rounds),verify_ms=sum(r[3] for r in rounds))
                    rows.append(row);(out/'requests.json').write_text(json.dumps(rows,indent=2))
                    print(quant,name,width,round(row['tps'],2),'TPS',row['acceptance'],flush=True)
                p.stdin.write('QUIT\n');p.stdin.flush()
                wire.write(p.stdout.read())
                if p.wait(timeout=120): raise RuntimeError('nonzero exit')
            finally:
                watchdog.cancel()
                if p.poll() is None: p.kill();p.wait()
                gpu.terminate();gpu.wait()

if __name__ == '__main__': main()
