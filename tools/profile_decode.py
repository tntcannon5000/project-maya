"""Bounded, sequential stock/trace comparison. No server or HTTP latency."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as SV
import strata_tokenizer as ST

PROMPTS = [
    ("csv", "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring."),
    ("explanation", "Explain how photosynthesis works, step by step, for a high-school student."),
    ("web", "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript."),
]
NSYS = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stock', 'trace', 'worker'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--exe', type=Path)
    ap.add_argument('--env', action='append', default=[])
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'trace':
        cmd = [str(NSYS), 'profile', '--show-output=true', '--trace=cuda,nvtx', '--sample=none',
               '--cpuctxsw=none', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
               '-o', str(out/'timeline'), sys.executable, str(Path(__file__).resolve()),
               '--mode', 'worker', '--out', str(out), '--repeats', str(a.repeats), '--tokens', str(a.tokens)]
        if a.exe:cmd += ['--exe',str(a.exe.resolve())]
        for setting in a.env:cmd += ['--env',setting]
        with (out/'profiler-output.txt').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        subprocess.run([str(NSYS), 'export', '--type', 'sqlite', '--output', str(out/'timeline.sqlite'),
                        str(out/'timeline.nsys-rep')], check=True)
        return
    cfg = json.loads((ROOT/'maya-gsq-q4attn.json').read_text(encoding='utf-8-sig'))
    tok = ST.Tokenizer.from_dir(Path(cfg['tokenizer']))
    exe = ROOT/'build'/('strata-stock-profile-reference.exe' if a.mode == 'stock' else 'strata.exe')
    if a.exe: exe=a.exe.resolve()
    cmd = [str(exe), '--serve', *SV.engine_args(cfg), '--short-read', '0', '--prompt-cache', '0', '--suffix-draft', '0']
    env = SV.child_env(cfg)
    for k in list(env):
        if k.startswith(('STRATA_GLM_PROF', 'MAYA_TIMELINE', 'STRATA_GLM_NSYS')): del env[k]
    env['STRATA_GLM_NO_MTP'] = '1'
    for setting in a.env:
        key,value=setting.split('=',1);env[key]=value
    if a.mode == 'worker':
        env['MAYA_TIMELINE'] = '1'
    manifest = dict(mode=a.mode, command=cmd, repeats=a.repeats, tokens=a.tokens,
                    binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), config=cfg,
                    environment={k:v for k,v in env.items() if k.startswith(('STRATA_', 'MAYA_'))})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results=[]
    with (out/'engine.log').open('w', encoding='utf-8') as log:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                             text=True, bufsize=1, cwd=ROOT, env=env)
        try:
            while True:
                line=p.stdout.readline()
                if not line: raise RuntimeError('engine exited before READY')
                if line.startswith('READY'): break
            print('READY', flush=True)
            requests=[('warmup', 'Say hello in five languages.', 64, -1)]
            requests += [(name, q, a.tokens, repeat) for repeat in range(a.repeats) for name,q in PROMPTS]
            for name,q,n,repeat in requests:
                ids=tok.encode('[gMASK]<sop><|user|>\n'+q+'<|assistant|>\n</think>', parse_special=True)
                start=time.perf_counter_ns(); arrivals=[]; output=[]
                p.stdin.write(f'GEN {n} temperature=0 '+','.join(map(str,ids))+'\n'); p.stdin.flush()
                while True:
                    line=p.stdout.readline()
                    if not line: raise RuntimeError('engine exited during request')
                    if line.startswith('T '):
                        arrivals.append((time.perf_counter_ns()-start)/1e6); output.append(int(line.split()[1]))
                    if line.startswith('ERR'): raise RuntimeError(line)
                    if line.startswith('DONE'): break
                f=line.split()
                row=dict(task=name, repeat=repeat, prompt_ids=ids, output_ids=output,
                         arrivals_ms=arrivals, done=line.strip(), prompt_ms=float(f[3]),
                         decode_ms=float(f[4]), tokens=int(f[1]))
                results.append(row)
                (out/'requests.json').write_text(json.dumps(results, indent=2))
                print(name, repeat, row['tokens'], row['decode_ms'], flush=True)
            p.stdin.write('QUIT\n'); p.stdin.flush()
            tail=p.stdout.read()
            (out/'worker-output.txt').write_text(tail)
            rc=p.wait(timeout=180)
            if rc: raise RuntimeError(f'child exit {rc}')
        finally:
            if p.poll() is None:
                p.terminate(); p.wait(timeout=30)
    if a.mode == 'trace':
        subprocess.run([str(NSYS), 'export', '--type', 'sqlite', '--output', str(out/'timeline.sqlite'),
                        str(out/'timeline.nsys-rep')], check=True)

if __name__ == '__main__': main()
