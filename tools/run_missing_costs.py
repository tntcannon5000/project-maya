"""Bounded missing-expert experiments from the validated three-costs candidate."""
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--baseline', action='store_true')
p.add_argument('--env', action='append', default=[])
p.add_argument('--occupied', type=int, default=0)
p.add_argument('--task', choices=['csv','web','explanation'], default='csv')
p.add_argument('--tokens', type=int)
p.add_argument('--trace', action='store_true')
p.add_argument('--route-log', action='store_true')
a = p.parse_args()
settings = ['STRATA_GLM_RAM_GB=101','MAYA_RAM_SHADOW_MB=24576','MAYA_RAM_SHADOW_EXPAND=1',
 'MAYA_DEFER_DEMOTIONS=layers','STRATA_GLM_KDA_GRAPH=1','MAYA_REUSE_POLICY=transition',
 'MAYA_OVERLAP_BLOCKS=32','MAYA_OVERLAP_CAPTURED_ONLY=1',
 'MAYA_CPU_EARLY=1','MAYA_CPU_PAUSE=1','MAYA_DOWN_BANDS=2',
 'MAYA_MOE_SUFFIX_GRAPH=1','MAYA_ROUTE_WARP=1','MAYA_CPU_PIPELINE=1','MAYA_CPU_PHASE_LIMIT=8']
cmd = [sys.executable, str(ROOT/'tools/run_transfer_lab.py'), a.name,
 '--evidence-root','docs/evidence/maya-missing-costs','--task',a.task,'--hardware']
if a.baseline: cmd += ['--exe','maya_transfer_lab_missing_baseline.exe']
if a.occupied: cmd += ['--occupied',str(a.occupied),'--corpus']
if a.tokens: cmd += ['--tokens',str(a.tokens)]
if a.trace: cmd += ['--trace']; settings += ['STRATA_GLM_TIMING=1']
if a.route_log: cmd += ['--route-log']
for setting in settings + a.env: cmd += ['--env',setting]
subprocess.run(cmd, cwd=ROOT, check=True)
