"""Run one bounded experiment with the retained campaign settings."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--env',action='append',default=[])
p.add_argument('--baseline',action='store_true');p.add_argument('--trace',action='store_true')
p.add_argument('--task',default='csv',choices=['csv','web','explanation']);p.add_argument('--occupied',type=int,default=0)
p.add_argument('--repeats',type=int,default=2)
a=p.parse_args()
settings=['STRATA_GLM_RAM_GB=101','MAYA_RAM_SHADOW_MB=24576','MAYA_RAM_SHADOW_EXPAND=1',
 'MAYA_DEFER_DEMOTIONS=layers','STRATA_GLM_KDA_GRAPH=1','MAYA_REUSE_POLICY=transition',
 'MAYA_OVERLAP_BLOCKS=32','MAYA_OVERLAP_CAPTURED_ONLY=1']
cmd=[sys.executable,str(ROOT/'tools/run_transfer_lab.py'),a.name,'--evidence-root','docs/evidence/maya-three-costs',
 '--task',a.task,'--repeats',str(a.repeats),'--hardware']
if a.baseline:cmd+=['--exe','maya_transfer_lab_three_costs_baseline.exe']
if a.trace:cmd+=['--trace'];settings+=['STRATA_GLM_TIMING=1']
if a.occupied:cmd+=['--occupied',str(a.occupied),'--corpus']
for setting in settings+a.env:cmd+=['--env',setting]
subprocess.run(cmd,cwd=ROOT,check=True)
