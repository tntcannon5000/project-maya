"""Focused pre-merge/merged qualification; two occupied contexts, fixed history."""
import subprocess,sys
from pathlib import Path
from run_c4_ab import RETAINED
R=Path(__file__).resolve().parents[1]
for task,occupied,order in [('short',0,['before','merged']),('8k',8192,['merged','before'])]:
 for branch in order:
  exe='maya_ab_pre117.exe' if branch=='before' else 'maya_ab_192k.exe'
  cmd=[sys.executable,str(R/'tools/run_upstream_ab.py'),branch+'-'+task,'--exe',exe,'--tokens','256','--hardware','--evidence-root','docs/evidence/maya-release-integration']
  if occupied:cmd+=['--occupied',str(occupied),'--corpus']
  for k,v in dict(STRATA_GLM_RAM_GB='101',STRATA_GLM_KDA_GRAPH='1',**RETAINED).items():cmd+=['--env',k+'='+v]
  print('START',branch,task,flush=True)
  subprocess.run(cmd,cwd=R,check=True)
