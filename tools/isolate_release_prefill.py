"""Separate upstream prefill arithmetic changes from merge regressions."""
import subprocess,sys
from pathlib import Path
from run_c4_ab import RETAINED
R=Path(__file__).resolve().parents[1]
for name,exe,settings in [('gpu-merged-glu-off',R/'build/maya_ab_192k.exe',dict(RETAINED,STRATA_GLM_MMQ_GLU='0')),('gpu-stock117',R.parent/'project-maya-upstream-v1.1.7/build/maya_ab_192k.exe',{})]:
 cmd=[sys.executable,str(R/'tools/run_upstream_ab.py'),name,'--exe',str(exe),'--tokens','128','--occupied','8192','--corpus','--evidence-root','docs/evidence/maya-release-integration']
 for k,v in dict(settings,STRATA_GLM_RAM_GB='101',STRATA_GLM_KDA_GRAPH='1',STRATA_GLM_CPU_PLAN='000000000').items():cmd+=['--env',k+'='+v]
 subprocess.run(cmd,cwd=R,check=True)
