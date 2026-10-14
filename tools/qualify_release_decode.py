"""GPU-only serial prompt avoids the unstable long batched-prefill control."""
import subprocess,sys
from pathlib import Path
from run_c4_ab import RETAINED
R=Path(__file__).resolve().parents[1]
for label,exe,settings in [('before',R/'build/maya_ab_pre117.exe',RETAINED),('merged',R/'build/maya_ab_192k.exe',RETAINED)]:
 cmd=[sys.executable,str(R/'tools/run_upstream_ab.py'),'gpu-short-'+label,'--exe',str(exe),'--tokens','128','--evidence-root','docs/evidence/maya-release-integration']
 for k,v in dict(settings,STRATA_GLM_RAM_GB='101',STRATA_GLM_KDA_GRAPH='1',STRATA_GLM_CPU_PLAN='000000000',MAYA_SHADOW_VERIFY='1').items():cmd+=['--env',k+'='+v]
 subprocess.run(cmd,cwd=R,check=True)
