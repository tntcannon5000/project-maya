"""Controlled GPU-backend replay and server lifecycle after release integration."""
import subprocess,sys
from pathlib import Path
from run_c4_ab import RETAINED
R=Path(__file__).resolve().parents[1]
for label,exe in [('before','maya_ab_pre117.exe'),('merged','maya_ab_192k.exe')]:
 cmd=[sys.executable,str(R/'tools/run_upstream_ab.py'),'gpu-'+label,'--exe',exe,'--tokens','128','--occupied','8192','--corpus','--evidence-root','docs/evidence/maya-release-integration']
 env=dict(RETAINED,STRATA_GLM_RAM_GB='101',STRATA_GLM_KDA_GRAPH='1',STRATA_GLM_CPU_PLAN='000000000',MAYA_SHADOW_VERIFY='1')
 for k,v in env.items():cmd+=['--env',k+'='+v]
 subprocess.run(cmd,cwd=R,check=True)
cmd=[sys.executable,str(R/'tools/profile_decode.py'),'--mode','stock','--exe',str(R/'build/strata.exe'),'--out','docs/evidence/maya-release-integration/lifecycle','--repeats','0','--cancel-smoke']
for k,v in dict(RETAINED,STRATA_GLM_RAM_GB='101',STRATA_GLM_POOL_GB='22',STRATA_GLM_CPU_PLAN='011223344',STRATA_GLM_USAGE='0',MAYA_SHADOW_VERIFY='1').items():cmd+=['--env',k+'='+v]
subprocess.run(cmd,cwd=R,check=True)
