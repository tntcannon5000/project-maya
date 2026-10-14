"""One reversed-order confirmation of the small short-context difference."""
import subprocess,sys
from pathlib import Path
from run_c4_ab import RETAINED
R=Path(__file__).resolve().parents[1]
for branch in ['merged','before']:
 exe='maya_ab_pre117.exe' if branch=='before' else 'maya_ab_192k.exe'
 cmd=[sys.executable,str(R/'tools/run_upstream_ab.py'),'confirm-'+branch,'--exe',exe,'--tokens','256','--hardware','--evidence-root','docs/evidence/maya-release-integration']
 for k,v in dict(STRATA_GLM_RAM_GB='101',STRATA_GLM_KDA_GRAPH='1',**RETAINED).items():cmd+=['--env',k+'='+v]
 subprocess.run(cmd,cwd=R,check=True)
