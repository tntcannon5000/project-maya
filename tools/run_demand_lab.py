"""Campaign two always starts from the retained RAM/scheduling configuration."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
args=[sys.executable,str(root/'tools/run_transfer_lab.py'),*sys.argv[1:],
      '--evidence-root','docs/evidence/maya-demand','--hardware',
      '--env','STRATA_GLM_TIMING=1',
      '--env','STRATA_GLM_RAM_GB=101','--env','MAYA_RAM_SHADOW_MB=24576',
      '--env','MAYA_RAM_SHADOW_EXPAND=1','--env','MAYA_DEFER_DEMOTIONS=layers']
raise SystemExit(subprocess.call(args,cwd=root))
