import subprocess,sys,json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
settings=['STRATA_GLM_RAM_GB=101','STRATA_GLM_KDA_GRAPH=1']
retained=['MAYA_RAM_SHADOW_MB=24576','MAYA_RAM_SHADOW_EXPAND=1','MAYA_DEFER_DEMOTIONS=layers','MAYA_REUSE_POLICY=transition','MAYA_OVERLAP_BLOCKS=32','MAYA_OVERLAP_CAPTURED_ONLY=1','MAYA_CPU_EARLY=1','MAYA_CPU_PAUSE=1','MAYA_DOWN_BANDS=2','MAYA_MOE_SUFFIX_GRAPH=1','MAYA_ROUTE_WARP=1','MAYA_CPU_PIPELINE=1','MAYA_CPU_PHASE_LIMIT=8']
for name,occupied in [('short',0),('8k',8192)]:
 for branch in (['upstream','retained'] if occupied==0 else ['retained','upstream']):
  exe=(r.parent/'project-maya-upstream-v1.1.7/build/maya_ab_192k.exe') if branch=='upstream' else r/'build/maya_ab_192k.exe'
  cmd=[sys.executable,str(r/'tools/run_upstream_ab.py'),branch+'-'+name,'--exe',str(exe),'--tokens','512','--hardware','--evidence-root','docs/evidence/maya-upstream-117-ab']
  if occupied:cmd+=['--occupied',str(occupied),'--corpus']
  for v in settings+(retained if branch=='retained' else []):cmd+=['--env',v]
  print('START',branch,name,flush=True)
  subprocess.run(cmd,cwd=r,check=True)
