$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python='.venv/Scripts/python.exe'
$settings=@('--env','STRATA_GLM_RAM_GB=101','--env','MAYA_RAM_SHADOW_MB=24576',
 '--env','MAYA_RAM_SHADOW_EXPAND=1','--env','MAYA_DEFER_DEMOTIONS=layers',
 '--env','STRATA_GLM_POOL_GB=22','--env','STRATA_GLM_CPU_PLAN=011223344','--env','STRATA_GLM_USAGE=0')
# Same preserved retained engine for both sides; only graph replay changes.
& $python tools/profile_decode.py --mode stock --exe build/strata_retained_r1.exe --out docs/evidence/maya-demand/free-baseline --repeats 2 @settings
if($LASTEXITCODE){throw 'Baseline generation failed'}
& $python tools/profile_decode.py --mode stock --exe build/strata_retained_r1.exe --out docs/evidence/maya-demand/free-graphs --repeats 2 @settings --env STRATA_GLM_KDA_GRAPH=1
if($LASTEXITCODE){throw 'Graph generation failed'}
& $python tools/run_demand_lab.py long-baseline --exe maya_transfer_lab_retained_r1.exe --occupied 8192 --corpus
if($LASTEXITCODE){throw 'Baseline long context failed'}
& $python tools/run_demand_lab.py long-graphs --exe maya_transfer_lab_retained_r1.exe --occupied 8192 --corpus --env STRATA_GLM_KDA_GRAPH=1
if($LASTEXITCODE){throw 'Graph long context failed'}
& $python tools/profile_decode.py --mode stock --exe build/strata_retained_r1.exe --out docs/evidence/maya-demand/lifecycle-graphs --repeats 0 --cancel-smoke @settings --env STRATA_GLM_KDA_GRAPH=1 --env STRATA_GLM_TIER_DIAG=1 --env MAYA_SHADOW_VERIFY=1
if($LASTEXITCODE){throw 'Graph lifecycle failed'}
