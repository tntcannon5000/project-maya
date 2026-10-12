$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = '.venv/Scripts/python.exe'
$settings = @('--env','STRATA_GLM_RAM_GB=101','--env','MAYA_RAM_SHADOW_MB=24576',
    '--env','MAYA_RAM_SHADOW_EXPAND=1','--env','MAYA_DEFER_DEMOTIONS=layers')
& $python tools/run_transfer_lab.py retained-clean @settings
if ($LASTEXITCODE) { throw 'Clean fixed-input validation failed' }
& $python tools/run_transfer_lab.py gpu-clean-cold --repeats 1 @settings --env STRATA_GLM_CPU_PLAN=000000000 --env STRATA_GLM_WARM=0 --env MAYA_SHADOW_VERIFY=1
if ($LASTEXITCODE) { throw 'Cold GPU-only ownership validation failed' }
& $python tools/profile_decode.py --mode stock --exe build/strata.exe --out docs/evidence/maya-transfer/lifecycle-retained --repeats 0 --cancel-smoke @settings --env STRATA_GLM_POOL_GB=22 --env STRATA_GLM_CPU_PLAN=011223344 --env STRATA_GLM_USAGE=0 --env STRATA_GLM_TIER_DIAG=1 --env MAYA_SHADOW_VERIFY=1
if ($LASTEXITCODE) { throw 'Cancellation/recovery validation failed' }
