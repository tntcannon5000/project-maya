param(
    [ValidateSet('baseline','placement','lru','batch','cpu','predict')]
    [string[]]$Cases=@('baseline')
)
$ErrorActionPreference='Stop'
if($Cases.Count -gt 2){throw 'Select one or two targeted cases per iteration.'}
Set-Location (Split-Path $PSScriptRoot -Parent)
$python='.venv/Scripts/python.exe'
# Event-stage diagnostics only, not scored TPS. Skip exactly 30 prompt forwards.
# Graph/overlap candidates already have node-level Nsight traces.
$common=@('--exe','maya_transfer_lab_demand_prototypes.exe','--repeats','1','--env','STRATA_GLM_PROF=30')
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
foreach($case in $Cases){
    $extra=switch($case){
        placement {@('--env','MAYA_SLOT_MAP=docs/evidence/maya-demand/trained-slots.txt','--env','MAYA_PROFILE_OVERRIDE=docs/evidence/maya-demand/trained-profile.bin')}
        lru {@('--env','STRATA_GLM_VRAM_EVICT=lru')}
        batch {@('--env','MAYA_BATCH_PULL=1')}
        cpu {@('--env','MAYA_CPU_PLAN_MAP=docs/evidence/maya-demand/cpu-wait-plan.txt')}
        predict {@('--env','STRATA_GLM_PREFETCH_N=1','--env','STRATA_GLM_PREFETCH_RANK=1','--env','STRATA_GLM_PREFETCH_AT=cpu')}
        default {@()}
    }
    & $python tools/run_demand_lab.py "stages-$case-$stamp" @common @extra
    if($LASTEXITCODE){throw "Stage $case failed"}
}
