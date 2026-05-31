# Run from the repository root, e.g. C:\GitHub\Research\pc-cc
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_pccc_tuning.ps1

$ErrorActionPreference = "Stop"

$JuliaFile = "sos/pccc_synth_platoon_sos.jl"
$ResultsDir = "results"
$LogDir = Join-Path $ResultsDir "pccc_tuning_logs"
$BackupFile = "$JuliaFile.bak_tuning"

if (!(Test-Path $JuliaFile)) {
    throw "Cannot find $JuliaFile. Run this script from the repository root."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Copy-Item $JuliaFile $BackupFile -Force

function Set-PcccConstants {
    param(
        [double]$WfDec,
        [double]$P2,
        [double]$P3A,
        [double]$P3B
    )

    $txt = Get-Content $JuliaFile -Raw
    $txt = [regex]::Replace($txt, 'const WF_DEC\s*=\s*[0-9.eE+-]+', ('const WF_DEC = {0:E1}' -f $WfDec))
    $txt = [regex]::Replace($txt, 'const IMP_MULT_P2\s*=\s*[0-9.eE+-]+', ('const IMP_MULT_P2   = {0:E1}' -f $P2))
    $txt = [regex]::Replace($txt, 'const IMP_MULT_P3_A\s*=\s*[0-9.eE+-]+', ('const IMP_MULT_P3_A = {0:E1}' -f $P3A))
    $txt = [regex]::Replace($txt, 'const IMP_MULT_P3_B\s*=\s*[0-9.eE+-]+', ('const IMP_MULT_P3_B = {0:E1}' -f $P3B))
    Set-Content $JuliaFile $txt -NoNewline
}

$Trials = @(
    @{Name="A_wf1e-3"; WF=1.0e-3; P2=1.0e-3; P3A=0.0;    P3B=0.0},
    @{Name="B_wf1e-3"; WF=1.0e-3; P2=0.0;    P3A=1.0e-3; P3B=1.0e-3},
    @{Name="C_wf1e-3"; WF=1.0e-3; P2=0.0;    P3A=0.0;    P3B=0.0},
    @{Name="A_wf1e-4"; WF=1.0e-4; P2=1.0e-3; P3A=0.0;    P3B=0.0},
    @{Name="B_wf1e-4"; WF=1.0e-4; P2=0.0;    P3A=1.0e-3; P3B=1.0e-3},
    @{Name="C_wf1e-4"; WF=1.0e-4; P2=0.0;    P3A=0.0;    P3B=0.0},
    @{Name="A_wf1e-5"; WF=1.0e-5; P2=1.0e-3; P3A=0.0;    P3B=0.0},
    @{Name="B_wf1e-5"; WF=1.0e-5; P2=0.0;    P3A=1.0e-3; P3B=1.0e-3},
    @{Name="C_wf1e-5"; WF=1.0e-5; P2=0.0;    P3A=0.0;    P3B=0.0}
)

foreach ($trial in $Trials) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running $($trial.Name): WF=$($trial.WF), P2=$($trial.P2), P3A=$($trial.P3A), P3B=$($trial.P3B)"
    Write-Host "============================================================"

    Set-PcccConstants -WfDec $trial.WF -P2 $trial.P2 -P3A $trial.P3A -P3B $trial.P3B

    $logFile = Join-Path $LogDir ("$($trial.Name).log")
    julia --project=. $JuliaFile 2>&1 | Tee-Object -FilePath $logFile

    if (Test-Path (Join-Path $ResultsDir "pccc_synth_platoon_sos_status.json")) {
        Copy-Item (Join-Path $ResultsDir "pccc_synth_platoon_sos_status.json") (Join-Path $LogDir ("$($trial.Name)_status.json")) -Force
    }
    if (Test-Path (Join-Path $ResultsDir "pccc_synth_platoon_sos.json")) {
        Copy-Item (Join-Path $ResultsDir "pccc_synth_platoon_sos.json") (Join-Path $LogDir ("$($trial.Name)_certificate.json")) -Force
    }
    if (Test-Path (Join-Path $ResultsDir "pccc_synth_platoon_sos.jld2")) {
        Copy-Item (Join-Path $ResultsDir "pccc_synth_platoon_sos.jld2") (Join-Path $LogDir ("$($trial.Name)_certificate.jld2")) -Force
    }

    $logText = Get-Content $logFile -Raw
    if ($logText -match "Feasible SOS certificate found") {
        Write-Host ""
        Write-Host "SUCCESS: feasible certificate found in trial $($trial.Name)."
        Write-Host "Kept patched constants in $JuliaFile. Original backup: $BackupFile"
        exit 0
    }
}

Write-Host ""
Write-Host "No feasible certificate found in the scripted trials. Logs are in $LogDir"
Write-Host "Original backup: $BackupFile"
exit 1
