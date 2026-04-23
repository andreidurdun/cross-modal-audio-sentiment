[CmdletBinding()]
param(
    [string]$PythonExe,
    [float[]]$Alphas = @(0.8, 0.9),
    [string]$StudentCheckpointBase = 'checkpoints\wavlm_audio_kd_text_en_text_fr_audio_alpha',
    [string]$ResultsBase = 'results\wavlm_audio_kd_text_en_text_fr_audio_alpha',
    [string]$EvalProfile = 'kd-validation',
    [int]$BatchSize = 8,
    [switch]$AllowOverwrite
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

if (-not $PythonExe) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = 'python'
    }
}

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path $Path)) {
        throw "$Description nu exista: $Path"
    }
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ''
    Write-Host ('=' * 90)
    Write-Host $Name
    Write-Host ('=' * 90)
    Write-Host "$PythonExe $($Arguments -join ' ')"

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pasul a esuat cu exit code ${LASTEXITCODE}: $Name"
    }
}

Assert-PathExists -Path $PythonExe -Description 'Interpretorul Python selectat'
Assert-PathExists -Path (Join-Path $projectRoot 'scripts\test_wavlm_audio.py') -Description 'Scriptul de testare WavLM'

foreach ($alpha in $Alphas) {
    $alphaTag = [string]::Format('{0:0.0}', $alpha).Replace('.', '')
    $studentCheckpointDir = "${StudentCheckpointBase}_a${alphaTag}"
    $resultsDir = "${ResultsBase}_a${alphaTag}"

    $checkpointPath = Join-Path $projectRoot $studentCheckpointDir
    $resultsPath = Join-Path $projectRoot $resultsDir

    Assert-PathExists -Path $checkpointPath -Description "Checkpoint model alpha=$alpha"

    if ((Test-Path (Join-Path $resultsPath 'test_results.json')) -and -not $AllowOverwrite.IsPresent) {
        throw "Rezultatele exista deja pentru alpha=$alpha la: $resultsPath`nRuleaza cu -AllowOverwrite daca vrei sa suprascrii."
    }

    Invoke-PythonStep -Name "Testare WavLM KD (alpha=$alpha)" -Arguments @(
        'scripts/test_wavlm_audio.py',
        '--checkpoint-dir', $studentCheckpointDir,
        '--output-dir', $resultsDir,
        '--eval-profile', $EvalProfile,
        '--batch-size', $BatchSize
    )
}

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline testare WavLM KD alpha sweep finalizat'
Write-Host ('=' * 90)
foreach ($alpha in $Alphas) {
    $alphaTag = [string]::Format('{0:0.0}', $alpha).Replace('.', '')
    $resultsDir = "${ResultsBase}_a${alphaTag}"
    Write-Host "  alpha=$alpha -> $resultsDir\test_results.json"
}
