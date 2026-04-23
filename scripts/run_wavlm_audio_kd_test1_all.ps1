[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$CheckpointPattern = 'wavlm_audio_kd*',
    [string]$OutputRoot = 'results\test1_eval_kd_all',
    [string]$Partition = 'Test1',
    [string]$EvalProfile = 'kd-validation',
    [int]$BatchSize = 8,
    [switch]$Resume,
    [bool]$IncludeAlreadyInMainEval = $false
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

$checkpointsRoot = Join-Path $projectRoot 'checkpoints'
$outputRootAbs = Join-Path $projectRoot $OutputRoot
New-Item -ItemType Directory -Path $outputRootAbs -Force | Out-Null

$alreadyInMainEval = @(
    'wavlm_audio_kd_text_en_text_fr_audio_retrained_teacher',
    'wavlm_audio_kd_text_en_text_de_audio'
)

$checkpointDirs = Get-ChildItem -Path $checkpointsRoot -Directory |
    Where-Object {
        $_.Name -like $CheckpointPattern -and
        (Test-Path (Join-Path $_.FullName 'best_model'))
    } |
    Sort-Object Name

if (-not $IncludeAlreadyInMainEval) {
    $checkpointDirs = $checkpointDirs | Where-Object { $alreadyInMainEval -notcontains $_.Name }
}

if (-not $checkpointDirs -or $checkpointDirs.Count -eq 0) {
    throw "Nu am gasit checkpoint-uri KD potrivite pentru pattern '$CheckpointPattern' (dupa filtrare)."
}

Write-Host "Modele KD selectate:"
foreach ($dir in $checkpointDirs) {
    Write-Host "  - $($dir.Name)"
}

foreach ($dir in $checkpointDirs) {
    $checkpointName = $dir.Name
    $outputDir = Join-Path $outputRootAbs $checkpointName
    $resultsFile = Join-Path $outputDir 'test_results.json'

    if ($Resume.IsPresent -and (Test-Path $resultsFile)) {
        Write-Host "[resume] Sar peste modelul deja evaluat: $checkpointName"
        continue
    }

    Invoke-PythonStep -Name "Testare KD Test1: $checkpointName" -Arguments @(
        'scripts/test_wavlm_audio.py',
        '--checkpoint-dir', (Join-Path 'checkpoints' $checkpointName),
        '--output-dir', (Join-Path $OutputRoot $checkpointName),
        '--partition', $Partition,
        '--eval-profile', $EvalProfile,
        '--batch-size', $BatchSize
    )
}

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline testare KD pe Test1 finalizat'
Write-Host ('=' * 90)
Write-Host "Rezultate in: $OutputRoot"
