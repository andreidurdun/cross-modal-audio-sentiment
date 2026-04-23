[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$TeacherCheckpointDir = 'checkpoints\ccmt_multimodal_text_en_text_fr_audio_retrained',
    [string]$TeacherEmbeddingsDir = 'MSP_Podcast\embeddings_text_en_text_fr_audio',
    [string]$TeacherModalities = 'text_en,text_fr,audio',
    [float[]]$Alphas = @(0.8, 0.9),
    [string]$StudentCheckpointBase = 'checkpoints\wavlm_audio_kd_text_en_text_fr_audio_alpha',
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
Assert-PathExists -Path (Join-Path $projectRoot 'scripts\train_wavlm_audio_kd.py') -Description 'Scriptul de train KD WavLM'
Assert-PathExists -Path (Join-Path $projectRoot $TeacherCheckpointDir) -Description 'Checkpoint profesor CCMT'
Assert-PathExists -Path (Join-Path $projectRoot $TeacherEmbeddingsDir) -Description 'Director embeddings profesor'

foreach ($alpha in $Alphas) {
    if ($alpha -lt 0.0 -or $alpha -gt 1.0) {
        throw "Alpha trebuie sa fie in intervalul [0, 1]. Valoare primita: $alpha"
    }

    $alphaTag = [string]::Format('{0:0.0}', $alpha).Replace('.', '')
    $studentCheckpointDir = "${StudentCheckpointBase}_a${alphaTag}"
    $studentCheckpointPath = Join-Path $projectRoot $studentCheckpointDir

    if ((Test-Path $studentCheckpointPath) -and -not $AllowOverwrite.IsPresent) {
        throw "Checkpoint output exista deja: $studentCheckpointPath`nRuleaza cu -AllowOverwrite sau schimba StudentCheckpointBase."
    }

    Invoke-PythonStep -Name "Antrenare WavLM KD (alpha=$alpha)" -Arguments @(
        'scripts/train_wavlm_audio_kd.py',
        '--teacher-checkpoint-dir', $TeacherCheckpointDir,
        '--teacher-embeddings-dir', $TeacherEmbeddingsDir,
        '--teacher-modalities', $TeacherModalities,
        '--checkpoint-dir', $studentCheckpointDir,
        '--alpha', ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:0.0}', $alpha))
    )
}

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline WavLM KD alpha sweep finalizat'
Write-Host ('=' * 90)
Write-Host "Teacher checkpoint dir: $TeacherCheckpointDir"
Write-Host "Teacher embeddings dir: $TeacherEmbeddingsDir"
Write-Host "Teacher modalities: $TeacherModalities"
Write-Host "Alphas rulate: $($Alphas -join ', ')"