[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$CcmtEmbeddingsDir = 'MSP_Podcast\embeddings',
    [string]$CcmtCheckpointDir = 'checkpoints\ccmt_multimodal_text_en_text_es_audio',
    [string]$TeacherCheckpointDir = 'checkpoints\ccmt_multimodal_text_en_text_fr_audio_retrained',
    [string]$TeacherEmbeddingsDir = 'MSP_Podcast\embeddings_text_en_text_fr_audio',
    [string]$TeacherModalities = 'text_en,text_fr,audio',
    [string]$StudentCheckpointDir = 'checkpoints\wavlm_audio_kd_text_en_text_fr_audio_alpha3',
    [float]$Alpha = 3.0
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
Assert-PathExists -Path (Join-Path $projectRoot 'scripts\train_ccmt_classification.py') -Description 'Scriptul de train CCMT clasificare'
Assert-PathExists -Path (Join-Path $projectRoot 'scripts\train_wavlm_audio_kd.py') -Description 'Scriptul de train KD WavLM'
Assert-PathExists -Path (Join-Path $projectRoot $CcmtEmbeddingsDir) -Description 'Embeddings pentru CCMT EN+ES+audio'
Assert-PathExists -Path (Join-Path $projectRoot $TeacherCheckpointDir) -Description 'Checkpoint profesor CCMT EN+FR+audio'
Assert-PathExists -Path (Join-Path $projectRoot $TeacherEmbeddingsDir) -Description 'Embeddings profesor EN+FR+audio'

if ($TeacherModalities -ne 'text_en,text_fr,audio') {
    Write-Host "[warn] TeacherModalities este '$TeacherModalities' (implicit recomandat: text_en,text_fr,audio)."
}

if ($Alpha -ne 3.0) {
    Write-Host "[warn] Rulezi cu alpha=$Alpha (request-ul initial a fost alpha=3)."
}

Invoke-PythonStep -Name 'Antrenare CCMT text_en+text_es+audio' -Arguments @(
    'scripts/train_ccmt_classification.py',
    '--modalities', 'text_en,text_es,audio',
    '--embeddings-dir', $CcmtEmbeddingsDir,
    '--checkpoint-dir', $CcmtCheckpointDir
)

Invoke-PythonStep -Name 'Antrenare WavLM KD cu profesor CCMT text_en+text_fr+audio' -Arguments @(
    'scripts/train_wavlm_audio_kd.py',
    '--teacher-checkpoint-dir', $TeacherCheckpointDir,
    '--teacher-embeddings-dir', $TeacherEmbeddingsDir,
    '--teacher-modalities', $TeacherModalities,
    '--checkpoint-dir', $StudentCheckpointDir,
    '--alpha', ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:0.0}', $Alpha))
)

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline finalizat: CCMT EN+ES+audio -> WavLM KD (teacher EN+FR+audio, alpha=3)'
Write-Host ('=' * 90)
Write-Host "CCMT checkpoint dir: $CcmtCheckpointDir"
Write-Host "Teacher checkpoint dir: $TeacherCheckpointDir"
Write-Host "Teacher modalities: $TeacherModalities"
Write-Host "Student checkpoint dir: $StudentCheckpointDir"
Write-Host "Alpha: $Alpha"