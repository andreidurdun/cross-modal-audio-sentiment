<#
.SYNOPSIS
    evalueaza toate modelele de clasificare pe partitia Test1

.PARAMETER RunId
    id-ul run-ului din MSP_Podcast/classification_pipeline/<RunId>

.PARAMETER OutputRoot
    directorul radacina pentru rezultate

.PARAMETER PythonExe
    executabilul Python; implicit .venv/Scripts/python.exe

.PARAMETER BatchSizeAudio
    batch size pentru modele audio

.PARAMETER BatchSizeText
    batch size pentru modele text

.PARAMETER BatchSizeCcmt
    batch size pentru modele CCMT

.PARAMETER Resume
    daca este setat, sare peste pasii care au deja test_results.json
#>
[CmdletBinding()]
param(
    [string]$RunId = '20260421_134357',
    [string]$OutputRoot = 'results/test1_eval',
    [string]$PythonExe = '',
    [int]$BatchSizeAudio = 8,
    [int]$BatchSizeText = 32,
    [int]$BatchSizeCcmt = 32,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

if (-not $PythonExe) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $PythonExe = if (Test-Path $venvPython) { $venvPython } else { 'python' }
}

$pipelineRoot = Join-Path $projectRoot "MSP_Podcast\classification_pipeline\$RunId"
$transcriptsDir = Join-Path $pipelineRoot 'transcripts'

$transcriptEn = Join-Path $transcriptsDir 'Transcription_en_test1.json'
$transcriptEs = Join-Path $transcriptsDir 'Transcription_es_test1.json'
$transcriptDe = Join-Path $transcriptsDir 'Transcription_de_test1.json'
$transcriptFr = Join-Path $transcriptsDir 'Transcription_fr_test1.json'

foreach ($f in @($transcriptEn, $transcriptEs, $transcriptDe, $transcriptFr)) {
    if (-not (Test-Path $f)) {
        throw "Transcript lipsa: $f"
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ''
    Write-Host ('=' * 80)
    Write-Host $Name
    Write-Host ('=' * 80)
    Write-Host "$PythonExe $($Arguments -join ' ')"

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pasul a esuat (exit $LASTEXITCODE): $Name"
    }
}

function Invoke-StepResumable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$OutputDir,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $marker = Join-Path $OutputDir 'test_results.json'
    if ($Resume.IsPresent -and (Test-Path $marker)) {
        Write-Host "[resume] Sar peste pasul deja complet: $Name"
        return
    }

    Invoke-Step -Name $Name -Arguments $Arguments
}

# --- Audio ---
Invoke-StepResumable 'wavlm_audio - Test1' "$OutputRoot/wavlm_audio" @(
    'scripts/test_wavlm_audio.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/wavlm_audio',
    '--output-dir', "$OutputRoot/wavlm_audio",
    '--batch-size', "$BatchSizeAudio",
    '--eval-profile', 'standard'
)

Invoke-StepResumable 'wavlm_audio_kd (EN+FR teacher) - Test1' "$OutputRoot/wavlm_audio_kd_text_en_text_fr_audio_retrained_teacher" @(
    'scripts/test_wavlm_audio.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/wavlm_audio_kd_text_en_text_fr_audio_retrained_teacher',
    '--output-dir', "$OutputRoot/wavlm_audio_kd_text_en_text_fr_audio_retrained_teacher",
    '--batch-size', "$BatchSizeAudio",
    '--eval-profile', 'kd-validation'
)

Invoke-StepResumable 'wavlm_audio_kd (EN+DE teacher) - Test1' "$OutputRoot/wavlm_audio_kd_text_en_text_de_audio" @(
    'scripts/test_wavlm_audio.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/wavlm_audio_kd_text_en_text_de_audio',
    '--output-dir', "$OutputRoot/wavlm_audio_kd_text_en_text_de_audio",
    '--batch-size', "$BatchSizeAudio",
    '--eval-profile', 'kd-validation'
)

# --- Text ---
Invoke-StepResumable 'roberta_text_en - Test1' "$OutputRoot/roberta_text_en" @(
    'scripts/test_roberta_text_en.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/roberta_text_en',
    '--output-dir', "$OutputRoot/roberta_text_en",
    '--transcript-json', $transcriptEn,
    '--batch-size', "$BatchSizeText"
)

Invoke-StepResumable 'roberta_text_es - Test1' "$OutputRoot/roberta_text_es" @(
    'scripts/test_roberta_text_es.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/roberta_text_es',
    '--output-dir', "$OutputRoot/roberta_text_es",
    '--transcript-json', $transcriptEs,
    '--batch-size', "$BatchSizeText"
)

Invoke-StepResumable 'roberta_text_de - Test1' "$OutputRoot/roberta_text_de" @(
    'scripts/test_roberta_text_de.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/roberta_text_de',
    '--output-dir', "$OutputRoot/roberta_text_de",
    '--transcript-json', $transcriptDe,
    '--batch-size', "$BatchSizeText"
)

Invoke-StepResumable 'roberta_text_fr - Test1' "$OutputRoot/roberta_text_fr" @(
    'scripts/test_roberta_text_fr.py',
    '--partition', 'Test1',
    '--checkpoint-dir', 'checkpoints/roberta_text_fr',
    '--output-dir', "$OutputRoot/roberta_text_fr",
    '--transcript-json', $transcriptFr,
    '--batch-size', "$BatchSizeText"
)

# --- CCMT ---
Invoke-StepResumable 'ccmt text_en+audio - Test1' "$OutputRoot/ccmt_multimodal_text_en_audio" @(
    'scripts/test_ccmt_multimodal.py',
    '--partition', 'test1',
    '--checkpoint-dir', 'checkpoints/ccmt_multimodal_text_en_audio',
    '--embeddings-dir', (Join-Path $pipelineRoot 'embeddings_text_en_audio'),
    '--modalities', 'text_en,audio',
    '--output-dir', "$OutputRoot/ccmt_multimodal_text_en_audio",
    '--batch-size', "$BatchSizeCcmt"
)

Invoke-StepResumable 'ccmt text_en+text_es+audio - Test1' "$OutputRoot/ccmt_multimodal_text_en_text_es_audio" @(
    'scripts/test_ccmt_multimodal.py',
    '--partition', 'test1',
    '--checkpoint-dir', 'checkpoints/ccmt_multimodal_text_en_text_es_audio',
    '--embeddings-dir', (Join-Path $pipelineRoot 'embeddings'),
    '--modalities', 'text_en,text_es,audio',
    '--output-dir', "$OutputRoot/ccmt_multimodal_text_en_text_es_audio",
    '--batch-size', "$BatchSizeCcmt"
)

Invoke-StepResumable 'ccmt text_en+text_de+audio - Test1' "$OutputRoot/ccmt_multimodal_text_en_text_de_audio" @(
    'scripts/test_ccmt_multimodal.py',
    '--partition', 'test1',
    '--checkpoint-dir', 'checkpoints/ccmt_multimodal_text_en_text_de_audio',
    '--embeddings-dir', (Join-Path $pipelineRoot 'embeddings_text_en_text_de_audio'),
    '--modalities', 'text_en,text_de,audio',
    '--output-dir', "$OutputRoot/ccmt_multimodal_text_en_text_de_audio",
    '--batch-size', "$BatchSizeCcmt"
)

Invoke-StepResumable 'ccmt text_en+text_fr+audio (retrained) - Test1' "$OutputRoot/ccmt_multimodal_text_en_text_fr_audio_retrained" @(
    'scripts/test_ccmt_multimodal.py',
    '--partition', 'test1',
    '--checkpoint-dir', 'checkpoints/ccmt_multimodal_text_en_text_fr_audio_retrained',
    '--embeddings-dir', (Join-Path $pipelineRoot 'embeddings_text_en_text_fr_audio'),
    '--modalities', 'text_en,text_fr,audio',
    '--output-dir', "$OutputRoot/ccmt_multimodal_text_en_text_fr_audio_retrained",
    '--batch-size', "$BatchSizeCcmt"
)

Invoke-StepResumable 'ccmt text_en+text_es+text_de+text_fr+audio - Test1' "$OutputRoot/ccmt_multimodal_text_en_text_es_text_de_text_fr_audio" @(
    'scripts/test_ccmt_multimodal.py',
    '--partition', 'test1',
    '--checkpoint-dir', 'checkpoints/ccmt_multimodal_text_en_text_es_text_de_text_fr_audio',
    '--embeddings-dir', (Join-Path $pipelineRoot 'embeddings_text_en_text_es_text_de_text_fr_audio'),
    '--modalities', 'text_en,text_es,text_de,text_fr,audio',
    '--output-dir', "$OutputRoot/ccmt_multimodal_text_en_text_es_text_de_text_fr_audio",
    '--batch-size', "$BatchSizeCcmt"
)

Write-Host ''
Write-Host ('=' * 80)
Write-Host "Evaluare Test1 finalizata. Rezultate in: $OutputRoot"
Write-Host ('=' * 80)
