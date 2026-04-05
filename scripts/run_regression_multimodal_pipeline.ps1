[CmdletBinding()]
param(
    [switch]$SkipBackbones,
    [switch]$SkipEmbeddings,
    [switch]$SkipCcmt,
    [string]$PythonExe
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

function Invoke-Step {
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

$backboneSteps = @(
    @('Train regression backbone: text_en', @('scripts/train_roberta_text_en_regression.py')),
    @('Train regression backbone: text_es', @('scripts/train_roberta_text_es_regression.py')),
    @('Train regression backbone: text_de', @('scripts/train_roberta_text_de_regression.py')),
    @('Train regression backbone: text_fr', @('scripts/train_roberta_text_fr_regression.py')),
    @('Train regression backbone: audio', @('scripts/train_wavlm_audio_regression.py'))
)

$combinations = @(
    @{
        Name = 'text_en + text_es + audio'
        Modalities = 'text_en,text_es,audio'
        EmbeddingsDir = 'MSP_Podcast/embeddings'
        CcmtCheckpointDir = 'checkpoints/ccmt_multimodal_regression_text_en_text_es_audio'
    },
    @{
        Name = 'text_en + audio'
        Modalities = 'text_en,audio'
        EmbeddingsDir = 'MSP_Podcast/embeddings_text_en_audio'
        CcmtCheckpointDir = 'checkpoints/ccmt_multimodal_regression_text_en_audio'
    },
    @{
        Name = 'text_en + text_de + audio'
        Modalities = 'text_en,text_de,audio'
        EmbeddingsDir = 'MSP_Podcast/embeddings_text_en_text_de_audio'
        CcmtCheckpointDir = 'checkpoints/ccmt_multimodal_regression_text_en_text_de_audio'
    },
    @{
        Name = 'text_en + text_fr + audio'
        Modalities = 'text_en,text_fr,audio'
        EmbeddingsDir = 'MSP_Podcast/embeddings_text_en_text_fr_audio'
        CcmtCheckpointDir = 'checkpoints/ccmt_multimodal_regression_text_en_text_fr_audio'
    }
)

if (-not $SkipBackbones) {
    foreach ($step in $backboneSteps) {
        Invoke-Step -Name $step[0] -Arguments $step[1]
    }
}

foreach ($combo in $combinations) {
    if (-not $SkipEmbeddings) {
        Invoke-Step -Name "Extract regression embeddings: $($combo.Name)" -Arguments @(
            'scripts/extract_and_save_embeddings.py',
            '--partition', 'train,val',
            '--modalities', $combo.Modalities,
            '--output-dir', $combo.EmbeddingsDir,
            '--en-checkpoint', 'checkpoints/roberta_text_en_regression',
            '--es-checkpoint', 'checkpoints/roberta_text_es_regression',
            '--de-checkpoint', 'checkpoints/roberta_text_de_regression',
            '--fr-checkpoint', 'checkpoints/roberta_text_fr_regression',
            '--audio-checkpoint', 'checkpoints/wavlm_audio_regression'
        )
    }

    if (-not $SkipCcmt) {
        Invoke-Step -Name "Train CCMT regression: $($combo.Name)" -Arguments @(
            'scripts/train_ccmt_regression.py',
            '--modalities', $combo.Modalities,
            '--embeddings-dir', $combo.EmbeddingsDir,
            '--checkpoint-dir', $combo.CcmtCheckpointDir
        )
    }
}

Write-Host ''
Write-Host 'Pipeline-ul de regresie s-a terminat cu succes.'