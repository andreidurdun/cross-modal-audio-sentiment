[CmdletBinding()]
param(
    [switch]$SkipBackbones,
    [switch]$SkipEmbeddings,
    [switch]$SkipCcmt,
    [switch]$Resume,
    [string]$PythonExe,
    [string]$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss')
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

function Assert-PathAbsent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (Test-Path $Path) {
        throw "$Description exista deja si ar putea fi suprascris: $Path"
    }
}

function Test-DirectoryEmptyOrMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $true
    }

    if (-not (Test-Path $Path -PathType Container)) {
        return $false
    }

    return $null -eq (Get-ChildItem -Force $Path | Select-Object -First 1)
}

function Get-StepState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetDir,

        [Parameter(Mandatory = $true)]
        [string[]]$RequiredPaths
    )

    if (-not (Test-Path $TargetDir)) {
        return 'missing'
    }

    if (-not (Test-Path $TargetDir -PathType Container)) {
        return 'invalid'
    }

    foreach ($requiredPath in $RequiredPaths) {
        if (-not (Test-Path $requiredPath)) {
            return 'partial'
        }
    }

    return 'complete'
}

function Assert-StepCanRunOrResume {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetDir,

        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [string[]]$RequiredPaths,

        [Parameter(Mandatory = $true)]
        [bool]$ResumeMode
    )

    $state = Get-StepState -TargetDir $TargetDir -RequiredPaths $RequiredPaths

    if (-not $ResumeMode) {
        if ($state -ne 'missing' -and -not (Test-DirectoryEmptyOrMissing -Path $TargetDir)) {
            throw "$Description exista deja si ar putea fi suprascris: $TargetDir"
        }
        return $false
    }

    if ($state -eq 'complete') {
        Write-Host "[resume] Sar peste pasul deja complet: $Description"
        return $true
    }

    if ($state -eq 'partial') {
        throw "$Description are artefacte partiale in $TargetDir. Resume sigur nu este posibil; foloseste alt RunName sau curata directorul."
    }

    if ($state -eq 'invalid') {
        throw "$Description are o cale invalida (nu este director): $TargetDir"
    }

    return $false
}

function Assert-StepComplete {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetDir,

        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [string[]]$RequiredPaths,

        [switch]$SuggestResumeWithoutSkipEmbeddings
    )

    $state = Get-StepState -TargetDir $TargetDir -RequiredPaths $RequiredPaths
    if ($state -eq 'complete') {
        return
    }

    if ($state -eq 'partial') {
        throw "$Description este incomplet in $TargetDir si nu poate fi folosit ca prerechizit."
    }

    if ($state -eq 'invalid') {
        throw "$Description are o cale invalida (nu este director): $TargetDir"
    }

    if ($SuggestResumeWithoutSkipEmbeddings) {
        throw "$Description nu exista: $TargetDir`nRuleaza din nou cu -Resume -SkipBackbones -RunName $RunName, fara -SkipEmbeddings."
    }

    throw "$Description nu exista: $TargetDir"
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

$checkpointRoot = Join-Path $projectRoot (Join-Path 'checkpoints\regression_pipeline' $RunName)
$embeddingsRoot = Join-Path $projectRoot (Join-Path 'MSP_Podcast\regression_pipeline' $RunName)

$backboneCheckpointDirs = @{
    text_en = Join-Path $checkpointRoot 'roberta_text_en_regression'
    text_es = Join-Path $checkpointRoot 'roberta_text_es_regression'
    text_de = Join-Path $checkpointRoot 'roberta_text_de_regression'
    text_fr = Join-Path $checkpointRoot 'roberta_text_fr_regression'
    audio = Join-Path $checkpointRoot 'wavlm_audio_regression'
}

$backboneCompletionFiles = @{
    text_en = @(
        (Join-Path $backboneCheckpointDirs.text_en 'best_model'),
        (Join-Path $backboneCheckpointDirs.text_en 'training_results.json')
    )
    text_es = @(
        (Join-Path $backboneCheckpointDirs.text_es 'best_model'),
        (Join-Path $backboneCheckpointDirs.text_es 'training_results.json')
    )
    text_de = @(
        (Join-Path $backboneCheckpointDirs.text_de 'best_model'),
        (Join-Path $backboneCheckpointDirs.text_de 'training_results.json')
    )
    text_fr = @(
        (Join-Path $backboneCheckpointDirs.text_fr 'best_model'),
        (Join-Path $backboneCheckpointDirs.text_fr 'training_results.json')
    )
    audio = @(
        (Join-Path $backboneCheckpointDirs.audio 'best_model'),
        (Join-Path $backboneCheckpointDirs.audio 'training_results.json')
    )
}

$requiredInputs = @(
    @{ Path = (Join-Path $projectRoot 'configs\training_config.json'); Description = 'Configuratia de antrenare' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Labels\labels_consensus.csv'); Description = 'Fisierul de labels' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Audios'); Description = 'Directorul audio' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Transcription_en.json'); Description = 'Transcriptia in engleza' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Transcription_es.json'); Description = 'Transcriptia in spaniola' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Transcription_de.json'); Description = 'Transcriptia in germana' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Transcription_fr.json'); Description = 'Transcriptia in franceza' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_roberta_text_en_regression.py'); Description = 'Scriptul de train text_en regression' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_roberta_text_es_regression.py'); Description = 'Scriptul de train text_es regression' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_roberta_text_de_regression.py'); Description = 'Scriptul de train text_de regression' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_roberta_text_fr_regression.py'); Description = 'Scriptul de train text_fr regression' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_wavlm_audio_regression.py'); Description = 'Scriptul de train audio regression' },
    @{ Path = (Join-Path $projectRoot 'scripts\extract_and_save_embeddings.py'); Description = 'Scriptul de extractie embeddings' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_ccmt_regression.py'); Description = 'Scriptul de train CCMT regression' }
)

foreach ($item in $requiredInputs) {
    Assert-PathExists -Path $item.Path -Description $item.Description
}

Assert-PathExists -Path $PythonExe -Description 'Interpretorul Python selectat'

$backboneSteps = @(
    @('Train regression backbone: text_en', @('scripts/train_roberta_text_en_regression.py', '--checkpoint-dir', $backboneCheckpointDirs.text_en)),
    @('Train regression backbone: text_es', @('scripts/train_roberta_text_es_regression.py', '--checkpoint-dir', $backboneCheckpointDirs.text_es)),
    @('Train regression backbone: text_de', @('scripts/train_roberta_text_de_regression.py', '--checkpoint-dir', $backboneCheckpointDirs.text_de)),
    @('Train regression backbone: text_fr', @('scripts/train_roberta_text_fr_regression.py', '--checkpoint-dir', $backboneCheckpointDirs.text_fr)),
    @('Train regression backbone: audio', @('scripts/train_wavlm_audio_regression.py', '--checkpoint-dir', $backboneCheckpointDirs.audio))
)

$combinations = @(
    @{
        Name = 'text_en + text_es + audio'
        Modalities = 'text_en,text_es,audio'
        EmbeddingsDir = (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio')
        CcmtCheckpointDir = (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_es_audio')
        ReuseFrom = @()
        EmbeddingsCompletionFiles = @(
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio') 'embeddings_train.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio') 'embeddings_val.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio') 'metadata_train.json'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio') 'metadata_val.json')
        )
        CcmtCompletionFiles = @(
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_es_audio') 'best_model.pt'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_es_audio') 'best_model_test_metrics.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_es_audio') 'training_history.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_es_audio') 'training_config.json')
        )
    },
    @{
        Name = 'text_en + audio'
        Modalities = 'text_en,audio'
        EmbeddingsDir = (Join-Path $embeddingsRoot 'embeddings_text_en_audio')
        CcmtCheckpointDir = (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_audio')
        ReuseFrom = @(
            (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio')
        )
        EmbeddingsCompletionFiles = @(
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_audio') 'embeddings_train.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_audio') 'embeddings_val.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_audio') 'metadata_train.json'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_audio') 'metadata_val.json')
        )
        CcmtCompletionFiles = @(
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_audio') 'best_model.pt'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_audio') 'best_model_test_metrics.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_audio') 'training_history.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_audio') 'training_config.json')
        )
    },
    @{
        Name = 'text_en + text_de + audio'
        Modalities = 'text_en,text_de,audio'
        EmbeddingsDir = (Join-Path $embeddingsRoot 'embeddings_text_en_text_de_audio')
        CcmtCheckpointDir = (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_de_audio')
        ReuseFrom = @(
            (Join-Path $embeddingsRoot 'embeddings_text_en_audio'),
            (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio')
        )
        EmbeddingsCompletionFiles = @(
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_de_audio') 'embeddings_train.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_de_audio') 'embeddings_val.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_de_audio') 'metadata_train.json'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_de_audio') 'metadata_val.json')
        )
        CcmtCompletionFiles = @(
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_de_audio') 'best_model.pt'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_de_audio') 'best_model_test_metrics.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_de_audio') 'training_history.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_de_audio') 'training_config.json')
        )
    },
    @{
        Name = 'text_en + text_fr + audio'
        Modalities = 'text_en,text_fr,audio'
        EmbeddingsDir = (Join-Path $embeddingsRoot 'embeddings_text_en_text_fr_audio')
        CcmtCheckpointDir = (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_fr_audio')
        ReuseFrom = @(
            (Join-Path $embeddingsRoot 'embeddings_text_en_audio'),
            (Join-Path $embeddingsRoot 'embeddings_text_en_text_es_audio')
        )
        EmbeddingsCompletionFiles = @(
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_fr_audio') 'embeddings_train.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_fr_audio') 'embeddings_val.pt'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_fr_audio') 'metadata_train.json'),
            (Join-Path (Join-Path $embeddingsRoot 'embeddings_text_en_text_fr_audio') 'metadata_val.json')
        )
        CcmtCompletionFiles = @(
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_fr_audio') 'best_model.pt'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_fr_audio') 'best_model_test_metrics.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_fr_audio') 'training_history.json'),
            (Join-Path (Join-Path $checkpointRoot 'ccmt_multimodal_regression_text_en_text_fr_audio') 'training_config.json')
        )
    }
)

if (-not $SkipBackbones) {
    foreach ($step in $backboneSteps) {
        $modality = ($step[0] -split ': ')[1]
        $targetDir = $backboneCheckpointDirs[$modality]
        $skipStep = Assert-StepCanRunOrResume -TargetDir $targetDir -Description "Directorul de checkpoint pentru backbone $modality" -RequiredPaths $backboneCompletionFiles[$modality] -ResumeMode $Resume
        if (-not $skipStep) {
            Invoke-Step -Name $step[0] -Arguments $step[1]
        }
    }
}
elseif (-not $SkipEmbeddings) {
    foreach ($modality in $backboneCheckpointDirs.Keys) {
        Assert-StepComplete -TargetDir $backboneCheckpointDirs[$modality] -Description "Checkpoint-ul backbone necesar pentru $modality" -RequiredPaths $backboneCompletionFiles[$modality]
    }
}

foreach ($combo in $combinations) {
    if (-not $SkipEmbeddings) {
        $skipEmbeddingsStep = Assert-StepCanRunOrResume -TargetDir $combo.EmbeddingsDir -Description "Directorul de embeddings pentru $($combo.Name)" -RequiredPaths $combo.EmbeddingsCompletionFiles -ResumeMode $Resume
        if (-not $skipEmbeddingsStep) {
            $embeddingArguments = @(
                'scripts/extract_and_save_embeddings.py',
                '--partition', 'train,val',
                '--modalities', $combo.Modalities,
                '--output-dir', $combo.EmbeddingsDir,
                '--en-checkpoint', $backboneCheckpointDirs.text_en,
                '--es-checkpoint', $backboneCheckpointDirs.text_es,
                '--de-checkpoint', $backboneCheckpointDirs.text_de,
                '--fr-checkpoint', $backboneCheckpointDirs.text_fr,
                '--audio-checkpoint', $backboneCheckpointDirs.audio
            )
            if ($combo.ReuseFrom.Count -gt 0) {
                $embeddingArguments += @('--reuse-from', ($combo.ReuseFrom -join ';'))
            }
            Invoke-Step -Name "Extract regression embeddings: $($combo.Name)" -Arguments $embeddingArguments
        }
    }
    elseif (-not $SkipCcmt) {
        Assert-StepComplete -TargetDir $combo.EmbeddingsDir -Description "Directorul de embeddings necesar pentru $($combo.Name)" -RequiredPaths $combo.EmbeddingsCompletionFiles -SuggestResumeWithoutSkipEmbeddings
    }

    if (-not $SkipCcmt) {
        $skipCcmtStep = Assert-StepCanRunOrResume -TargetDir $combo.CcmtCheckpointDir -Description "Directorul CCMT pentru $($combo.Name)" -RequiredPaths $combo.CcmtCompletionFiles -ResumeMode $Resume
        if (-not $skipCcmtStep) {
            Invoke-Step -Name "Train CCMT regression: $($combo.Name)" -Arguments @(
                'scripts/train_ccmt_regression.py',
                '--modalities', $combo.Modalities,
                '--embeddings-dir', $combo.EmbeddingsDir,
                '--checkpoint-dir', $combo.CcmtCheckpointDir
            )
        }
    }
}

Write-Host ''
Write-Host "RunName: $RunName"
Write-Host "Checkpoint root: $checkpointRoot"
Write-Host "Embeddings root: $embeddingsRoot"
Write-Host 'Pipeline-ul de regresie s-a terminat cu succes.'