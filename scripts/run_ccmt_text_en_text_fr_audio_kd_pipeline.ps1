[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$EmbeddingsDir = 'MSP_Podcast\embeddings_text_en_text_fr_audio',
    [string]$TeacherCheckpointDir = 'checkpoints\ccmt_multimodal_text_en_text_fr_audio_retrained',
    [string]$StudentCheckpointDir = 'checkpoints\wavlm_audio_kd_text_en_text_fr_audio_retrained_teacher',
    [switch]$GenerateEmbeddingsIfMissing,
    [switch]$Resume,
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

if ($Resume.IsPresent -and $AllowOverwrite.IsPresent) {
    throw 'Optiunile -Resume si -AllowOverwrite se exclud reciproc.'
}

$teacherModalities = 'text_en,text_fr,audio'
$embeddingsDirPath = Join-Path $projectRoot $EmbeddingsDir
$teacherCheckpointPath = Join-Path $projectRoot $TeacherCheckpointDir
$studentCheckpointPath = Join-Path $projectRoot $StudentCheckpointDir

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

function Assert-CanCreateOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [bool]$AllowOverwriteMode
    )

    if ((Test-Path $Path) -and -not $AllowOverwriteMode) {
        throw "$Description exista deja: $Path`nRuleaza din nou cu -AllowOverwrite daca vrei sa suprascrii artefactele."
    }
}

function Get-StepState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,

        [Parameter(Mandatory = $true)]
        [string[]]$RequiredPaths
    )

    if (-not (Test-Path $TargetPath)) {
        return 'missing'
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
        [string]$TargetPath,

        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [string[]]$RequiredPaths,

        [Parameter(Mandatory = $true)]
        [bool]$ResumeMode
    )

    $state = Get-StepState -TargetPath $TargetPath -RequiredPaths $RequiredPaths

    if (-not $ResumeMode) {
        if ($state -ne 'missing') {
            throw "$Description exista deja si ar putea fi suprascris: $TargetPath"
        }
        return $false
    }

    if ($state -eq 'complete') {
        Write-Host "[resume] Sar peste pasul deja complet: $Description"
        return $true
    }

    if ($state -eq 'partial') {
        throw "$Description are artefacte partiale in $TargetPath. Resume sigur nu este posibil; foloseste alt director sau curata artefactele."
    }

    return $false
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

$requiredInputs = @(
    @{ Path = $PythonExe; Description = 'Interpretorul Python selectat' },
    @{ Path = (Join-Path $projectRoot 'scripts\extract_and_save_embeddings.py'); Description = 'Scriptul de extractie embeddings' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_ccmt_classification.py'); Description = 'Scriptul de train CCMT clasificare' },
    @{ Path = (Join-Path $projectRoot 'scripts\train_wavlm_audio_kd.py'); Description = 'Scriptul de train KD WavLM' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Labels\labels_consensus.csv'); Description = 'Fisierul de labels' }
)

foreach ($item in $requiredInputs) {
    Assert-PathExists -Path $item.Path -Description $item.Description
}

$embeddingRequiredPaths = @(
    (Join-Path $embeddingsDirPath 'embeddings_train.pt'),
    (Join-Path $embeddingsDirPath 'embeddings_val.pt'),
    (Join-Path $embeddingsDirPath 'metadata_train.json'),
    (Join-Path $embeddingsDirPath 'metadata_val.json')
)

$teacherRequiredPaths = @(
    (Join-Path $teacherCheckpointPath 'best_model.pt'),
    (Join-Path $teacherCheckpointPath 'training_config.json'),
    (Join-Path $teacherCheckpointPath 'best_model_test_metrics.json')
)

$studentRequiredPaths = @(
    (Join-Path $studentCheckpointPath 'best_model'),
    (Join-Path $studentCheckpointPath 'training_results.json')
)

if (-not $Resume.IsPresent) {
    if ($GenerateEmbeddingsIfMissing.IsPresent) {
        foreach ($requiredPath in $embeddingRequiredPaths) {
            if ((Test-Path $requiredPath) -and -not $AllowOverwrite.IsPresent) {
                throw "Embeddings profesor exista deja: $requiredPath`nRuleaza cu -AllowOverwrite sau fara -GenerateEmbeddingsIfMissing."
            }
        }
    }

    Assert-CanCreateOutput -Path $teacherCheckpointPath -Description 'Checkpoint-ul profesorului CCMT' -AllowOverwriteMode $AllowOverwrite.IsPresent
    Assert-CanCreateOutput -Path $studentCheckpointPath -Description 'Checkpoint-ul studentului WavLM KD' -AllowOverwriteMode $AllowOverwrite.IsPresent
}

$skipEmbeddings = $false
if ($GenerateEmbeddingsIfMissing.IsPresent) {
    $skipEmbeddings = Assert-StepCanRunOrResume -TargetPath $embeddingsDirPath -Description 'Embeddings profesor text_en,text_fr,audio' -RequiredPaths $embeddingRequiredPaths -ResumeMode $Resume.IsPresent
    if (-not $skipEmbeddings) {
        $embeddingArguments = @(
            'scripts/extract_and_save_embeddings.py',
            '--partition', 'train,val',
            '--modalities', $teacherModalities,
            '--output-dir', $EmbeddingsDir,
            '--reuse-from', 'MSP_Podcast/embeddings_text_en_audio'
        )

        if ($AllowOverwrite.IsPresent) {
            $embeddingArguments += '--allow-overwrite'
        }

        Invoke-PythonStep -Name 'Generare embeddings profesor text_en,text_fr,audio' -Arguments $embeddingArguments
    }
}
else {
    foreach ($requiredPath in $embeddingRequiredPaths) {
        Assert-PathExists -Path $requiredPath -Description 'Artefact embeddings profesor'
    }
}

$skipTeacher = Assert-StepCanRunOrResume -TargetPath $teacherCheckpointPath -Description 'Reantrenare CCMT text_en,text_fr,audio' -RequiredPaths $teacherRequiredPaths -ResumeMode $Resume.IsPresent
if (-not $skipTeacher) {
    Invoke-PythonStep -Name 'Reantrenare CCMT profesor text_en,text_fr,audio' -Arguments @(
        'scripts/train_ccmt_classification.py',
        '--modalities', $teacherModalities,
        '--embeddings-dir', $EmbeddingsDir,
        '--checkpoint-dir', $TeacherCheckpointDir
    )
}

$skipStudent = Assert-StepCanRunOrResume -TargetPath $studentCheckpointPath -Description 'Antrenare WavLM KD cu profesorul CCMT text_en,text_fr,audio' -RequiredPaths $studentRequiredPaths -ResumeMode $Resume.IsPresent
if (-not $skipStudent) {
    Invoke-PythonStep -Name 'Antrenare WavLM KD cu profesor CCMT text_en,text_fr,audio' -Arguments @(
        'scripts/train_wavlm_audio_kd.py',
        '--teacher-checkpoint-dir', $TeacherCheckpointDir,
        '--teacher-embeddings-dir', $EmbeddingsDir,
        '--teacher-modalities', $teacherModalities,
        '--checkpoint-dir', $StudentCheckpointDir
    )
}

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline CCMT + WavLM KD finalizat'
Write-Host ('=' * 90)
Write-Host "Teacher modalities: $teacherModalities"
Write-Host "Embeddings dir: $EmbeddingsDir"
Write-Host "Teacher checkpoint dir: $TeacherCheckpointDir"
Write-Host "Student checkpoint dir: $StudentCheckpointDir"
Write-Host "Generate embeddings if missing: $($GenerateEmbeddingsIfMissing.IsPresent)"
Write-Host "Resume mode: $($Resume.IsPresent)"