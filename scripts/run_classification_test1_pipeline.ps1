[CmdletBinding()]
param(
    [string]$Partition = 'test1',
    [string]$PythonExe,
    [string]$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss'),
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
        [bool]$ResumeMode,

        [switch]$TreatExistingDirectoryAsConflict
    )

    $state = Get-StepState -TargetPath $TargetPath -RequiredPaths $RequiredPaths

    if (-not $ResumeMode) {
        if ($TreatExistingDirectoryAsConflict.IsPresent) {
            if ($state -ne 'missing' -and -not (Test-DirectoryEmptyOrMissing -Path $TargetPath)) {
                throw "$Description exista deja si ar putea fi suprascris: $TargetPath"
            }
        }
        elseif ($state -ne 'missing') {
            throw "$Description exista deja si ar putea fi suprascris: $TargetPath"
        }
        return $false
    }

    if ($state -eq 'complete') {
        Write-Host "[resume] Sar peste pasul deja complet: $Description"
        return $true
    }

    if ($state -eq 'partial') {
        throw "$Description are artefacte partiale in $TargetPath. Resume sigur nu este posibil; foloseste alt RunName sau curata artefactele."
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

$supportedPartitions = @('test1')
if ($supportedPartitions -notcontains $Partition.ToLowerInvariant()) {
    throw "Partition invalida: $Partition. In prezent scriptul suporta doar: $($supportedPartitions -join ', ')"
}

$requiredInputs = @(
    @{ Path = $PythonExe; Description = 'Interpretorul Python selectat' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Labels\labels_consensus.csv'); Description = 'Fisierul de labels' },
    @{ Path = (Join-Path $projectRoot 'MSP_Podcast\Audios'); Description = 'Directorul audio' },
    @{ Path = (Join-Path $projectRoot 'scripts\01_run_transcription.py'); Description = 'Scriptul de transcriere' },
    @{ Path = (Join-Path $projectRoot 'scripts\02_run_translation_es.py'); Description = 'Scriptul de traducere spaniola' },
    @{ Path = (Join-Path $projectRoot 'scripts\02_run_translation_de.py'); Description = 'Scriptul de traducere germana' },
    @{ Path = (Join-Path $projectRoot 'scripts\02_run_translation_fr.py'); Description = 'Scriptul de traducere franceza' },
    @{ Path = (Join-Path $projectRoot 'scripts\extract_and_save_embeddings.py'); Description = 'Scriptul de extractie embeddings' }
)

foreach ($item in $requiredInputs) {
    Assert-PathExists -Path $item.Path -Description $item.Description
}

$runRoot = Join-Path $projectRoot (Join-Path 'MSP_Podcast\classification_pipeline' $RunName)
$transcriptsDir = Join-Path $runRoot 'transcripts'
$transcriptionEn = Join-Path $transcriptsDir 'Transcription_en_test1.json'
$transcriptionEs = Join-Path $transcriptsDir 'Transcription_es_test1.json'
$transcriptionDe = Join-Path $transcriptsDir 'Transcription_de_test1.json'
$transcriptionFr = Join-Path $transcriptsDir 'Transcription_fr_test1.json'

$embeddingSpecs = @(
    @{
        Name = 'Embeddings classification: text_en,audio'
        Modalities = 'text_en,audio'
        OutputDir = (Join-Path $runRoot 'embeddings_text_en_audio')
        ReuseFrom = $null
    },
    @{
        Name = 'Embeddings classification: text_en,text_es,audio'
        Modalities = 'text_en,text_es,audio'
        OutputDir = (Join-Path $runRoot 'embeddings')
        ReuseFrom = (Join-Path $runRoot 'embeddings_text_en_audio')
    },
    @{
        Name = 'Embeddings classification: text_en,text_de,audio'
        Modalities = 'text_en,text_de,audio'
        OutputDir = (Join-Path $runRoot 'embeddings_text_en_text_de_audio')
        ReuseFrom = (Join-Path $runRoot 'embeddings_text_en_audio')
    },
    @{
        Name = 'Embeddings classification: text_en,text_fr,audio'
        Modalities = 'text_en,text_fr,audio'
        OutputDir = (Join-Path $runRoot 'embeddings_text_en_text_fr_audio')
        ReuseFrom = (Join-Path $runRoot 'embeddings_text_en_audio')
    }
)

if (-not $Resume.IsPresent) {
    Assert-CanCreateOutput -Path $transcriptionEn -Description 'Fisierul transcriptiei in engleza pentru pipeline' -AllowOverwriteMode $AllowOverwrite.IsPresent
    Assert-CanCreateOutput -Path $transcriptionEs -Description 'Fisierul transcriptiei in spaniola pentru pipeline' -AllowOverwriteMode $AllowOverwrite.IsPresent
    Assert-CanCreateOutput -Path $transcriptionDe -Description 'Fisierul transcriptiei in germana pentru pipeline' -AllowOverwriteMode $AllowOverwrite.IsPresent
    Assert-CanCreateOutput -Path $transcriptionFr -Description 'Fisierul transcriptiei in franceza pentru pipeline' -AllowOverwriteMode $AllowOverwrite.IsPresent

    foreach ($spec in $embeddingSpecs) {
        $completionFile = Join-Path $spec.OutputDir "embeddings_$Partition.pt"
        Assert-CanCreateOutput -Path $completionFile -Description $spec.Name -AllowOverwriteMode $AllowOverwrite.IsPresent
    }
}

New-Item -ItemType Directory -Path $transcriptsDir -Force | Out-Null

$skipEnglish = Assert-StepCanRunOrResume -TargetPath $transcriptionEn -Description 'Fisierul transcriptiei in engleza pentru pipeline' -RequiredPaths @($transcriptionEn) -ResumeMode $Resume.IsPresent
if (-not $skipEnglish) {
    Invoke-PythonStep -Name 'Transcriere in engleza pentru test1' -Arguments @(
        'scripts/01_run_transcription.py',
        '--partitions', 'Test1',
        '--output-json', $transcriptionEn
    )
}

$skipSpanish = Assert-StepCanRunOrResume -TargetPath $transcriptionEs -Description 'Fisierul transcriptiei in spaniola pentru pipeline' -RequiredPaths @($transcriptionEs) -ResumeMode $Resume.IsPresent
if (-not $skipSpanish) {
    Invoke-PythonStep -Name 'Traducere in spaniola pentru test1' -Arguments @(
        'scripts/02_run_translation_es.py',
        '--input-path', $transcriptionEn,
        '--output-path', $transcriptionEs
    )
}

$skipGerman = Assert-StepCanRunOrResume -TargetPath $transcriptionDe -Description 'Fisierul transcriptiei in germana pentru pipeline' -RequiredPaths @($transcriptionDe) -ResumeMode $Resume.IsPresent
if (-not $skipGerman) {
    Invoke-PythonStep -Name 'Traducere in germana pentru test1' -Arguments @(
        'scripts/02_run_translation_de.py',
        '--input-path', $transcriptionEn,
        '--output-path', $transcriptionDe
    )
}

$skipFrench = Assert-StepCanRunOrResume -TargetPath $transcriptionFr -Description 'Fisierul transcriptiei in franceza pentru pipeline' -RequiredPaths @($transcriptionFr) -ResumeMode $Resume.IsPresent
if (-not $skipFrench) {
    Invoke-PythonStep -Name 'Traducere in franceza pentru test1' -Arguments @(
        'scripts/02_run_translation_fr.py',
        '--input-path', $transcriptionEn,
        '--output-path', $transcriptionFr
    )
}

foreach ($spec in $embeddingSpecs) {
    $completionFile = Join-Path $spec.OutputDir "embeddings_$Partition.pt"
    $metadataFile = Join-Path $spec.OutputDir "metadata_$Partition.json"
    $skipEmbeddings = Assert-StepCanRunOrResume -TargetPath $spec.OutputDir -Description $spec.Name -RequiredPaths @($completionFile, $metadataFile) -ResumeMode $Resume.IsPresent -TreatExistingDirectoryAsConflict
    if ($skipEmbeddings) {
        continue
    }

    $arguments = @(
        'scripts/extract_and_save_embeddings.py',
        '--partition', $Partition,
        '--output-dir', $spec.OutputDir,
        '--modalities', $spec.Modalities,
        '--dataset-root', 'MSP_Podcast',
        '--transcripts-en-json', $transcriptionEn,
        '--transcripts-es-json', $transcriptionEs,
        '--transcripts-de-json', $transcriptionDe,
        '--transcripts-fr-json', $transcriptionFr
    )

    if ($spec.ReuseFrom) {
        $arguments += @('--reuse-from', $spec.ReuseFrom)
    }

    if ($AllowOverwrite.IsPresent) {
        $arguments += '--allow-overwrite'
    }

    Invoke-PythonStep -Name $spec.Name -Arguments $arguments
}

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline finalizat'
Write-Host ('=' * 90)
Write-Host "Resume mode: $($Resume.IsPresent)"
Write-Host "Artefacte transcripturi: $transcriptsDir"
Write-Host "Artefacte embeddings: $runRoot"