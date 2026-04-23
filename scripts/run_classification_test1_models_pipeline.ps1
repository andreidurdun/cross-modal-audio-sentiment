[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$Partition = 'test1',
    [string]$OutputRoot = 'results\classification_test1_suite',
    [int]$TextBatchSize = 32,
    [int]$AudioBatchSize = 8,
    [int]$CcmtBatchSize = 32
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
Assert-PathExists -Path (Join-Path $projectRoot 'scripts\test_classification_test1_suite.py') -Description 'Scriptul suite de testare Test1'

Invoke-PythonStep -Name 'Testare modele clasificare pe Test1' -Arguments @(
    'scripts/test_classification_test1_suite.py',
    '--partition', $Partition,
    '--output-root', $OutputRoot,
    '--text-batch-size', $TextBatchSize,
    '--audio-batch-size', $AudioBatchSize,
    '--ccmt-batch-size', $CcmtBatchSize
)

Write-Host ''
Write-Host ('=' * 90)
Write-Host 'Pipeline testare Test1 finalizat'
Write-Host ('=' * 90)
Write-Host "Rezumat: $OutputRoot\summary.json"
