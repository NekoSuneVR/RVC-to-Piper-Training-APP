[CmdletBinding()]
param(
  [ValidateSet('cpu','cuda')][string]$Engine = 'cuda',
  [switch]$InstallBuildTools
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Tools = Join-Path $Root 'tools'
$AppPython = Join-Path $Tools 'python\python.exe'
$TrainerRoot = Join-Path $Tools 'piper-trainer'
$Venv = Join-Path $TrainerRoot '.venv'
$TrainerPython = Join-Path $Venv 'Scripts\python.exe'
$Source = Join-Path $TrainerRoot 'piper1-gpl'
$Zip = Join-Path $TrainerRoot 'piper1-gpl.zip'
$Extract = Join-Path $TrainerRoot 'source-download'

function Invoke-Checked([scriptblock]$Command, [string]$Description) {
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE." }
}

function Get-Download([string]$Url, [string]$Destination) {
  Write-Host "Downloading $Url" -ForegroundColor Cyan
  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
}

function Find-VsWhere {
  $Candidate = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (Test-Path $Candidate) { return $Candidate }
  return $null
}

function Test-CppBuildTools {
  if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return $true }
  $VsWhere = Find-VsWhere
  if (-not $VsWhere) { return $false }
  $InstallPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  return [bool]$InstallPath
}

New-Item -ItemType Directory -Force -Path $TrainerRoot | Out-Null

if (-not (Test-Path $AppPython)) {
  throw "App-local Python was not found at $AppPython. Run Easy Setup.cmd first."
}

if (-not (Test-CppBuildTools)) {
  if ($InstallBuildTools) {
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) { throw 'winget was not found, so Visual Studio Build Tools cannot be installed automatically.' }
    Write-Host 'Installing Visual Studio 2022 C++ Build Tools. Windows may request elevation...' -ForegroundColor Yellow
    Invoke-Checked {
      & winget.exe install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-package-agreements --accept-source-agreements --override '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    } 'Visual Studio Build Tools installation'
  } else {
    throw @'
Piper training needs the Microsoft C++ build tools to compile monotonic alignment.
Install "Visual Studio 2022 Build Tools" with the "Desktop development with C++" workload,
or run this script again with -InstallBuildTools.
'@
  }
}

if (-not (Test-Path $TrainerPython)) {
  Write-Host 'Creating isolated Piper training Python environment...' -ForegroundColor Cyan
  Invoke-Checked { & $AppPython -m venv $Venv } 'Piper trainer virtual environment creation'
}

if (-not (Test-Path (Join-Path $Source 'src\piper\train'))) {
  if (Test-Path $Source) { Remove-Item -LiteralPath $Source -Recurse -Force }
  if (Test-Path $Extract) { Remove-Item -LiteralPath $Extract -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $Extract | Out-Null
  Get-Download 'https://github.com/OHF-Voice/piper1-gpl/archive/refs/heads/main.zip' $Zip
  Expand-Archive -Force $Zip $Extract
  $Downloaded = Join-Path $Extract 'piper1-gpl-main'
  if (-not (Test-Path (Join-Path $Downloaded 'src\piper\train'))) {
    throw 'The downloaded Piper source did not contain src\piper\train.'
  }
  Move-Item -LiteralPath $Downloaded -Destination $Source
  Remove-Item -LiteralPath $Zip -Force
  Remove-Item -LiteralPath $Extract -Recurse -Force
}

Write-Host 'Installing Piper training dependencies...' -ForegroundColor Cyan
Invoke-Checked { & $TrainerPython -m pip install --upgrade pip setuptools wheel } 'pip bootstrap'
Invoke-Checked { & $TrainerPython -m pip install --upgrade scikit-build cmake ninja cython } 'Piper build dependencies'
$EditableSpec = "$Source[train]"
Invoke-Checked { & $TrainerPython -m pip install -e $EditableSpec } 'Piper training package installation'

if ($Engine -eq 'cuda') {
  Write-Host 'Selecting CUDA 12.8 PyTorch wheels...' -ForegroundColor Cyan
  Invoke-Checked { & $TrainerPython -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 } 'CUDA PyTorch installation'
} else {
  Write-Host 'Selecting CPU PyTorch wheels...' -ForegroundColor Cyan
  Invoke-Checked { & $TrainerPython -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu } 'CPU PyTorch installation'
}

$AlignDir = Join-Path $Source 'src\piper\train\vits\monotonic_align'
Write-Host 'Building Piper monotonic alignment extension...' -ForegroundColor Cyan
Push-Location $AlignDir
try {
  Invoke-Checked { & $TrainerPython setup.py build_ext --inplace } 'Piper monotonic alignment build'
} finally {
  Pop-Location
}

Write-Host 'Verifying Piper trainer...' -ForegroundColor Cyan
Invoke-Checked {
  & $TrainerPython -c "import torch; import piper.train; from piper.train.vits.monotonic_align.core import maximum_path_c; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
} 'Piper trainer verification'

$Marker = Join-Path $TrainerRoot '.setup-complete'
Set-Content -LiteralPath $Marker -Encoding UTF8 -Value "piper-trainer; engine=$Engine; completed=$(Get-Date -Format o); source=https://github.com/OHF-Voice/piper1-gpl"

Write-Host ''
Write-Host 'Piper trainer is ready.' -ForegroundColor Green
Write-Host "Python: $TrainerPython"
Write-Host "Source: $Source"
Write-Host 'Open Build Piper Model.cmd and build your RVC dataset before training.'
