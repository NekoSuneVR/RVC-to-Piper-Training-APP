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
$Marker = Join-Path $TrainerRoot '.setup-complete'

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

function Build-PiperEspeakBridge([string]$SourceDir, [string]$PythonExe) {
  $RealPiper = Join-Path $SourceDir 'src\piper'
  $ExistingBridge = Get-ChildItem -LiteralPath $RealPiper -Filter 'espeakbridge*.pyd' -File -ErrorAction SilentlyContinue | Select-Object -First 1
  $RealData = Join-Path $RealPiper 'espeak-ng-data'
  if ($ExistingBridge -and (Test-Path $RealData)) {
    Write-Host "Piper eSpeak bridge already present: $($ExistingBridge.Name)" -ForegroundColor DarkGreen
    return
  }

  $ShortRoot = Join-Path $env:TEMP ("ppb-" + $PID)
  $ShortSource = Join-Path $ShortRoot 'piper'
  if (Test-Path $ShortRoot) {
    Remove-Item -LiteralPath $ShortRoot -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $ShortSource | Out-Null

  try {
    Write-Host "Copying Piper source to short native-build path: $ShortSource" -ForegroundColor DarkCyan
    Copy-Item -Path (Join-Path $SourceDir '*') -Destination $ShortSource -Recurse -Force

    Push-Location $ShortSource
    try {
      Write-Host 'Building Piper eSpeak bridge and embedded eSpeak-ng data...' -ForegroundColor Cyan
      Invoke-Checked { & $PythonExe setup.py build_ext --inplace } 'Piper eSpeak bridge build'
    } finally {
      Pop-Location
    }

    $BuiltBridge = Get-ChildItem -LiteralPath $ShortSource -Filter 'espeakbridge*.pyd' -File -Recurse |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if (-not $BuiltBridge) {
      throw "Piper native build finished without producing espeakbridge*.pyd under $ShortSource"
    }

    $ShortData = Join-Path $ShortSource 'src\piper\espeak-ng-data'
    if (-not (Test-Path $ShortData)) {
      throw "Piper native build did not create espeak-ng-data at $ShortData"
    }

    Get-ChildItem -LiteralPath $RealPiper -Filter 'espeakbridge*.pyd' -File -ErrorAction SilentlyContinue |
      Remove-Item -Force -ErrorAction SilentlyContinue
    Copy-Item -LiteralPath $BuiltBridge.FullName -Destination (Join-Path $RealPiper $BuiltBridge.Name) -Force

    if (Test-Path $RealData) {
      Remove-Item -LiteralPath $RealData -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $RealData | Out-Null
    Copy-Item -Path (Join-Path $ShortData '*') -Destination $RealData -Recurse -Force

    Write-Host "Installed Piper eSpeak bridge: $($BuiltBridge.Name)" -ForegroundColor Green
  } finally {
    if (Test-Path $ShortRoot) {
      Remove-Item -LiteralPath $ShortRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

function Build-PiperMonotonicAlignment([string]$AlignDir, [string]$PythonExe) {
  # Piper's setup.py passes an absolute path to core.pyx into Cython. On Windows,
  # setuptools mirrors that path below build\temp when creating core.obj. If the
  # app itself lives in a deep folder, MSVC can fail with C1083 / Invalid argument.
  # Build from a deliberately short temporary path, then copy the compiled .pyd
  # back into Piper's real package directory.
  $ShortBuild = Join-Path $env:TEMP ("pma-" + $PID)
  if (Test-Path $ShortBuild) {
    Remove-Item -LiteralPath $ShortBuild -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $ShortBuild | Out-Null

  try {
    Copy-Item -LiteralPath (Join-Path $AlignDir 'setup.py') -Destination (Join-Path $ShortBuild 'setup.py') -Force
    Copy-Item -LiteralPath (Join-Path $AlignDir 'core.pyx') -Destination (Join-Path $ShortBuild 'core.pyx') -Force

    Write-Host "Compiling alignment from short path: $ShortBuild" -ForegroundColor DarkCyan
    Push-Location $ShortBuild
    try {
      Invoke-Checked {
        & $PythonExe setup.py build_ext --inplace --build-temp (Join-Path $ShortBuild 'obj')
      } 'Piper monotonic alignment build'
    } finally {
      Pop-Location
    }

    $BuiltCore = Get-ChildItem -LiteralPath $ShortBuild -Filter 'core*.pyd' -File |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if (-not $BuiltCore) {
      throw "Piper monotonic alignment compiled without producing core*.pyd in $ShortBuild"
    }

    Get-ChildItem -LiteralPath $AlignDir -Filter 'core*.pyd' -File -ErrorAction SilentlyContinue |
      Remove-Item -Force -ErrorAction SilentlyContinue
    Copy-Item -LiteralPath $BuiltCore.FullName -Destination (Join-Path $AlignDir $BuiltCore.Name) -Force
    Write-Host "Installed compiled alignment extension: $($BuiltCore.Name)" -ForegroundColor Green
  } finally {
    if (Test-Path $ShortBuild) {
      Remove-Item -LiteralPath $ShortBuild -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

function Ensure-CudaTorch([string]$PythonExe) {
  Write-Host 'Checking whether PyTorch is a CUDA build...' -ForegroundColor Cyan
  $CudaBuild = & $PythonExe -c "import torch; print(torch.version.cuda or '')"
  $TorchCheckExit = $LASTEXITCODE

  if (($TorchCheckExit -ne 0) -or [string]::IsNullOrWhiteSpace(($CudaBuild | Out-String).Trim())) {
    Write-Host 'Current Torch is missing CUDA support. Replacing it with the CUDA 12.8 wheel...' -ForegroundColor Yellow
    Invoke-Checked {
      & $PythonExe -m pip install --upgrade --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
    } 'CUDA PyTorch installation'
  } else {
    Write-Host "CUDA-enabled Torch already installed (CUDA runtime $($CudaBuild | Out-String).Trim())." -ForegroundColor DarkGreen
  }

  Invoke-Checked {
    & $PythonExe -c "import torch,sys; print('torch', torch.__version__); print('torch_cuda_runtime', torch.version.cuda); sys.exit(0 if torch.version.cuda else 2)"
  } 'CUDA PyTorch build verification'

  & $PythonExe -c "import torch; print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
  if ($LASTEXITCODE -ne 0) {
    Write-Host 'WARNING: Torch CUDA diagnostics failed. Training can still fall back to CPU when Device is set to auto.' -ForegroundColor Yellow
  }
}

New-Item -ItemType Directory -Force -Path $TrainerRoot | Out-Null
if (Test-Path $Marker) {
  Remove-Item -LiteralPath $Marker -Force -ErrorAction SilentlyContinue
}

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
Piper training needs the Microsoft C++ build tools to compile its native extensions.
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

Write-Host 'Building Piper eSpeak native bridge...' -ForegroundColor Cyan
Build-PiperEspeakBridge -SourceDir $Source -PythonExe $TrainerPython

if ($Engine -eq 'cuda') {
  Write-Host 'Selecting CUDA 12.8 PyTorch wheels...' -ForegroundColor Cyan
  Ensure-CudaTorch -PythonExe $TrainerPython
} else {
  Write-Host 'Selecting CPU PyTorch wheels...' -ForegroundColor Cyan
  Invoke-Checked { & $TrainerPython -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu } 'CPU PyTorch installation'
}

$AlignDir = Join-Path $Source 'src\piper\train\vits\monotonic_align'
Write-Host 'Building Piper monotonic alignment extension...' -ForegroundColor Cyan
Build-PiperMonotonicAlignment -AlignDir $AlignDir -PythonExe $TrainerPython

Write-Host 'Verifying Piper trainer, eSpeak phonemizer and native extensions...' -ForegroundColor Cyan
Invoke-Checked {
  & $TrainerPython -c "import torch; import piper.train; from piper.phonemize_espeak import EspeakPhonemizer; p=EspeakPhonemizer(); assert p.phonemize('en-GB-x-rp','Piper trainer verification.'); from piper.train.vits.monotonic_align.core import maximum_path_c; print('espeak_voice', 'en-GB-x-rp'); print('espeakbridge_ok', True); print('alignment_ok', True); print('torch', torch.__version__); print('torch_cuda_runtime', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
} 'Piper trainer verification'

Set-Content -LiteralPath $Marker -Encoding UTF8 -Value "piper-trainer-v3; engine=$Engine; completed=$(Get-Date -Format o); source=https://github.com/OHF-Voice/piper1-gpl"

Write-Host ''
Write-Host 'Piper trainer is ready.' -ForegroundColor Green
Write-Host "Python: $TrainerPython"
Write-Host "Source: $Source"
Write-Host 'Native eSpeak bridge: ready'
Write-Host 'Monotonic alignment: ready'
