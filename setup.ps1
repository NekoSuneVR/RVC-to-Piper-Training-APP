[CmdletBinding()]
param(
  [ValidateSet('cpu','cuda')][string]$Engine = 'cpu',
  [switch]$NoLaunch
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Tools = Join-Path $Root 'tools'
$PythonDir = Join-Path $Tools 'python'
$RvcDir = Join-Path $Tools 'rvc'
$PiperDir = Join-Path $Tools 'piper'
$ModelsDir = Join-Path $Root 'models\piper'
New-Item -ItemType Directory -Force -Path $Tools | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

function Get-Download([string]$Url, [string]$Destination) {
  Write-Host "Downloading $([IO.Path]::GetFileName($Destination))..." -ForegroundColor Cyan
  Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
}

function Invoke-Checked([scriptblock]$Command, [string]$Description) {
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Description failed with exit code $LASTEXITCODE."
  }
}

Write-Host 'RVC + Piper Studio easy setup' -ForegroundColor Green
Write-Host 'This installs local runtimes only. Models remain on your computer.'

function Find-Python312 {
  $Candidates = @()
  if (Get-Command py.exe -ErrorAction SilentlyContinue) {
    $FromLauncher = & py.exe -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $FromLauncher) { $Candidates += $FromLauncher.Trim() }
  }
  $Candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
  $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($PythonCommand) { $Candidates += $PythonCommand.Source }
  foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
    if (-not (Test-Path $Candidate)) { continue }
    & $Candidate -c "import tkinter; assert __import__('sys').version_info[:2] == (3, 12)" 2>$null
    if ($LASTEXITCODE -eq 0) { return $Candidate }
  }
  return $null
}

$BasePython = Find-Python312
if (-not $BasePython) {
  $PythonInstaller = Join-Path $Tools 'python-installer.exe'
  Get-Download 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' $PythonInstaller
  Write-Host 'Installing Python with GUI support...' -ForegroundColor Cyan
  $Install = Start-Process -FilePath $PythonInstaller -ArgumentList @('/quiet','InstallAllUsers=0','Include_pip=1','Include_tcltk=1','Include_launcher=1','Include_test=0','Shortcuts=0','AssociateFiles=0','PrependPath=0') -Wait -PassThru
  if ($Install.ExitCode -ne 0) {
    throw "Python installation failed with exit code $($Install.ExitCode)."
  }
  Remove-Item -LiteralPath $PythonInstaller
  $BasePython = Find-Python312
  if (-not $BasePython) { throw 'Python installed, but Python 3.12 with Tkinter could not be located.' }
}

$Python = Join-Path $PythonDir 'python.exe'
$PortableReady = $false
if (Test-Path $Python) {
  & $Python -c "import tkinter; assert __import__('sys').version_info[:2] == (3, 12)" 2>$null
  $PortableReady = ($LASTEXITCODE -eq 0)
}
if (-not $PortableReady) {
  if (Test-Path $PythonDir) {
    $ResolvedPythonDir = [IO.Path]::GetFullPath($PythonDir)
    $ResolvedToolsPrefix = [IO.Path]::GetFullPath($Tools) + [IO.Path]::DirectorySeparatorChar
    if (-not $ResolvedPythonDir.StartsWith($ResolvedToolsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
      throw "Refusing to replace Python outside the app tools folder: $ResolvedPythonDir"
    }
    Remove-Item -LiteralPath $ResolvedPythonDir -Recurse -Force
  }
  $BasePythonDir = Split-Path -Parent $BasePython
  Write-Host 'Creating the app-local portable Python runtime...' -ForegroundColor Cyan
  New-Item -ItemType Directory -Force -Path $PythonDir | Out-Null
  Copy-Item -Path (Join-Path $BasePythonDir '*') -Destination $PythonDir -Recurse -Force
}
Invoke-Checked { & $Python -c "import tkinter; assert __import__('sys').version_info[:2] == (3, 12)" } 'Portable Python GUI verification'

if (-not (Test-Path (Join-Path $PiperDir 'piper\piper.exe'))) {
  $PiperZip = Join-Path $Tools 'piper.zip'
  Get-Download 'https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip' $PiperZip
  New-Item -ItemType Directory -Force -Path $PiperDir | Out-Null
  Expand-Archive -Force $PiperZip $PiperDir
  Remove-Item -LiteralPath $PiperZip
}

$DefaultVoice = Join-Path $ModelsDir 'en_GB-alba-medium.onnx'
$DefaultConfig = Join-Path $ModelsDir 'en_GB-alba-medium.onnx.json'
if (-not (Test-Path $DefaultVoice)) {
  Get-Download 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx?download=true' $DefaultVoice
}
if (-not (Test-Path $DefaultConfig)) {
  Get-Download 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json?download=true' $DefaultConfig
}

if (-not (Test-Path (Join-Path $RvcDir 'infer\cli.py'))) {
  if (Test-Path $RvcDir) {
    throw "An incomplete RVC folder exists at $RvcDir. Rename that folder and launch again so setup can safely retry."
  }
  $RvcZip = Join-Path $Tools 'rvc-main.zip'
  $RvcExtractRoot = Join-Path $Tools 'rvc-download'
  Get-Download 'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/archive/refs/heads/main.zip' $RvcZip
  New-Item -ItemType Directory -Force -Path $RvcExtractRoot | Out-Null
  Expand-Archive -Force $RvcZip $RvcExtractRoot
  $ExtractedRvc = Join-Path $RvcExtractRoot 'Retrieval-based-Voice-Conversion-WebUI-main'
  if (-not (Test-Path (Join-Path $ExtractedRvc 'infer\cli.py'))) {
    throw 'The downloaded RVC archive did not contain the expected offline inference program.'
  }
  Move-Item -LiteralPath $ExtractedRvc -Destination $RvcDir
  Remove-Item -LiteralPath $RvcZip
}

Write-Host 'Installing the RVC engine (this is the longest step)...' -ForegroundColor Cyan
Invoke-Checked { & $Python -m pip install --upgrade pip } 'pip upgrade'
if ($Engine -eq 'cuda') {
  $Requirements = Join-Path $RvcDir 'requirments_cu128_py312.txt'
} else {
  $Requirements = Join-Path $RvcDir 'requirments_cpu_py312.txt'
}
if (-not (Test-Path $Requirements)) {
  throw "The RVC dependency file was not found: $Requirements"
}
Invoke-Checked { & $Python -m pip install -r $Requirements } 'RVC dependency installation'
if ($Engine -eq 'cpu') {
  Write-Host 'Selecting the reliable CPU inference backend...' -ForegroundColor Cyan
  Invoke-Checked { & $Python -m pip uninstall -y torch-directml onnxruntime-directml } 'DirectML removal'
  Invoke-Checked { & $Python -m pip install 'onnxruntime>=1.24.4,<2' } 'CPU ONNX Runtime installation'
}

Write-Host 'Downloading the RVC inference models...' -ForegroundColor Cyan
$AssetsDir = Join-Path $RvcDir 'assets'
$HubertConfig = Join-Path $AssetsDir 'hubert_base\config.json'
if (-not (Test-Path $HubertConfig)) {
  Invoke-Checked { & $Python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lj1995/VoiceConversionWebUI', allow_patterns=['hubert_base/*'], local_dir=r'$AssetsDir')" } 'HuBERT model download'
}
$RmvpeDir = Join-Path $AssetsDir 'rmvpe'
$RmvpeModel = Join-Path $RmvpeDir 'rmvpe.pt'
if (-not (Test-Path $RmvpeModel)) {
  New-Item -ItemType Directory -Force -Path $RmvpeDir | Out-Null
  Get-Download 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true' $RmvpeModel
}
$RmvpeOnnx = Join-Path $RmvpeDir 'rmvpe.onnx'
if (-not (Test-Path $RmvpeOnnx)) {
  Get-Download 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx?download=true' $RmvpeOnnx
}
$Ffmpeg = Join-Path $RvcDir 'ffmpeg.exe'
$Ffprobe = Join-Path $RvcDir 'ffprobe.exe'
if (-not (Test-Path $Ffmpeg)) {
  Get-Download 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true' $Ffmpeg
}
if (-not (Test-Path $Ffprobe)) {
  Get-Download 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true' $Ffprobe
}

Set-Content -LiteralPath (Join-Path $Tools '.setup-complete') -Value "setup-v5; completed=$(Get-Date -Format o); engine=$Engine"

Write-Host ''
Write-Host 'Setup finished. A default Piper voice is ready; add your RVC .pth model in the Models & setup tab.' -ForegroundColor Green
if (-not $NoLaunch) {
  Start-Process -FilePath $Python -ArgumentList (Join-Path $Root 'app.py') -WorkingDirectory $Root
}
