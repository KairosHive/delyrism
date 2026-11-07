# One-time setup for local development of the delyrism website (Windows,
# NVIDIA GPU).  Idempotent.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
#
#   .venv   Python 3.11 - torch cu128 - the delyrism package with every extra
#           (engine, delyrism.server, local audio / image / video / pose
#           models), editable - Node 22 (nodejs-wheel) - frontend npm install
#
# The TouchDesigner installation (particles, voice, transcription) lives in the
# do-you-dream repository, which installs this package as a dependency.
# Models (Hugging Face embedders, MediaPipe) download on first use.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Check() { if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE)" } }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step 'installing uv'
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Step '.venv (delyrism package, Node)'
if (-not (Test-Path .venv\Scripts\python.exe)) { uv venv --python 3.11 .venv; Check }
$py = '.venv\Scripts\python.exe'
uv pip install --python $py torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; Check
uv pip install --python $py -e '.[all]'; Check
uv pip install --python $py 'nodejs-wheel>=22,<23'; Check

Step 'frontend (npm install)'
$env:Path = "$root\.venv\Scripts;$env:Path"
Push-Location web/frontend
npm install; Check
Pop-Location

Step 'done - start with scripts/run.ps1'
