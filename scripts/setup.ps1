# One-time setup of the local realtime stack (Windows, NVIDIA GPU).  Idempotent.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 [-NoTts] [-NoSkills]
#
#   .venv       Python 3.11 - torch cu128 - the delyrism package (engine +
#               delyrism.server, editable) - Node 22
#               (nodejs-wheel, so no system Node is needed) - frontend npm install
#   .venv-tts   Python 3.11 - torch 2.6 cu126 - Chatterbox-Turbo TTS service
#               (its pins clash with the backend, hence a second env)
#   .claude/skills/td-*   TDMCPSkills (github.com/TouchDesigner/TDMCPSkills),
#               agent skills for working on the TouchDesigner patch over TDMCP
#
# Everything else is fetched on first run: Hugging Face models (embedders,
# Chatterbox), MediaPipe .task files.  TouchDesigner itself (2025.x) is the one
# manual install; TDMCP is embedded in td/delyrism_particles.toe.
param([switch]$NoTts, [switch]$NoSkills)
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

# ---------------------------------------------------------------- backend env
Step '.venv (backend, engine, Node)'
if (-not (Test-Path .venv\Scripts\python.exe)) { uv venv --python 3.11 .venv; Check }
$py = '.venv\Scripts\python.exe'
uv pip install --python $py torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; Check
uv pip install --python $py -r requirements.txt -r web/backend/requirements.txt --override scripts/overrides.txt; Check
uv pip install --python $py -e '.[all]' --override scripts/overrides.txt; Check
uv pip install --python $py 'nodejs-wheel>=22,<23'; Check

Step 'frontend (npm install)'
$env:Path = "$root\.venv\Scripts;$env:Path"
Push-Location web/frontend
npm install; Check
Pop-Location

# ---------------------------------------------------------------- TTS env
if (-not $NoTts) {
    Step '.venv-tts (Chatterbox-Turbo)'
    if (-not (Test-Path .venv-tts\Scripts\python.exe)) { uv venv --python 3.11 .venv-tts; Check }
    $tpy = '.venv-tts\Scripts\python.exe'
    uv pip install --python $tpy torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126; Check
    uv pip install --python $tpy -r web/tts/requirements.txt; Check
}

# ---------------------------------------------------------------- agent skills
if (-not $NoSkills) {
    Step 'TDMCPSkills -> .claude/skills'
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "tdmcpskills-$([guid]::NewGuid().ToString('N'))"
    git clone --depth 1 https://github.com/TouchDesigner/TDMCPSkills.git $tmp; Check
    New-Item -ItemType Directory -Force .claude\skills | Out-Null
    Get-ChildItem "$tmp\skills" -Directory -Filter 'td-*' | ForEach-Object {
        $dst = Join-Path .claude\skills $_.Name
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item -Recurse $_.FullName $dst
    }
    Remove-Item -Recurse -Force $tmp
}

Step 'done - start everything with scripts/run.ps1'
