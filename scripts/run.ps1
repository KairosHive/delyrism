# Start the local realtime stack, each service in its own window.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run.ps1 [-NoTts] [-NoTD] [-NoFrontend]
#
#   backend   FastAPI  http://localhost:8000   (OSC -> TouchDesigner on 127.0.0.1:7000)
#   frontend  Next.js  http://localhost:3000   (Live tab drives the graph)
#   tts       Chatterbox-Turbo  http://localhost:8010  (speaks the query's top word)
#   td        td/delyrism_particles.toe (TouchDesigner 2025.x)
#
# Cloudflare embedders / LLMs: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
# are taken from the process or, failing that, the Windows User environment.
# Without them pick the local sentence-transformer backend in the sidebar.
# Run scripts/setup.ps1 once first.
param([switch]$NoTts, [switch]$NoTD, [switch]$NoFrontend)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

foreach ($v in 'CLOUDFLARE_ACCOUNT_ID', 'CLOUDFLARE_API_TOKEN') {
    if (-not [Environment]::GetEnvironmentVariable($v, 'Process')) {
        $u = [Environment]::GetEnvironmentVariable($v, 'User')
        if ($u) { [Environment]::SetEnvironmentVariable($v, $u.Trim(), 'Process') }
    }
}
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

function Launch($title, $cmd) {
    $full = "`$host.UI.RawUI.WindowTitle = 'delyrism - $title'; Set-Location '$root'; $cmd"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $full
}

Launch 'backend' '& .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --app-dir web/backend'
if (-not $NoFrontend) {
    Launch 'frontend' "`$env:Path = '$root\.venv\Scripts;' + `$env:Path; `$env:NEXT_PUBLIC_API_BASE = 'http://localhost:8000'; npm --prefix web/frontend run dev"
}
if (-not $NoTts) {
    Launch 'tts' '& .\.venv-tts\Scripts\python.exe -m uvicorn tts_server:app --app-dir web/tts --port 8010'
}
if (-not $NoTD) {
    Start-Process (Join-Path $root 'td\delyrism_particles.toe')
}
Write-Host 'backend :8000 - frontend :3000 - tts :8010 - open http://localhost:3000'
