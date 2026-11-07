# Start the delyrism website locally: API and frontend, each in its own window.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run.ps1
#
#   backend   python -m delyrism.server   http://localhost:8000
#   frontend  Next.js dev server           http://localhost:3000
#
# Cloudflare embedders / LLMs: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
# are taken from the process or, failing that, the Windows User environment.
# Without them pick a local embedder in the sidebar.  Run scripts/setup.ps1 once.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

foreach ($v in 'CLOUDFLARE_ACCOUNT_ID', 'CLOUDFLARE_API_TOKEN') {
    if (-not [Environment]::GetEnvironmentVariable($v, 'Process')) {
        $u = [Environment]::GetEnvironmentVariable($v, 'User')
        if ($u) { [Environment]::SetEnvironmentVariable($v, $u.Trim(), 'Process') }
    }
}
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

function Launch($title, $cmd) {
    $full = "`$host.UI.RawUI.WindowTitle = 'delyrism - $title'; Set-Location '$root'; $cmd"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $full
}

Launch 'backend' '& .\.venv\Scripts\python.exe -m delyrism.server --port 8000'
Launch 'frontend' "`$env:Path = '$root\.venv\Scripts;' + `$env:Path; `$env:NEXT_PUBLIC_API_BASE = 'http://localhost:8000'; npm --prefix web/frontend run dev"
Write-Host 'backend :8000 - frontend :3000 - open http://localhost:3000'
