# Deja la app lista en Windows: comprueba Python y uv, instala dependencias y ejecuta `mavericks setup`.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1 [-FromUrl <enlace>] [-From <zip>] [-NoAdapter]
param([string]$FromUrl = "", [string]$From = "", [switch]$NoAdapter)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # raíz del repositorio
function Fail($problem, $fix) { Write-Host "`nERROR: $problem`nSolución: $fix" -ForegroundColor Red; exit 1 }

# 1. Python 3.11+
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Fail "No se encuentra Python." "Instala Python 3.11 o superior desde https://www.python.org/downloads/ (marca «Add python.exe to PATH») y vuelve a ejecutar este script." }
$version = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ([version]$version -lt [version]"3.11") { Fail "Python $version es demasiado antiguo." "Instala Python 3.11 o superior y vuelve a ejecutar este script." }
Write-Host "Python $version"

# 2. uv (si falta, se instala con pip en el usuario)
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) { $uvCmd = @("uv") } else {
    & python -m uv --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Instalando uv…"
        & python -m pip install --user uv
        if ($LASTEXITCODE -ne 0) { Fail "No se pudo instalar uv." "Ejecuta `python -m pip install --user uv` a mano y comprueba tu conexión." }
    }
    $uvCmd = @("python", "-m", "uv")
}
Write-Host "uv: $($uvCmd -join ' ')"

# 3. Dependencias y puesta a punto (descarga de datos, adaptador si hay GPU, e5 y prueba)
$env:UV_SYSTEM_CERTS = "1"
& $uvCmd[0] $uvCmd[1..($uvCmd.Count)] sync --extra embed --extra hub
if ($LASTEXITCODE -ne 0) { Fail "`uv sync` ha fallado." "Comprueba tu conexión (con inspección TLS, `$env:UV_SYSTEM_CERTS = '1'` ya está activado en este script) y repite." }
$extra = @("setup", "--skip-sync")
if ($FromUrl) { $extra += @("--from-url", $FromUrl) }
if ($From) { $extra += @("--from", $From) }
if ($NoAdapter) { $extra += "--no-adapter" }
& $uvCmd[0] $uvCmd[1..($uvCmd.Count)] run mavericks @extra
exit $LASTEXITCODE
