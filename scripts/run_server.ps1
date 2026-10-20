# Arranca el servidor del modelo ajustado (B) en la GPU y espera a /health.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_server.ps1 [-Adapter outputs\ft\runs\b1\adapter_final] [-Port 8765]
# No para Ollama: el modelo ajustado (~5 GB) y el de Ollama deben caber a la vez en la GPU (se mide y se informa).
param([string]$Adapter = "outputs\ft\runs\b1\adapter_final", [int]$Port = 8765)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # raíz del repositorio
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "No se encuentra nvidia-smi: no hay GPU NVIDIA disponible" }
$mem = nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
Write-Host "GPU: $mem"
if (-not (Test-Path "$Adapter\adapter_model.safetensors")) { throw "No existe el adaptador en $Adapter" }
try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).ok) { Write-Host "El servidor ya está arrancado en el puerto $Port"; exit 0 } } catch {}
$env:PYTHONPATH = "src"; $env:HF_HUB_OFFLINE = "1"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "outputs\ft" | Out-Null
$log = "outputs\ft\server.log"
Start-Process -FilePath ".venv_ft\Scripts\python.exe" -ArgumentList "-m", "mavericks.ft.server", "--adapter", $Adapter, "--port", $Port -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "outputs\ft\server_err.log"
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep 2
    try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.ok) { Write-Host "Servidor listo en el puerto $Port (adaptador $($h.adapter))"; Write-Host ("VRAM usada: " + (nvidia-smi --query-gpu=memory.used --format=csv,noheader)); exit 0 } } catch {}
}
throw "El servidor no respondió a /health en 3 minutos; mira outputs\ft\server_err.log"
