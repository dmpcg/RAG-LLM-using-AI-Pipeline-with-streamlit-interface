# Financial Report Insights - Robust Desktop Launcher with Self-Healing
# Run with: powershell -ExecutionPolicy Bypass -File "C:\Users\dtmcg\RAG-LLM-project\launch_financial_insights.ps1"

param(
    [int]$Port = 8501,
    [int]$MaxStartupWaitSeconds = 45,
    [switch]$ForceRestart,
    [switch]$Visible
)

$ErrorActionPreference = "Continue"
$projectRoot = "C:\Users\dtmcg\RAG-LLM-project"
$appDir = Join-Path $projectRoot "financial-report-insights"
$venvPython = Join-Path $appDir ".venv\Scripts\python.exe"
$venvStreamlit = Join-Path $appDir ".venv\Scripts\streamlit.exe"
$logFile = Join-Path $projectRoot "launch_financial_insights.log"
$url = "http://localhost:$Port"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $logFile -Value $entry -ErrorAction SilentlyContinue
    if ($Level -eq "ERROR") { Write-Host $entry -ForegroundColor Red } else { Write-Host $entry }
}

function Test-PortListening {
    param([int]$Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return ($connections.Count -gt 0)
    } catch { return $false }
}

function Test-UrlHealthy {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        return ($response.StatusCode -eq 200)
    } catch { return $false }
}

function Stop-PortOwner {
    # Kill whatever process is LISTENING on the given port, regardless of its name.
    # A Streamlit server runs as python.exe (ProcessName 'python'), so name-based
    # matching (*streamlit*) never finds it and a stale instance keeps squatting on
    # the port, causing every relaunch to fail the health check. Kill by port owner.
    param([int]$Port)
    try {
        $owners = (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).OwningProcess |
            Sort-Object -Unique
        foreach ($pidToKill in $owners) {
            if ($pidToKill) {
                Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
                Write-Log "Stopped stale process PID $pidToKill holding port $Port."
            }
        }
    } catch { }
}

function Ensure-VenvPython {
    if (-not (Test-Path $venvPython)) {
        Write-Log "Venv Python not found at $venvPython. Self-healing: attempting to recreate venv..." "ERROR"
        Push-Location $appDir
        if (Test-Path ".venv") { Remove-Item ".venv" -Recurse -Force -ErrorAction SilentlyContinue }
        # Prefer Python Launcher (py) with a recent 3.x for wheel compatibility (numpy 2.x, streamlit etc.).
        # Falls back to bare 'python' (may be pythoncore-3.14 which has caused prior numpy/streamlit corruption on this machine).
        $basePy = $null
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $basePy = "py"
            Write-Log "Using 'py' launcher for venv creation."
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            $basePy = "python"
            Write-Log "WARNING: 'py' launcher not found; falling back to bare 'python'. This has previously produced broken venvs (numpy._utils, streamlit import failures) on this system."
        } else {
            Write-Log "No base python found to create venv." "ERROR"
            Pop-Location
            return $false
        }
        & $basePy -m venv .venv 2>&1 | Out-Null
        if (-not (Test-Path $venvPython)) {
            Write-Log "venv creation failed with base '$basePy'." "ERROR"
            Pop-Location
            return $false
        }
        & $venvPython -m pip install --upgrade pip 2>&1 | Out-Null
        if (Test-Path "requirements.txt") {
            & $venvPython -m pip install -r requirements.txt --no-cache-dir 2>&1 | Out-Null
        }
        # Extra pin for streamlit to counter partial/corrupted installs seen in verification runs.
        & $venvPython -m pip install --no-cache-dir "streamlit>=1.30.0,<2.0.0" 2>&1 | Out-Null
        Pop-Location
        if (-not (Test-Path $venvPython)) {
            Write-Log "Failed to recreate venv. MANUAL HEAL (run in pwsh from project root):\ncd financial-report-insights\nif (Test-Path .venv) { rm -r -fo .venv }\npy -m venv .venv   # or use a known-good python.exe that supports the requirements\n.\.venv\Scripts\python -m pip install --upgrade pip\n.\.venv\Scripts\python -m pip install -r requirements.txt\n.\.venv\Scripts\python -m pip install 'streamlit>=1.30,<2' ruff pytest\nThen re-run the launcher or the DEBUG visible shortcut." "ERROR"
            return $false
        }
        Write-Log "Venv recreated successfully."
    }
    return $true
}

function Validate-Imports {
    Write-Log "Validating critical imports (self-healing on failure)..."
    # Include $appDir in sys.path so the import test works even if the ps1 is called from project root (CWD not app dir)
    $validateCode = @"
import sys
import os
sys.path.insert(0, r'$appDir')
print('Python:', sys.executable)
print('CWD:', os.getcwd())
errors = []
try:
    import numpy as np
    print('numpy:', np.__version__)
except Exception as e: errors.append(f'numpy: {e}')
try:
    import pandas as pd
    print('pandas: OK')
except Exception as e: errors.append(f'pandas: {e}')
try:
    import streamlit as st
    print('streamlit:', getattr(st, '__version__', 'present-but-no-version'))
except Exception as e: errors.append(f'streamlit: {e}')
try:
    from app_local import SimpleRAG
    print('SimpleRAG: OK')
except Exception as e: errors.append(f'SimpleRAG: {e}')
if errors:
    print('VALIDATION_ERRORS:' + '|'.join(errors))
    sys.exit(1)
else:
    print('VALIDATION_SUCCESS')
"@
    $tempFile = [System.IO.Path]::GetTempFileName() + ".py"
    $validateCode | Out-File -FilePath $tempFile -Encoding UTF8
    $result = & $venvPython $tempFile 2>&1
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    $output = $result -join "`n"
    Write-Log $output
    if ($output -match "VALIDATION_SUCCESS") {
        return $true
    } else {
        Write-Log "Import validation failed. Attempting self-healing reinstall..." "WARN"
        & $venvPython -m pip install --force-reinstall --no-cache-dir numpy pandas streamlit 2>&1 | ForEach-Object { Write-Log $_ }
        # Re-validate once
        $result2 = & $venvPython $tempFile 2>&1
        if (($result2 -join "`n") -match "VALIDATION_SUCCESS") {
            Write-Log "Self-healing succeeded."
            return $true
        }
        Write-Log "Self-healing failed after reinstall. The base python used to create this .venv may be incompatible (see earlier logs for 'py' vs bare 'python'). Run the MANUAL HEAL command printed above." "ERROR"
        return $false
    }
}

function Ensure-Ollama {
    $ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaProc) {
        Write-Log "Ollama not detected. Attempting to start..." "WARN"
        $ollamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
        if ($ollamaExe) {
            Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 4
            $ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
        }
    }
    if ($ollamaProc) {
        Write-Log "Ollama is running (PID $($ollamaProc.Id))."
    } else {
        Write-Log "Ollama could not be started. App may have limited functionality." "WARN"
        return $false
    }
    # Self-healing: ensure the exact embedding model required by config (mxbai-embed-large, 1024d)
    Write-Log "Ensuring embedding model mxbai-embed-large (required for RAG 1024-dim)..."
    & ollama pull mxbai-embed-large 2>&1 | Out-Null
    Write-Log "Embedding model pull/check complete."
    return $true
}

function Start-Or-Ensure-App {
    param([int]$Port, [int]$MaxWait)
    if ((Test-PortListening -Port $Port) -and (Test-UrlHealthy -Url "http://localhost:$Port") -and -not $ForceRestart) {
        Write-Log "App already healthy on port $Port. Opening browser..."
        return $true
    }
    Write-Log "App not running or force restart requested. Starting Streamlit..."
    # Kill whatever is squatting on the target port for a clean start (interpreter-agnostic).
    Stop-PortOwner -Port $Port
    Start-Sleep -Seconds 1
    $args = @("run", "streamlit_app_local.py", "--server.headless", "true", "--server.port", "$Port")
    $style = if ($Visible) { "Normal" } else { "Hidden" }
    Start-Process -FilePath $venvStreamlit -ArgumentList $args -WorkingDirectory $appDir -WindowStyle $style -ErrorAction SilentlyContinue
    if ($Visible) { Write-Log "Launched Streamlit in VISIBLE mode for debugging (no -WindowStyle Hidden)." }
    # Self-healing wait loop
    for ($i = 0; $i -lt $MaxWait; $i++) {
        if ((Test-PortListening -Port $Port) -and (Test-UrlHealthy -Url "http://localhost:$Port")) {
            Write-Log "App is up after $($i+1) seconds."
            return $true
        }
        Start-Sleep -Seconds 1
        if (($i % 5) -eq 0) { Write-Log "Waiting for app startup... ($i/$MaxWait)" }
    }
    Write-Log "App failed to become healthy after $MaxWait seconds. Attempting one more restart..." "WARN"
    Stop-PortOwner -Port $Port
    Start-Sleep -Seconds 1
    Start-Process -FilePath $venvStreamlit -ArgumentList $args -WorkingDirectory $appDir -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 15
    if ((Test-PortListening -Port $Port) -and (Test-UrlHealthy -Url "http://localhost:$Port")) {
        Write-Log "App recovered on retry."
        return $true
    }
    Write-Log "App launch failed. Check logs and ensure dependencies/Ollama are correct." "ERROR"
    return $false
}

# === MAIN ===
Write-Log "=== Financial Report Insights Launcher Started ==="
if (-not (Ensure-VenvPython)) { exit 1 }
if (-not (Validate-Imports)) { exit 1 }
Ensure-Ollama | Out-Null
$started = Start-Or-Ensure-App -Port $Port -MaxWait $MaxStartupWaitSeconds
if ($started) {
    Write-Log "Opening browser at $url"
    Start-Process $url
    Write-Log "Launch complete. Self-healing checks passed."
} else {
    Write-Log "Launch did not complete successfully. See log: $logFile" "ERROR"
}
Write-Log "=== Launcher Finished ==="
