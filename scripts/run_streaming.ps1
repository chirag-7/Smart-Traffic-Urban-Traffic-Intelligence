# Phase 3 launcher — start all 5 streaming jobs as independent host processes.
#
# Why independent processes:
# Each branch gets its own JVM driver, checkpoint, and process boundary, so a
# crash in one (e.g. weather API schema change) does not kill the others.
#
# Heads-up on resources:
# Each Spark driver consumes ~1.5-2 GB resident RAM. Running all 5 on a 16 GB
# laptop with Docker also up is feasible but tight. For dev iteration prefer
# stream_processor.py (single process, all branches) instead.
#
# Usage (from repo root, with venv activated):
#   .\scripts\run_streaming.ps1                # start all 5
#   .\scripts\run_streaming.ps1 -Only sensors  # start just the sensors job
#   .\scripts\run_streaming.ps1 -StopAll       # kill any running stream_* python processes

param(
    [string]$Only = "",
    [switch]$StopAll = $false
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "venv python not found at $VenvPython. Activate or create the venv first."
}

$jobs = @(
    @{ Name = "sensors";     Script = "spark_jobs\stream_sensors.py" },
    @{ Name = "cctv";        Script = "spark_jobs\stream_cctv.py" },
    @{ Name = "gps";         Script = "spark_jobs\stream_gps.py" },
    @{ Name = "weather";     Script = "spark_jobs\stream_weather.py" },
    @{ Name = "predictions"; Script = "spark_jobs\stream_predictions.py" }
)

if ($StopAll) {
    Write-Host "Stopping any running stream_*.py Python processes..." -ForegroundColor Yellow
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "spark_jobs\\stream_" } |
        ForEach-Object {
            Write-Host "  Stopping PID $($_.ProcessId): $($_.CommandLine)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Write-Host "Done."
    return
}

if ($Only -ne "") {
    $jobs = $jobs | Where-Object { $_.Name -eq $Only }
    if (-not $jobs) {
        Write-Error "Unknown job '$Only'. Choices: sensors, cctv, gps, weather, predictions."
    }
}

$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

foreach ($job in $jobs) {
    $scriptPath = Join-Path $RepoRoot $job.Script
    $logPath = Join-Path $LogDir ("stream_" + $job.Name + ".log")

    Write-Host "Launching $($job.Name) -> $logPath" -ForegroundColor Green
    Start-Process -FilePath $VenvPython `
                  -ArgumentList $scriptPath `
                  -WorkingDirectory $RepoRoot `
                  -WindowStyle Hidden `
                  -RedirectStandardOutput $logPath `
                  -RedirectStandardError ($logPath + ".err")
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "All requested streams launched in the background." -ForegroundColor Green
Write-Host "Logs in: $LogDir"
Write-Host "Tail one with:  Get-Content $LogDir\stream_sensors.log -Wait -Tail 30"
Write-Host "Stop all with:  .\scripts\run_streaming.ps1 -StopAll"
