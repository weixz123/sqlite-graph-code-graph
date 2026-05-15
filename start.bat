@echo off
setlocal EnableExtensions
title Code Graph System

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%" || (
    echo [ERROR] Failed to enter project directory: %PROJECT_DIR%
    pause
    exit /b 1
)

echo ================================
echo   Code Graph System - Startup
echo ================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv was not found in PATH.
    echo Install uv first: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] .venv was not found. Running uv sync...
    uv sync
    if errorlevel 1 (
        echo [ERROR] uv sync failed.
        pause
        exit /b 1
    )
    echo [OK] Environment initialized.
    echo.
)

if not exist "query_sessions" mkdir "query_sessions"

echo [INFO] Checking port 9961...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":9961 " ^| findstr "LISTEN"') do (
    echo [INFO] Stopping existing process on port 9961: PID=%%P
    taskkill /PID %%P /F >nul 2>&1
)

echo [INFO] Starting Flask Web UI...
start "Code Graph - Web UI" /D "%PROJECT_DIR%" cmd /k uv run python graph_gui.py

echo [INFO] Waiting for Web UI at http://127.0.0.1:9961 ...
set /a WAIT_COUNT=0

:waitloop
".venv\Scripts\python.exe" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9961/', timeout=2)" >nul 2>&1
if not errorlevel 1 goto ready

set /a WAIT_COUNT+=1
if %WAIT_COUNT% GEQ 30 (
    echo [ERROR] Web UI startup timed out after 30 seconds.
    echo Check the "Code Graph - Web UI" window for Python errors.
    pause
    exit /b 1
)

timeout /t 1 /nobreak >nul
goto waitloop

:ready
echo [OK] Web UI is ready.
if not "%CODE_GRAPH_NO_BROWSER%"=="1" start "" "http://127.0.0.1:9961"

echo.
echo ================================
echo   Service started
echo   Web UI: http://127.0.0.1:9961
echo.
echo   MCP client config:
echo   command: uv
echo   args: ["--directory", "%PROJECT_DIR%", "run", "code_graph_mcp_server.py", "--db", "%PROJECT_DIR%code_graph.sqlite"]
echo ================================
echo.
echo Press Ctrl+C in this window to stop monitoring.
echo Close the "Code Graph - Web UI" window to stop the Web UI.
echo.

if "%CODE_GRAPH_NO_MONITOR%"=="1" exit /b 0

:keepalive
timeout /t 30 /nobreak >nul
netstat -ano | findstr ":9961 " | findstr "LISTEN" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Web UI is no longer listening on port 9961.
    exit /b 0
)
goto keepalive
