@echo off
rem ============================================================
rem  自习座位监测服务 - 后台托管启动脚本
rem  供 Windows 任务计划程序或其他脱离会话的进程调用，
rem  不带交互（无 pause），输出追加写入 logs\uvicorn_service.log
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_EXE=D:\miniconda3\envs\seat\python.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

if not exist "%PYTHON_EXE%" (
    echo [ERROR] python not found: %PYTHON_EXE% >> logs\uvicorn_service.log
    exit /b 1
)

if not exist "logs" mkdir "logs"

echo ================ start %DATE% %TIME% ================ >> logs\uvicorn_service.log
"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >> logs\uvicorn_service.log 2>&1
echo ================ exit  %DATE% %TIME% code=%errorlevel% ================ >> logs\uvicorn_service.log
