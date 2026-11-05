@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 自习室座位监测服务 - http://127.0.0.1:8000/

set PYTHON_EXE=D:\miniconda3\envs\seat\python.exe

echo ============================================================
echo   自习室座位占用监测系统 - 本地服务
echo ============================================================
echo   项目目录 : %CD%
echo   访问地址 : http://127.0.0.1:8000/
echo   停止服务 : 在本窗口按 Ctrl+C，或直接关闭窗口
echo ============================================================
echo.

if not exist "%PYTHON_EXE%" (
    echo [错误] 找不到 Python 环境：
    echo        %PYTHON_EXE%
    echo        请确认 conda 的 seat 环境已创建，或修改本文件第 7 行的路径。
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr /R /C:"TCP.*:8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 8000 端口已经被占用，服务可能已经在运行了。
    echo        先试着直接打开 http://127.0.0.1:8000/
    echo        如果需要重启：关闭窗口后运行 stop_server.bat，再双击本文件。
    echo.
    start "" http://127.0.0.1:8000/
    pause
    exit /b 0
)

echo 正在启动，首次启动要加载模型，约 5-10 秒，浏览器会自动打开……
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:8000/"

"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo ============================================================
echo   服务已停止，退出码 %errorlevel%
echo   如果上方有红色报错信息，请截图反馈，不要直接关窗口
echo ============================================================
pause
