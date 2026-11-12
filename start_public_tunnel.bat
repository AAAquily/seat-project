@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not defined SEAT_PORT set "SEAT_PORT=8000"
title 公网访问通道 - 端口 %SEAT_PORT%

rem ============================================================
rem  把本机正在运行的服务映射成公网 https 链接（免注册）：
rem  手机用流量 和 异地电脑都能打开。
rem  前提：先用 start_server.bat 把服务跑起来（那个窗口别关）。
rem ============================================================

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if defined SEAT_PYTHON if exist "%SEAT_PYTHON%" set "PYTHON_EXE=%SEAT_PYTHON%"
if not defined PYTHON_EXE if exist "D:\miniconda3\envs\seat\python.exe" set "PYTHON_EXE=D:\miniconda3\envs\seat\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\seat\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\seat\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\seat\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\seat\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
)

echo ============================================================
echo   公网访问通道
echo ============================================================
echo   本机端口 : %SEAT_PORT%
echo   说明     : 运行后会给出一行 https 链接 + 二维码
echo              手机扫码即可打开，无需同一 Wi-Fi
echo   有效期   : 免费通道约 60 分钟，过期重新双击本文件
echo   关闭     : 在本窗口按 Ctrl+C
echo ============================================================
echo.

if not defined PYTHON_EXE (
    echo [错误] 没找到 Python 解释器，请先运行 start_server.bat 检查环境。
    pause
    exit /b 1
)

netstat -ano | findstr /R /C:"TCP.*:%SEAT_PORT% .*LISTENING" >nul
if errorlevel 1 (
    echo [错误] %SEAT_PORT% 端口没有服务在跑。
    echo        请先双击 start_server.bat，等它显示"服务已启动"后再运行本文件。
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" scripts\share_public.py --port %SEAT_PORT%

echo.
echo 公网通道已关闭。
pause
