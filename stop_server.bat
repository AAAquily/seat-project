@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 停止自习室监测服务

echo 正在查找占用 8000 端口的进程...
set FOUND=0

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:"TCP.*:8000 .*LISTENING"') do (
    echo   结束进程 PID %%p
    taskkill /F /PID %%p >nul 2>&1
    set FOUND=1
)

if "%FOUND%"=="0" (
    echo   没有发现正在运行的服务，8000 端口是空闲的。
) else (
    echo   已停止。
)

echo.
pause
