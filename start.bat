@echo off
chcp 65001 >nul
rem ============================================================
rem 自习室座位占用识别系统 —— 一键启动
rem 双击本文件即可启动服务（自动激活 seat 环境）
rem ============================================================

echo [1/3] 激活 conda 环境 seat ...
call conda activate seat
if errorlevel 1 (
    echo [错误] 环境激活失败，请确认已安装 Anaconda 且存在 seat 环境
    pause
    exit /b 1
)

echo [2/3] 进入项目目录 ...
cd /d %~dp0

echo [3/3] 启动服务，浏览器访问 http://127.0.0.1:8000 ...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
