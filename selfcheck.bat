@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 自习室座位识别系统 - 安装自检

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if defined SEAT_PYTHON if exist "%SEAT_PYTHON%" set "PYTHON_EXE=%SEAT_PYTHON%"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\seat\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\seat\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\seat\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\seat\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda3\envs\seat\python.exe" set "PYTHON_EXE=D:\miniconda3\envs\seat\python.exe"
if not defined PYTHON_EXE if exist "C:\ProgramData\miniconda3\envs\seat\python.exe" set "PYTHON_EXE=C:\ProgramData\miniconda3\envs\seat\python.exe"
if not defined PYTHON_EXE if exist "C:\ProgramData\anaconda3\envs\seat\python.exe" set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\seat\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
)

echo ============================================================
echo   安装自检（约 1 分钟，首次加载模型会慢一些）
echo   Python: %PYTHON_EXE%
echo ============================================================
echo.

if not defined PYTHON_EXE (
    echo [失败] 没找到 Python，请先安装 Anaconda 并创建 seat 环境。
    pause
    exit /b 1
)

echo [1/4] 检查依赖包 ...
"%PYTHON_EXE%" -c "import ultralytics, fastapi, uvicorn, cv2, yaml, numpy, openpyxl; print('      依赖 OK')"
if errorlevel 1 (
    echo [失败] 依赖不完整，请执行：
    echo        "%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

echo [2/4] 检查权重与数据文件 ...
if not exist "runs\train\seat_v7b\weights\best.pt" (
    echo [失败] 缺少 runs\train\seat_v7b\weights\best.pt
    goto fail
)
if not exist "runs\train\seat_v6\weights\best.pt" (
    echo [失败] 缺少 runs\train\seat_v6\weights\best.pt
    goto fail
)
if not exist "models\yolov8n.pt" (
    echo [失败] 缺少 models\yolov8n.pt
    goto fail
)
if not exist "data\raw_real\real_03.jpg" (
    echo [失败] 缺少 data\raw_real\real_03.jpg
    goto fail
)
echo       权重与数据 OK

echo [3/4] 运行状态机单元测试（应 13 条全过）...
"%PYTHON_EXE%" -m unittest discover -s tests
if errorlevel 1 (
    echo [失败] 单元测试未通过
    goto fail
)

echo [4/4] 标定场景判定 real_03（期望：有人 6 / 有物无人 2 / 空置 3 / 在座 6 人）...
"%PYTHON_EXE%" scripts\judge_seats.py --image data\raw_real\real_03.jpg --weights runs\train\seat_v6\weights\best.pt --conf 0.15 --out-json runs\judge\selftest_real03.json --out-image runs\judge\selftest_real03.jpg
if errorlevel 1 (
    echo [失败] 判定脚本执行失败
    goto fail
)

echo.
echo ============================================================
echo   自检完成。上面汇总与「期望」一致即可交付使用；
echo   标注图见 runs\judge\selftest_real03.jpg
echo ============================================================
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   自检未通过，请把本窗口截图发给项目负责人。
echo ============================================================
pause
exit /b 1
