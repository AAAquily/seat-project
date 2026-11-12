@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
rem 端口可用环境变量覆盖：set SEAT_PORT=8011 后再双击本文件
if not defined SEAT_PORT set "SEAT_PORT=8000"
rem 监听地址：0.0.0.0 = 同一 Wi-Fi 下的手机/平板/其他电脑都能访问
rem          127.0.0.1 = 仅本机可访问（更安全，但其他设备打不开）
if not defined SEAT_HOST set "SEAT_HOST=0.0.0.0"
title 自习室座位监测服务 - 端口 %SEAT_PORT%

rem ============================================================
rem  Python 解释器自动探测（换机器不用改脚本）
rem  优先级：本目录 .venv 和 环境变量 SEAT_PYTHON 和 conda 的 seat 环境
rem           最后才用 PATH 里的 python
rem  若都不合适，可取消下面一行的注释并写死绝对路径：
rem  set "PYTHON_EXE=D:\miniconda3\envs\seat\python.exe"
rem ============================================================
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

rem 取本机局域网 IP：解析 ipconfig 的 IPv4 行（不嵌套引号，避免 for /f 解析失败）
set "LAN_IP="
if not "%SEAT_HOST%"=="127.0.0.1" (
    for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /c:"IPv4"') do if not defined LAN_IP set "LAN_IP=%%i"
)
if defined LAN_IP set "LAN_IP=%LAN_IP: =%"

echo ============================================================
echo   自习室座位占用监测系统 - 本地服务
echo ============================================================
echo   项目目录 : %CD%
echo   Python   : %PYTHON_EXE%
echo   本机访问 : http://127.0.0.1:%SEAT_PORT%/
if defined LAN_IP echo   其他设备 : http://!LAN_IP!:%SEAT_PORT%/
if defined LAN_IP echo              同一 Wi-Fi 下，手机/平板输入上面地址即可
if not defined LAN_IP echo   其他设备 : 没取到局域网 IP，请用 ipconfig 查看 IPv4 后拼上 :%SEAT_PORT%
echo   停止服务 : 在本窗口按 Ctrl+C，或直接关闭窗口
echo ============================================================
echo.

if not defined PYTHON_EXE (
    echo [错误] 没有找到可用的 Python 解释器。
    echo        请先按《交接说明》安装 Anaconda 并创建 seat 环境，
    echo        或设置环境变量 SEAT_PYTHON 指向 python.exe。
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import ultralytics, fastapi, uvicorn, cv2, yaml" >nul 2>&1
if errorlevel 1 (
    echo [错误] 这个 Python 缺少依赖，无法启动服务：
    echo        %PYTHON_EXE%
    echo        请先执行：
    echo        "%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr /R /C:"TCP.*:%SEAT_PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo [提示] %SEAT_PORT% 端口已经被占用，服务可能已经在运行了。
    echo        先试着直接打开 http://127.0.0.1:%SEAT_PORT%/
    if defined LAN_IP echo        其他设备打开 http://!LAN_IP!:%SEAT_PORT%/
    echo        如果需要重启：关闭窗口后运行 stop_server.bat，再双击本文件。
    echo.
    start "" http://127.0.0.1:%SEAT_PORT%/
    pause
    exit /b 0
)

echo 正在启动，首次启动要加载模型，约 5-10 秒，浏览器会自动打开……
start "" /min cmd /c "timeout /t 6 >nul & start http://127.0.0.1:%SEAT_PORT%/"

"%PYTHON_EXE%" -m uvicorn app.main:app --host %SEAT_HOST% --port %SEAT_PORT%

echo.
echo ============================================================
echo   服务已停止，退出码 %errorlevel%
echo 如果上方有红色报错信息，请截图反馈，不要直接关窗口
echo ============================================================
pause
