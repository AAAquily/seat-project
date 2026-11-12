@echo off
chcp 936 >nul
setlocal
cd /d "%~dp0"

rem ============================================================
rem  在 Windows 防火墙里放行服务端口，让同一 Wi-Fi 下的
rem  手机 / 平板 / 其他电脑能访问。
rem  双击本文件即可，会自动弹出管理员授权（UAC），点"是"。
rem
rem  作用：新增一条入站规则 SeatProject-PORT，PORT 为端口号
rem  撤销：netsh advfirewall firewall delete rule name="SeatProject-8000"
rem ============================================================

if "%~1"=="" (set "PORT=8000") else (set "PORT=%~1")

rem 没有管理员权限时，自我提权重启一次（会弹 UAC）
net session >nul 2>&1
if errorlevel 1 (
    echo 需要管理员权限，正在请求提权……请在弹窗里点"是"。
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '%PORT%' -Verb RunAs"
    exit /b 0
)

echo ============================================================
echo   放行 TCP 端口 %PORT%（局域网访问）
echo ============================================================
netsh advfirewall firewall delete rule name="SeatProject-%PORT%" >nul 2>&1
netsh advfirewall firewall add rule name="SeatProject-%PORT%" dir=in action=allow protocol=TCP localport=%PORT% profile=any
if errorlevel 1 (
    echo [失败] 规则添加失败，请确认是在管理员窗口里运行。
) else (
    echo [完成] 已允许入站 TCP %PORT%，所有网络类型（专用/公用）生效。
    echo.
    echo 下一步：双击 start_server.bat 启动服务，
    echo         手机连同一个 Wi-Fi，打开窗口里显示的"其他设备"地址。
)

echo.
pause
