@echo off
chcp 65001 >nul
cd /d D:\seat-project

rem Launch LabelImg with full output redirected to a log file,
rem so a crash leaves a readable traceback instead of vanishing.

call D:\miniconda3\Scripts\activate.bat seat

if not exist logs mkdir logs

echo [1/2] Starting LabelImg in debug mode. Log file: logs\labelimg_run.log
labelImg data\workspace\images data\workspace\labels\classes.txt data\workspace\labels > logs\labelimg_run.log 2>&1

echo [2/2] LabelImg exited. Last lines of the log:
echo ------------------------------------------------------------
powershell -NoProfile -Command "Get-Content logs\labelimg_run.log -Tail 25"
echo ------------------------------------------------------------
pause
