@echo off
chcp 65001 >nul
cd /d D:\seat-project

rem Activate the seat conda environment, then launch LabelImg
rem with the workspace images, class list, and label save directory.
call D:\miniconda3\Scripts\activate.bat seat

echo [1/2] Starting LabelImg (YOLO mode, remember to switch format to YOLO) ...
labelImg data\workspace\images data\workspace\classes.txt data\workspace\labels

echo [2/2] LabelImg closed. Labels are in data\workspace\labels
pause
