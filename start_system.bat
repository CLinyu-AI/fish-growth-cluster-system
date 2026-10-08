@echo off
chcp 65001 >nul
title 鱼类生长特征分析系统

echo =====================================
echo    鱼类生长特征分析系统启动器
echo =====================================

:: 端口
set PORT=5000

:: conda环境
set ENV=flask_pytorch

echo.
echo [1/4] 激活 Conda 环境...

call conda activate %ENV%
if errorlevel 1 (
echo.
echo ❌ 未找到环境 %ENV%
pause
exit
)

echo.
echo [2/4] 启动 Flask 服务...

start "FishSystem" /min cmd /c "python app.py"

echo.
echo [3/4] 等待系统启动...

set /a wait=0

:check

timeout /t 1 >nul
set /a wait+=1

netstat -ano | findstr ":%PORT%" >nul

if %errorlevel%==0 (
goto open
)

if %wait% GEQ 20 (
echo ❌ 启动超时
pause
exit
)

goto check


:open

echo.
echo ✅ 系统启动成功
echo.

echo 访问地址:
echo http://127.0.0.1:%PORT%

start http://127.0.0.1:%PORT%

echo.
echo 按任意键关闭系统
pause >nul

echo.
echo 正在关闭服务...

taskkill /f /im python.exe /fi "WINDOWTITLE eq FishSystem" >nul

echo 服务已关闭

exit