@echo off
setlocal
title Chatterbox Long-Form UI

rem ------------------------------------------------------------
rem Chatterbox Long-Form UI - Windows launcher
rem
rem Application directory:
rem     C:\AI\chatterbox
rem
rem Virtual environment:
rem     C:\AI\chatterbox\venv
rem ------------------------------------------------------------

set "APP_DIR=C:\AI\chatterbox"
set "VENV=%APP_DIR%\venv"

if not exist "%APP_DIR%" (
    echo.
    echo ERROR: Application directory was not found:
    echo     %APP_DIR%
    echo.
    pause
    exit /b 1
)

if not exist "%VENV%\Scripts\activate.bat" (
    echo.
    echo ERROR: Chatterbox virtual environment was not found:
    echo     %VENV%
    echo.
    pause
    exit /b 1
)

pushd "%APP_DIR%"

call "%VENV%\Scripts\activate.bat"

:menu
cls
echo ============================================================
echo               Chatterbox Long-Form UI
echo ============================================================
echo.
echo Application folder:
echo   %APP_DIR%
echo.
echo Virtual environment:
echo   %VENV%
echo.
echo   [1] Launch Chatterbox Turbo Long-Form UI
echo   [2] Launch Chatterbox Multilingual V3 Long-Form UI
echo   [3] Open activated command prompt
echo   [Q] Quit
echo.
choice /C 123Q /N /M "Select an option: "

if errorlevel 4 goto quit
if errorlevel 3 goto shell
if errorlevel 2 goto v3
if errorlevel 1 goto turbo

:turbo
cls
echo Starting Chatterbox Turbo Long-Form UI...
echo.
python "%APP_DIR%\longform_turbo_gui.py"
echo.
echo Chatterbox Turbo has closed.
pause
goto menu

:v3
cls
echo Starting Chatterbox Multilingual V3 Long-Form UI...
echo.
python "%APP_DIR%\longform_v3_gui.py"
echo.
echo Chatterbox Multilingual V3 has closed.
pause
goto menu

:shell
cls
echo Chatterbox environment activated.
echo Type EXIT to return to the launcher.
echo.
cmd /k
goto menu

:quit
popd
endlocal
exit /b 0
