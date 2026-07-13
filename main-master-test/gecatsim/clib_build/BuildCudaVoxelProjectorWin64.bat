@echo off
setlocal

set SCRIPT_DIR=%~dp0
set SRC=%SCRIPT_DIR%src\voxelized_projector_cuda.cu
set OUT_DIR=%SCRIPT_DIR%..\lib
set OUT=%OUT_DIR%\catsim_voxel_projector_cuda.dll

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

where nvcc >nul 2>nul
if errorlevel 1 (
    echo ERROR: nvcc was not found. Install CUDA Toolkit and ensure nvcc is on PATH.
    exit /b 1
)

where cl >nul 2>nul
if not errorlevel 1 goto have_cl

set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto check_cl

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL goto check_cl

if exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"

:check_cl
where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe was not found. Install Visual Studio Build Tools with the C++ workload.
    exit /b 1
)

:have_cl
nvcc -O3 --use_fast_math -Xcompiler="/MD" -shared "%SRC%" -o "%OUT%"
if errorlevel 1 (
    echo ERROR: CUDA voxel projector build failed.
    exit /b 1
)

echo Built %OUT%
endlocal
