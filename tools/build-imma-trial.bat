@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"%~dp0..\.venv\Scripts\cmake.exe" -S "%~dp0..\tests\imma_q4" -B "%~dp0..\build-imma-q4" -G Ninja -DCMAKE_MAKE_PROGRAM="%~dp0..\.venv\Scripts\ninja.exe" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
if errorlevel 1 exit /b %errorlevel%
"%~dp0..\.venv\Scripts\cmake.exe" --build "%~dp0..\build-imma-q4" -j 8

