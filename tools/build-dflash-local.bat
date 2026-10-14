@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"%~dp0..\.venv\Scripts\cmake.exe" --build "%~dp0..\build" --target strata glm_dflash_test -j 12
