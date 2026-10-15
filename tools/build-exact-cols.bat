@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"%~dp0..\.venv\Scripts\cmake.exe" --build "%~dp0..\build" --target strata glm_mv_nt_test maya_verify_scratch_test -j 12
