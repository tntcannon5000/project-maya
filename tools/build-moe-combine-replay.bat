@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"%~dp0..\.venv\Scripts\cmake.exe" --build "%~dp0..\build" --target maya_verify_scratch_test maya_moe_combine_replay -j 12
