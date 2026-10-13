@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"%~dp0..\.venv\Scripts\cmake.exe" -S "%~dp0.." -B "%~dp0..\build" -DSTRATA_BUILD_TESTS=ON
"%~dp0..\.venv\Scripts\cmake.exe" --build "%~dp0..\build" --target q2_pair_parity native_split_test -j 8
