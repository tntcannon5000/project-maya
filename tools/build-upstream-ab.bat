@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "MAYA_ROOT=%~dp0.."
set "MAYA_BASE=%~dp0..\..\project-maya-upstream-v1.1.7"
"%MAYA_ROOT%\.venv\Scripts\cmake.exe" -G Ninja "-DCMAKE_MAKE_PROGRAM=%MAYA_ROOT%\.venv\Scripts\ninja.exe" -S "%MAYA_BASE%" -B "%MAYA_BASE%\build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.EXE" "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" "-DSTRATA_GGML_DIR=%MAYA_ROOT%\third_party\llama.cpp" || exit /b 1
"%MAYA_ROOT%\.venv\Scripts\cmake.exe" --build "%MAYA_BASE%\build" --target strata maya_ab_192k -j 12
