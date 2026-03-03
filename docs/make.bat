@ECHO OFF

REM Minimal Sphinx make.bat
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" goto help

if "%1"=="clean" goto clean
if "%1"=="html" goto html
if "%1"=="linkcheck" goto linkcheck

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:html
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR%
goto end

:linkcheck
%SPHINXBUILD% -M linkcheck %SOURCEDIR% %BUILDDIR%
goto end

:end
