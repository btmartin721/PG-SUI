@echo on
setlocal enabledelayedexpansion
"%PYTHON%" -m pip install . -vv
if errorlevel 1 exit 1
