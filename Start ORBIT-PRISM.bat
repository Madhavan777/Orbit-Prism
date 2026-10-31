@echo off
cd /d %~dp0
set PY=python
if exist "C:\Users\MG-Laptop\rvenv\Scripts\python.exe" set PY=C:\Users\MG-Laptop\rvenv\Scripts\python.exe
start "" http://localhost:8000
%PY% -m uvicorn backend.api:app --port 8000
