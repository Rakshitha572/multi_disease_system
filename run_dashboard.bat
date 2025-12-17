@echo off
cd /d C:\Projects\multi-disease-prediction
call conda activate base
start http://127.0.0.1:5000
python src\webapp\app.py
pause
