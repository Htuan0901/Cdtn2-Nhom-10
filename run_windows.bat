@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo [1/3] Tao moi truong ao...
  python -m venv .venv
  call .venv\Scripts\activate
  echo [2/3] Cai thu vien...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
echo [3/3] Khoi dong ung dung...
streamlit run app.py
pause
