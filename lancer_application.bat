@echo off
rem Double-clic pour lancer l'application (Windows). Elle s'ouvre dans le navigateur.
rem Pour l'arrêter : fermer cette fenêtre noire.
cd /d "%~dp0"
echo Demarrage de l'application... le navigateur va s'ouvrir dans quelques secondes.
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run 08_application.py --server.headless true --browser.gatherUsageStats false
pause
