@echo off
rem Double-clic : construit le conteneur Docker (la 1re fois ~10 min) puis lance l'application.
rem Docker Desktop doit être ouvert. Pour arrêter : fermer cette fenêtre.
cd /d "%~dp0"
echo === 1/2 Construction du conteneur (rapide si deja fait) ===
docker build -t detection-avions .
if errorlevel 1 (
  echo.
  echo ERREUR : la construction a echoue. Docker Desktop est-il bien ouvert ?
  pause
  exit /b 1
)
echo.
echo === 2/2 Lancement : ouvre http://localhost:8501 dans ton navigateur ===
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8501"
docker run --rm -p 8501:8501 --name detection-avions detection-avions
pause
