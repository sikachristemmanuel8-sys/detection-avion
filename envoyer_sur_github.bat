@echo off
rem Double-clic pour envoyer le projet sur GitHub (une fenêtre de connexion peut s'ouvrir).
cd /d "%~dp0"
echo Envoi du projet sur GitHub...
"C:\Program Files\Git\cmd\git.exe" add -A
"C:\Program Files\Git\cmd\git.exe" commit -m "Mise a jour du projet" >nul 2>&1
"C:\Program Files\Git\cmd\git.exe" push -u origin main
echo.
echo Si tu vois "main -> main" ci-dessus, c'est envoye :
echo https://github.com/sikachristemmanuel8-sys/detection-avions
pause
