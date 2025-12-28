@echo off
echo ============================================================
echo    SecureSwipe AI - Starting Application
echo ============================================================
echo.
echo Step 1: Installing dependencies (if not already installed)...
pip install -r requirements.txt
echo.
echo Step 2: Starting Flask server...
echo.
echo The application will be available at: http://localhost:5000
echo.
echo Press CTRL+C to stop the server when done.
echo ============================================================
echo.
python app.py
pause

