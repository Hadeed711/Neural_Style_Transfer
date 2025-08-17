@echo off
echo Starting AI Neural Style Transfer Studio...
echo.
echo Opening in your default web browser...
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py --server.headless false --server.enableCORS false --server.enableXsrfProtection false
pause
