@echo off
setlocal

rem Get the directory path where the batch file is located
set "script_dir=%~dp0"

rem Check if the virtual environment already exists in the script's directory
if exist "%script_dir%MASSQL_API\Scripts\activate.bat" (
    echo MASSQL_API already exists. Activating...
    call "%script_dir%MASSQL_API\Scripts\activate.bat"
) else (
    echo Creating virtual environment in "%script_dir%MASSQL_API"...

    rem Try to create venv with "python" (works even when "py" is not available)
    python -m venv "%script_dir%MASSQL_API"
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment. Check if Python is installed and in PATH.
        pause
        goto :eof
    )

    call "%script_dir%MASSQL_API\Scripts\activate.bat"

    echo Installing the requirements...
    call "%script_dir%MASSQL_API\Scripts\python.exe" -m pip install --upgrade pip
    call "%script_dir%MASSQL_API\Scripts\python.exe" -m pip install -r "%script_dir%requirements.txt"
    
    echo Installing Jupyter Notebook with compatible packages...
    call "%script_dir%MASSQL_API\Scripts\python.exe" -m pip install jupyter notebook==6.5.2 traitlets==5.9.0 ipython==7.31.1
)

rem Start Jupyter Notebook
call "%script_dir%MASSQL_API\Scripts\jupyter-notebook.exe"

pause
