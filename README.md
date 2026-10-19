# ML-Term-Project
optimization methods for training ml models

Use the project's virtual environment so the scripts and installed packages use
the same Python interpreter. From this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

If `.venv` does not exist, create it with an installed Python interpreter first:

```powershell
py -3.14 -m venv .venv
```

If Windows' `py` or `python` aliases fail, use the full path to the installed
Python executable to create the environment. Once `.venv` exists, the commands
above do not depend on those aliases or PowerShell activation permissions.

In IntelliJ with the Python plugin enabled, use
`.venv\Scripts\python.exe` as the project interpreter. The **Train** run
configuration runs `train.py` with that interpreter and the project working
directory. **Main** runs the FastF1 example, and **Download OpenF1 Data** regenerates
the training CSV files. Fully exit and restart IntelliJ after changing its SDK
registration externally, so the application reloads its interpreter registry.

`train.py` reads `bahrain_2024_clean.csv` from this folder. To download and
regenerate the CSV files, run:

```powershell
.\.venv\Scripts\python.exe openf1_test.py
```

The downloader needs internet access. `main.py` is a separate FastF1 data-loading
example; training uses the cleaned CSV and can run offline.
