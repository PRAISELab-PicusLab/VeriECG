#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup di sessione:
- installa le dipendenze da requirements.txt
- installa GEM come pacchetto Python (pip install -e)
- applica la patch write -> append in model_ecg_resume.py
"""

import subprocess
from pathlib import Path
import fileinput


# ============================================================
# PATH DI BASE
# ============================================================

BASE = Path(__file__).resolve().parents[1]
GEM_DIR = BASE / "external" / "GEM"
REQ_FILE = BASE / "requirements.txt"


# ============================================================
# 1. INSTALLA LE DIPENDENZE
# ============================================================


subprocess.run(
    ["pip", "install", "-r", str(REQ_FILE)],
    check=True
)

print("✔️ Dipendenze installate\n")


# ============================================================
# 2. INSTALLA GEM COME PACCHETTO PYTHON
# ============================================================

subprocess.run(
    ["pip", "install", "-e", str(GEM_DIR), "--no-deps"],
    check=True
)

print("✔️ GEM installato correttamente\n")


# ============================================================
# 3. write -> append in model_ecg_resume.py
# ============================================================

script_path = GEM_DIR / "llava" / "eval" / "model_ecg_resume.py"

for line in fileinput.input(script_path, inplace=True):
    if "open(args.answers_file" in line:
        line = line.replace('"w"', '"a"')
    print(line, end="")

print("✔️ Patch applicata\n")
