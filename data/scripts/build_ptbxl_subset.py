#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crea un subset di PTB-XL contenente solo gli ECG presenti nel benchmark
ecg-grounding-test-ptbxl.json.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm


# ============================================================
# PATH DI BASE
# ============================================================

BASE = Path(__file__).resolve().parents[1]

JSON_PATH = BASE / "data" / "ecg_bench" / "ecg-grounding-test-ptbxl.json"

# cartella con i dati PTB-XL originali (records500)
TS_BASE = BASE / "data" / "ecg_timeseries" / "ptbxl" / "records500"

# cartella di output del subset
OUT_DIR = BASE / "data" / "ptbxl_subset"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARICA JSON
# ============================================================

with open(JSON_PATH) as f:
    data = json.load(f)

errors = []
copied = 0


# ============================================================
# LOOP SUGLI ECG
# ============================================================

for item in tqdm(data):

    try:
        # es: ecg_ptbxl_benchmarking/.../19000/19208_hr
        ecg_path = item["ecg"]

        # prendi solo la parte dopo "ptbxl/"
        relative = ecg_path.split("ptbxl/")[-1]

        # costruisci il path completo
        full_path = TS_BASE / relative.replace("records500/", "")
        full_path = str(full_path).replace(".dat", "").replace(".hea", "")

        dat_file = full_path + ".dat"
        hea_file = full_path + ".hea"

        # verifica esistenza
        if not Path(dat_file).exists():
            raise FileNotFoundError(dat_file)

        # copia nel subset
        shutil.copy(dat_file, OUT_DIR)
        shutil.copy(hea_file, OUT_DIR)

        copied += 1

    except Exception as e:
        errors.append(str(e))


# ============================================================
# RISULTATI
# ============================================================

print(f"Copiati: {copied}")
print(f"Errori: {len(errors)}")
