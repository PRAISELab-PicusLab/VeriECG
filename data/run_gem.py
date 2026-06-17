#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

OUTPUT_PATH = BASE / "output_gem.json"
TEST_FILE = BASE / "data" / "ptbxl_subset.json"
FILTERED_FILE = BASE / "data" / "ptbxl_subset-RESUME.json"

MAX_SAMPLES_PER_RUN = 5

SCRIPT_PATH = BASE / "external" / "GEM" / "llava" / "eval" / "model_ecg_resume.py"

while True:

    print("\n" + "="*80)
    print("🔄 NUOVA ITERAZIONE")
    print("="*80)

    # 1. Leggi ID già processati
    processed_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        processed_ids.add(result['question_id'])
                    except:
                        pass

    # 2. Carica dataset completo
    with open(TEST_FILE, 'r') as f:
        full_dataset = json.load(f)

    # 3. Filtra NON processati
    remaining_dataset = [
        item for item in full_dataset
        if item['id'] not in processed_ids
    ]

    if len(remaining_dataset) == 0:
        print("\n🎉 COMPLETATO: tutti gli ECG processati!")
        break

    # 4. Limita batch
    limited_dataset = remaining_dataset[:MAX_SAMPLES_PER_RUN]

    # 5. Salva batch
    with open(FILTERED_FILE, 'w') as f:
        json.dump(limited_dataset, f, indent=2)

    print(f"\n📊 Processati: {len(processed_ids)}/{len(full_dataset)}")
    print(f"🔄 Rimasti: {len(remaining_dataset)}")
    print(f"📦 Batch corrente: {len(limited_dataset)}")

    # 6. ESECUZIONE GEM
    cmd = [
        "python", "-u",
        str(SCRIPT_PATH),
        "--model-path", str(BASE / "models" / "GEM_model"),
        "--question-file", str(FILTERED_FILE),
        "--answers-file", str(OUTPUT_PATH),
        "--ecg-folder", str(BASE / "data" / "ptbxl_subset"),
        "--image-folder", str(BASE / "data" / "ecg_images" / "gen_images")
    ]

    print("\n🚀 Avvio GEM...\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line.strip())

    process.wait()

    print("\n✅ Batch completato!")
    time.sleep(2)
