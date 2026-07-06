#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

ECG_FOLDER = BASE / "data" / "ptbxl_subset"
OUTPUT_JSON = BASE / "data" / "ptbxl_subset.json"

PROMPT = (
    "Interpret the provided ECG image, identify key features and abnormalities "
    "in each lead, and generate a clinical diagnosis that is supported by the observed evidence"
)

data = []

for root, dirs, files in os.walk(ECG_FOLDER):
    for file in files:
        if file.endswith(".hea"):
            base = file.replace(".hea", "")
            rel_path = Path(root).joinpath(base).relative_to(ECG_FOLDER)

            image_name = base + "-0.png"

            data.append({
                "id": base,
                "ecg": str(rel_path),
                "image": f"ptb-xl-gen/{image_name}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "assistant", "value": ""}
                ]
            })

with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Creati {len(data)} ECG")
