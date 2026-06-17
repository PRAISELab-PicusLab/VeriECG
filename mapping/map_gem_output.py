#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mapping dell'output GEM:
- legge output_gem.json 
- chiama LLM NVIDIA per estrarre osservazioni e diagnosi
- normalizza gli ID
- salva output strutturato
"""

import json
import requests
import re
import time
from pathlib import Path

# ============================================================
# MODEL 
# ============================================================

MODEL = "meta/llama-3.3-70b-instruct" 
MODEL_SAFE = MODEL.replace("/", "_")

# ============================================================
# PATH DI BASE
# ============================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE / "data" / "output_gem.json"
OUTPUT_FILE = BASE / "data" / f"output_gem_structured_{MODEL_SAFE}.json"
OBS_FILE = BASE / "mapping" / "OBS_LIST.json"
DIAG_FILE = BASE / "mapping" / "DIAG_LIST.json"

API_KEY = "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" # INSERT API KEY
MAX_TEST = 5   # None per processare tutto

# ============================================================
# CARICA DIZIONARI
# ============================================================

with open(OBS_FILE) as f:
    osservazioni = json.load(f)

with open(DIAG_FILE) as f:
    diagnosi = json.load(f)

OBS_IDS_SET = {o["id"].strip().upper() for o in osservazioni}
DIAG_IDS_SET = {d["diagnosis_id"].strip().upper() for d in diagnosi}

OBS_LIST_STR = "\n".join(
    f"{o['id']}: {o.get('description','')}" for o in osservazioni
)

DIAG_LIST_STR = ", ".join(d["diagnosis_id"] for d in diagnosi)

# ============================================================
# PROMPT
# ============================================================

PROMPT_OBS = """
Extract ECG observations from the report by analyzing each lead separately.

Return ONLY JSON.
Do NOT include explanations.

VERY IMPORTANT:
- The report is structured by ECG leads (Lead I, Lead II, Lead III, aVR, aVL, aVF, V1–V6).
- For EACH lead, identify ONLY the observations explicitly mentioned in that lead.
- You MUST map each observation to the EXACT OBS ID provided below.
- Do NOT infer observations.
- Do NOT include observations that are stated as absent or normal unless explicitly described.
- Do NOT generalize across leads.
- Only extract what is explicitly written.
- If a lead is not mentioned, do not assign observations to it.

Allowed OBS IDs:
{OBS_LIST}

Output format:
{
  "observations": [
    "OBS_ID_LEAD_X",
    "OBS_ID_LEAD_Y"
  ]
}

Report:
{text}
"""

PROMPT_DIAG =  """
Extract ECG diagnoses explicitly stated in the report.

Return ONLY JSON.

VERY IMPORTANT:
- Only include diagnoses explicitly confirmed in the text.
- Do NOT infer diagnoses.
- If a condition is explicitly excluded, do NOT include it.

Allowed DIAG IDs:
{DIAG_LIST}

Output format:
{"diagnoses": ["DIAG_ID"]}

Report:
{text}
"""

# ============================================================
# CHIAMATA LLM NVIDIA
# ============================================================

def call_llm(prompt):

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2048
    }

    r = requests.post(url, headers=headers, json=payload, timeout=240)

    if r.status_code != 200:
        print("❌ API ERROR:", r.status_code, r.text)
        return ""

    return r.json()["choices"][0]["message"]["content"]

# ============================================================
# ESTRAZIONE JSON
# ============================================================

def extract_json(text):
    candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
    for c in candidates:
        try:
            json.loads(c)
            return c
        except:
            pass
    return None

# ============================================================
# NORMALIZZAZIONE
# ============================================================

def normalize_obs(parsed):
    clean = []
    for o in parsed.get("observations", []):
        obs_id = None
        if isinstance(o, dict):
            obs_id = o.get("id") or o.get("feature") or o.get("observation")
            if not obs_id and len(o) == 1:
                obs_id = list(o.keys())[0]
        elif isinstance(o, str):
            obs_id = o
        if obs_id:
            obs_id = obs_id.strip().upper()
            if obs_id in OBS_IDS_SET:
                clean.append(obs_id)
    return list(set(clean))

def normalize_diag(parsed):
    clean = []
    for d in parsed.get("diagnoses", []):
        diag_id = None
        if isinstance(d, dict):
            diag_id = d.get("id") or d.get("diagnosis")
        elif isinstance(d, str):
            diag_id = d
        if diag_id:
            diag_id = diag_id.strip().upper()
            if diag_id in DIAG_IDS_SET:
                clean.append(diag_id)
    return list(set(clean))

# ============================================================
# PIPELINE
# ============================================================

results = []

with open(INPUT_FILE) as f:
    for i, line in enumerate(f):

        if MAX_TEST and i >= MAX_TEST:
            break

        if not line.strip():
            continue

        data = json.loads(line)
        ecg_id = data["question_id"]
        report = data["text"][:1500]

        print(f"\n🔄 {i} → {ecg_id}")

        # ===== OBS =====
        prompt_obs = PROMPT_OBS.replace("{OBS_LIST}", OBS_LIST_STR).replace("{text}", report)
        raw_obs = call_llm(prompt_obs)

        observations = []
        if raw_obs:
            json_obs = extract_json(raw_obs)
            if json_obs:
                observations = normalize_obs(json.loads(json_obs))

        time.sleep(1)

        # ===== DIAG =====
        prompt_diag = PROMPT_DIAG.replace("{DIAG_LIST}", DIAG_LIST_STR).replace("{text}", report)
        raw_diag = call_llm(prompt_diag)

        diagnoses = []
        if raw_diag:
            json_diag = extract_json(raw_diag)
            if json_diag:
                diagnoses = normalize_diag(json.loads(json_diag))

        print("✅ OBS:", observations)
        print("✅ DIAG:", diagnoses)

        results.append({
            "id": ecg_id,
            "observations": observations,
            "diagnoses": diagnoses
        })

# ============================================================
# SALVA RISULTATI
# ============================================================

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ COMPLETATO")
print("📂 File salvato in:", OUTPUT_FILE)
