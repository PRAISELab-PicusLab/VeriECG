import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json
import re
from pathlib import Path

# modello: Qwen 2.5 3B Instruct

# ============================================================
# LISTE OBS / DIAG
# ============================================================

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

_MAPPING_DIR = BASE_DIR / "mapping"

with open(_MAPPING_DIR / "OBS_LIST.json", "r") as f:
    _OBS_LIST = json.load(f)

with open(_MAPPING_DIR / "DIAG_LIST.json", "r") as f:
    _DIAG_LIST = json.load(f)

OBS_LIST_STR = "\n".join(
    f"{o['id']}: {o['description']} ({o['feature']})" for o in _OBS_LIST
)
DIAG_LIST_STR = "\n".join(
    f"{d['diagnosis_id']}: {d['diagnosis_name']}" for d in _DIAG_LIST
)

# ============================================================
# CARICAMENTO MODELLO — Qwen 2.5 3B Instruct (NO quantizzazione)
# ============================================================

MAPPING_MODEL = "Qwen/Qwen2.5-3B-Instruct"

print("🔄 Loading Qwen 2.5 3B Instruct (FP16, no quantization)...")

map_tokenizer = AutoTokenizer.from_pretrained(
    MAPPING_MODEL,
    use_fast=True
)

map_model = AutoModelForCausalLM.from_pretrained(
    MAPPING_MODEL,
    device_map="cuda",
    torch_dtype=torch.float16
)

map_model.eval()

# ============================================================
# PROMPT PER OSSERVAZIONI
# ============================================================

PROMPT_OBS_LOCAL = """
You are an expert ECG information extraction system.
Return ONLY valid JSON. No explanations.

Extract ECG observations from the report by analyzing each lead separately.

Rules:
- Leads: I, II, III, aVR, aVL, aVF, V1–V6
- For EACH lead, extract ONLY observations explicitly mentioned in that lead
- Map each observation to EXACT OBS ID from the list
- Do NOT infer anything
- Do NOT include normal/absent findings unless explicitly described
- Do NOT generalize across leads
- If a lead is not mentioned, ignore it

Allowed OBS IDs:
{OBS_LIST}

Output format:
{{
  "observations": [
    "OBS_ID_LEAD_X",
    "OBS_ID_LEAD_Y"
  ]
}}

Report:
{text}
"""

# ============================================================
# PROMPT PER DIAGNOSI
# ============================================================

PROMPT_DIAG_LOCAL = """
You are an expert ECG diagnosis extraction system.
Return ONLY valid JSON. No explanations.

Extract ECG diagnoses explicitly stated in the report.

Rules:
- Include ONLY diagnoses explicitly confirmed
- Do NOT infer anything
- If a diagnosis is excluded, do NOT include it

Allowed DIAG IDs:
{DIAG_LIST}

Output format:
{{"diagnoses": ["DIAG_ID"]}}

Report:
{text}
"""

# ============================================================
# GENERAZIONE — QWEN INSTRUCT
# ============================================================

def run_local_llm(prompt):
    map_tokenizer.padding_side = "left"   

    inputs = map_tokenizer(prompt, return_tensors="pt").to(map_model.device)

    with torch.no_grad():
        output = map_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False   # deterministico per mapping
        )

    return map_tokenizer.decode(output[0], skip_special_tokens=True)

# ============================================================
# PARSING JSON
# ============================================================

def extract_json_local(text):
    candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
    for c in candidates:
        try:
            return json.loads(c)
        except:
            pass
    return {"observations": [], "diagnoses": []}

# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================

def extract_claims_from_report_local(report_text, OBS_LIST_STR, DIAG_LIST_STR):

    # ---- osservazioni ----
    prompt_obs = PROMPT_OBS_LOCAL.format(
        OBS_LIST=OBS_LIST_STR,
        text=report_text
    )
    out_obs = run_local_llm(prompt_obs)
    parsed_obs = extract_json_local(out_obs)
    oss_GEM = parsed_obs.get("observations", [])

    # ---- diagnosi ----
    prompt_diag = PROMPT_DIAG_LOCAL.format(
        DIAG_LIST=DIAG_LIST_STR,
        text=report_text
    )
    out_diag = run_local_llm(prompt_diag)
    parsed_diag = extract_json_local(out_diag)
    diag_GEM = parsed_diag.get("diagnoses", [])

    # normalizzazione
    oss_GEM = [o.strip().upper() for o in oss_GEM]
    diag_GEM = [d.strip().upper() for d in diag_GEM]

    return oss_GEM, diag_GEM
