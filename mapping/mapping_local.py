import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from pathlib import Path

# ============================================================
# LISTE OBS / DIAG (caricate da mapping/OBS_LIST.json e DIAG_LIST.json)
# ============================================================

_MAPPING_DIR = Path(__file__).resolve().parent

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
# CARICAMENTO MODELLO LOCALE
# ============================================================

# Scaricato offline su /leonardo_scratch via setup/download_dracarys.py
# (il nodo di calcolo non ha accesso a internet)
MAPPING_MODEL = "/leonardo_scratch/large/userexternal/vmoscato/models/dracarys-llama-3.1-70b-instruct"

print("🔄 Loading Dracarys‑Llama‑3.1‑70B‑Instruct for claim extraction...")
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

map_tokenizer = AutoTokenizer.from_pretrained(MAPPING_MODEL, use_fast=False)
map_model = AutoModelForCausalLM.from_pretrained(
    MAPPING_MODEL,
    quantization_config=_bnb_config,
    device_map="auto"
)
map_model.eval()

# ============================================================
# PROMPT PER ESTRARRE OSSERVAZIONI
# ============================================================

PROMPT_OBS_LOCAL = """
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

# ============================================================
# PROMPT PER ESTRARRE DIAGNOSI
# ============================================================

PROMPT_DIAG_LOCAL = """
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
# FUNZIONE DI GENERAZIONE
# ============================================================

def run_local_llm(prompt):
    inputs = map_tokenizer(prompt, return_tensors="pt").to(map_model.device)
    with torch.no_grad():
        output = map_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0
        )
    text = map_tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# ============================================================
# ESTRAZIONE JSON
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
