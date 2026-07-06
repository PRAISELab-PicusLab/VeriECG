# MAPPING CON QWEN

#  - vocabolario dalle liste slim mapping/OBS_LIST.json e mapping/DIAG_LIST.json
#    (generate da mapping/genera_liste_da_rule_engine.py a partire dai dizionari
#    del rule engine). Se cambiano i dizionari del rule engine, vanno rigenerate le liste.
#  - convenzione ID identica alla ground truth: osservazioni = 'id';
#  - estrazione robusta _extract_ids: filtro sul vocabolario reale + recupero
#    degli ID sporcati da un suffisso + dedup, indipendente da un JSON ben formato.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from pathlib import Path

# ============================================================
# VOCABOLARIO OBS / DIAG  ==  liste SLIM allineate al rule engine
# ============================================================

_MAPPING_DIR = Path(__file__).resolve().parent / "mapping"

with open(_MAPPING_DIR / "OBS_LIST.json", "r", encoding="utf-8") as f:
    _OBS_LIST = json.load(f)
with open(_MAPPING_DIR / "DIAG_LIST.json", "r", encoding="utf-8") as f:
    _DIAG_LIST = json.load(f)

# osservazioni: id + descrizione + feature + LEAD (il lead aiuta Qwen a mappare
# il reperto giusto). diagnosi: diagnosis_id + nome leggibile.
OBS_LIST_STR = "\n".join(
    f"{o['id']}: {o['description']} ({o['feature']}) [lead: {o.get('lead', 'All')}]"
    for o in _OBS_LIST
)
DIAG_LIST_STR = "\n".join(
    f"{d['diagnosis_id']}: {d['diagnosis_name']}" for d in _DIAG_LIST
)

# vocabolari validi per il filtro post-generazione
VALID_OBS = {o["id"].strip().upper() for o in _OBS_LIST}
VALID_DIAG = {d["diagnosis_id"].strip().upper() for d in _DIAG_LIST}

print(f"[mapping] vocabolario slim (OBS_LIST/DIAG_LIST) -> "
      f"OBS: {len(VALID_OBS)} | DIAG: {len(VALID_DIAG)}")

# ============================================================
# CARICAMENTO MODELLO — Qwen 2.5 3B Instruct (FP16)
# ============================================================
MAPPING_MODEL = "Qwen/Qwen2.5-3B-Instruct"
print("Loading Qwen 2.5 3B Instruct (FP16)...")
map_tokenizer = AutoTokenizer.from_pretrained(MAPPING_MODEL, use_fast=True)
map_model = AutoModelForCausalLM.from_pretrained(
    MAPPING_MODEL,
    device_map="cuda",
    torch_dtype=torch.float16,
)
map_model.eval()
print("Qwen pronto.")

# ============================================================
# PROMPT ROBUSTI (da test_qwen_mapping)
# ============================================================
PROMPT_OBS_LOCAL = """You are an expert ECG information extraction system.
From the ECG report below, select the observations that are EXPLICITLY supported by the report.

STRICT RULES:
- Choose IDs ONLY from the "Allowed OBS IDs" list below.
- Copy each ID EXACTLY as it appears in the list (verbatim). Do NOT add lead
  suffixes such as "_lead_I" or "_V1_to_v2", do NOT change spelling, do NOT
  invent new IDs. The IDs already encode the lead when relevant.
- Include an ID only if the report explicitly supports that finding.
- Return ONLY valid JSON, no explanations, no extra text.

Allowed OBS IDs:
{OBS_LIST}

Output ONLY a JSON object of this form, using IDs copied verbatim from the list:
{{"observations": ["<exact_id_from_the_list>", "<exact_id_from_the_list>"]}}

Report:
{text}
"""

PROMPT_DIAG_LOCAL = """You are an expert ECG diagnosis extraction system.
Return ONLY valid JSON. No explanations.

Extract ECG diagnoses explicitly stated in the report.

Rules:
- Choose IDs ONLY from the "Allowed DIAG IDs" list below, copied EXACTLY (verbatim).
- Include ONLY diagnoses explicitly confirmed
- Do NOT infer anything
- If a diagnosis is excluded, do NOT include it

Allowed DIAG IDs:
{DIAG_LIST}

Output format:
{{"diagnoses": ["<exact_id_from_the_list>"]}}

Report:
{text}
"""

# ============================================================
# GENERAZIONE — QWEN INSTRUCT (chat template + decodifica solo token nuovi)
# ============================================================
def run_local_llm(prompt):
    map_tokenizer.padding_side = "left"
    # chat template: Qwen2.5-Instruct va guidato col suo template, altrimenti
    # col testo grezzo ricopia/continua il prompt invece di rispondere.
    messages = [{"role": "user", "content": prompt}]
    text = map_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = map_tokenizer(text, return_tensors="pt").to(map_model.device)
    with torch.no_grad():
        output = map_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,   # deterministico per il mapping
        )
    # decodifica SOLO i token nuovi (esclude il prompt), altrimenti il testo
    # conterrebbe anche l'esempio del prompt e verrebbe ripescato quello.
    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    return map_tokenizer.decode(gen_tokens, skip_special_tokens=True)

# ============================================================
# ESTRAZIONE ID ROBUSTA (da test_qwen_mapping)
# ============================================================
def _extract_ids(raw_text, valid_set):
    """Estrae gli ID: match esatto sul vocabolario reale, con RECUPERO degli ID
    'sporcati' da un suffisso (es. 'OBS_..._LEAD_I' -> 'OBS_...'). Deduplica in
    ordine ed e' robusto anche se il JSON e' troncato/malformato."""
    valid_sorted = sorted(valid_set, key=len, reverse=True)  # ID piu' lunghi prima
    seen, out = set(), []
    # NB: [^"]+ (non [A-Za-z0-9_]+) per catturare anche gli ID con caratteri
    # speciali: es. Q_WAVE_..., FIRST_DEGREE_..._WITH_NORMAL_QRS.
    for tok in re.findall(r'"([^"]+)"', raw_text):
        t = tok.strip().upper()
        match = None
        if t in valid_set:
            match = t
        else:
            for vid in valid_sorted:
                if t.startswith(vid + "_"):
                    match = vid
                    break
        if match and match not in seen:
            seen.add(match)
            out.append(match)
    return out

def extract_claims_from_report_local(report_text, OBS_LIST_STR, DIAG_LIST_STR, verbose=False):
    out_obs_raw = run_local_llm(PROMPT_OBS_LOCAL.format(OBS_LIST=OBS_LIST_STR, text=report_text))
    out_diag_raw = run_local_llm(PROMPT_DIAG_LOCAL.format(DIAG_LIST=DIAG_LIST_STR, text=report_text))
    oss_GEM = _extract_ids(out_obs_raw, VALID_OBS)
    diag_GEM = _extract_ids(out_diag_raw, VALID_DIAG)
    if verbose:
        print("=== RAW QWEN (osservazioni) ===\n", out_obs_raw, "\n")
        print("=== RAW QWEN (diagnosi) ===\n", out_diag_raw, "\n")
    return oss_GEM, diag_GEM
