# MAPPING CON LLM LOCALE  (default: meta-llama/Meta-Llama-3-8B-Instruct)
#
#  - vocabolario dalle liste slim mapping/OBS_LIST.json e mapping/DIAG_LIST.json
#    (generate da mapping/genera_liste_da_rule_engine.py a partire dai dizionari
#    del rule engine). Se cambiano i dizionari del rule engine, vanno rigenerate le liste.
#  - convenzione ID identica alla ground truth: osservazioni = 'id'; diagnosi = 'diagnosis_id'.
#  - PROMPT v2: regola di NEGAZIONE esplicita + reperti NORMALI (sblocca le
#    osservazioni sulla P) + few-shot + breve reasoning prima del JSON. Riempiti
#    con .replace (NON .format): le graffe del JSON negli esempi restano singole.
#  - RETE DI SICUREZZA regex anti-negazione (negation_filter): scarta le OBS
#    anormali (ST depression, low voltage, Q patologiche, ...) se nel referto la
#    keyword compare solo in forma NEGATA.
#  - estrazione robusta _extract_ids: filtro sul vocabolario reale + recupero
#    degli ID sporcati da un suffisso + dedup, indipendente da un JSON ben formato.
#
# NB modello: qualunque LLM instruct compatibile con il chat template va bene.
# Il default e' Llama-3-8B; per usarne un altro cambia MAPPING_MODEL sotto.
# Vincolo se usato NELLO STESSO kernel di GEM (transformers==4.38.1): usare modelli
# che caricano su 4.38.1 (es. Llama-3). Gemma-2/3 richiedono transformers>=4.42
# e vanno usati solo in un processo DISACCOPPIATO.

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from pathlib import Path

# ============================================================
# VOCABOLARIO OBS / DIAG  ==  liste SLIM allineate al rule engine
# ============================================================
# I JSON stanno nella STESSA cartella di questo file (mapping/).
_MAPPING_DIR = Path(__file__).resolve().parent

with open(_MAPPING_DIR / "OBS_LIST.json", "r", encoding="utf-8") as f:
    _OBS_LIST = json.load(f)
with open(_MAPPING_DIR / "DIAG_LIST.json", "r", encoding="utf-8") as f:
    _DIAG_LIST = json.load(f)

# osservazioni: id + descrizione + feature + LEAD (il lead aiuta a mappare
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
# CARICAMENTO MODELLO — TOGGLE
# ============================================================
# >>> SCEGLI QUI il modello del mapping <<<
MAPPING_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"   # repo Meta ufficiale (GATED -> accesso approvato + HF_TOKEN)
# MAPPING_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"  # mirror pubblico Llama-3 (NON gated)
# (va bene qualunque altro LLM instruct che carichi su transformers 4.38.1 e usi il chat template)

_ml = MAPPING_MODEL.lower()
_IS_LLAMA3 = "llama-3" in _ml

# I modelli Llama-3 / Gemma sono GATED su Hugging Face: accetta la licenza sul sito
# e fornisci un token. In Colab: Secrets -> HF_TOKEN; altrimenti esporta HF_TOKEN.
try:
    from google.colab import userdata
    _HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    _HF_TOKEN = os.environ.get("HF_TOKEN")

# ------------------------------------------------------------
# PRECISIONE DI CARICAMENTO
# NB: nel kernel di GEM (transformers==4.38.1) la quantizzazione 4-bit "from scratch"
# della bitsandbytes recente e' ROTTA -> errore:
#   "Blockwise 4bit quantization only supports 16/32-bit floats, but got torch.uint8".
# Su A100-80GB il mapper NON ha bisogno del 4-bit: si carica in fp16 e si salta del
# tutto quel codepath. Per usare comunque il 4-bit (serve solo per modelli >~20B):
#   metti USE_4BIT=True, AGGIUNGI 'bitsandbytes==0.43.1' alle dipendenze
#   (accanto a transformers==4.38.1) e RIAVVIA il runtime.
# ------------------------------------------------------------
USE_4BIT = False   # fp16 di default: evita il bug bitsandbytes

_tok_kwargs = {"use_fast": True}
if _HF_TOKEN:
    _tok_kwargs["token"] = _HF_TOKEN
map_tokenizer = AutoTokenizer.from_pretrained(MAPPING_MODEL, **_tok_kwargs)

_load_kwargs = dict(device_map="cuda", torch_dtype=torch.float16)
if _HF_TOKEN:
    _load_kwargs["token"] = _HF_TOKEN
if USE_4BIT:
    _load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    print(f"[mapping] carico {MAPPING_MODEL} (4-bit)...")
else:
    print(f"[mapping] carico {MAPPING_MODEL} (fp16)...")

map_model = AutoModelForCausalLM.from_pretrained(MAPPING_MODEL, **_load_kwargs)
map_model.eval()

# Llama-3-Instruct termina i turni con <|eot_id|>: aggiungilo agli stop token,
# altrimenti la generazione greedy prosegue fino a max_new_tokens.
_eot = map_tokenizer.convert_tokens_to_ids("<|eot_id|>")
_TERMINATORS = [map_tokenizer.eos_token_id]
if _eot is not None and _eot != map_tokenizer.unk_token_id:
    _TERMINATORS.append(_eot)
if map_tokenizer.pad_token_id is None:
    map_tokenizer.pad_token = map_tokenizer.eos_token
print(f"[mapping] {MAPPING_MODEL} pronto ({'4-bit' if USE_4BIT else 'fp16'}).")

# ============================================================
# PROMPT v2 — negazione + reperti normali + few-shot + reasoning
#   (riempiti con .replace: le graffe del JSON negli esempi restano singole)
# ============================================================
PROMPT_OBS_LOCAL = """You are an expert ECG information extraction system.
From the ECG REPORT, output the observation IDs that the report ASSERTS AS PRESENT.

# NEGATION IS CRITICAL
A finding that the report says is ABSENT, NORMAL, ruled out, or unlikely must NOT be
extracted as an ABNORMAL finding. Treat these as NEGATED and DO NOT extract the abnormal ID:
"no", "not", "without", "does not show", "not indicative of", "rules out",
"reduces the likelihood/suspicion", "unlikely", "absence of", "free of",
"within normal limits", "unremarkable".
Example: "QRS within normal limits, no ST depression" -> DO NOT output
OBS_QRS_LOW_VOLTAGE_LIMB / OBS_QRS_LOW_VOLTAGE_PRECORDIAL / OBS_ST_SEGMENT_DEPRESSION.

# NORMAL FINDINGS COUNT
When the report AFFIRMS a normal finding, output the matching NORMAL observation ID.
Example: "normal P wave amplitude and duration; PR interval normal; each P precedes a QRS"
-> OBS_P_AMPLITUDE_NORMAL, OBS_P_DURATION_NORMAL, OBS_P_PRECEDES_QRS, OBS_PR_INTERVAL_NORMAL.

# RULES
- Choose IDs ONLY from "Allowed OBS IDs", copied verbatim. Do not invent IDs or add lead suffixes.
- Extract a finding ONLY if affirmed as present (see NEGATION).

Allowed OBS IDs:
{OBS_LIST}

# WORKED EXAMPLES
Report: "Heart rate 72 bpm. Normal P wave amplitude and duration. PR interval normal.
QRS within normal limits. No ST depression, no pathological Q waves, T waves upright."
Reasoning: HR normal; P amplitude/duration normal; PR normal; QRS normal. ST depression
and Q waves are NEGATED -> exclude them.
{"observations": ["OBS_HEART_RATE_NORMAL_RANGE","OBS_P_AMPLITUDE_NORMAL","OBS_P_DURATION_NORMAL","OBS_PR_INTERVAL_NORMAL","OBS_QRS_DURATION_NORMAL"]}

Report: "Deep S wave in V1-V2, ST depression in lateral leads, left-sided T wave inversion."
Reasoning: all three are affirmed as present.
{"observations": ["OBS_S_WAVE_DEEP_V1_V2","OBS_ST_DEPRESSION_LEFT_SIDED","OBS_T_WAVE_INVERSION_LEFT_SIDED"]}

# NOW ANALYSE THIS REPORT
Report:
{text}

Give ONE short line of reasoning, then on the LAST line output ONLY the JSON object:
{"observations": [...]}
"""

PROMPT_DIAG_LOCAL = """You are an expert ECG diagnosis extraction system.
Output the diagnosis IDs the report CONFIRMS as present.

# NEGATION
Do NOT output a diagnosis that is ruled out / excluded / said absent or unlikely
("no evidence of", "rules out", "reduces the suspicion of", "unlikely", "without").

# RULES
- Choose IDs ONLY from "Allowed DIAG IDs", verbatim. Do not infer beyond what is stated.

Allowed DIAG IDs:
{DIAG_LIST}

# EXAMPLES
Report: "Sinus tachycardia. No evidence of infarction or bundle branch block."
{"diagnoses": ["SINUS_TACHYCARDIA"]}
Report: "Normal sinus rhythm, ECG within normal limits."
{"diagnoses": ["SINUS_RHYTHM","NORM"]}

Report:
{text}

Output ONLY the JSON object on the last line:
{"diagnoses": [...]}
"""

# ============================================================
# RETE DI SICUREZZA anti-negazione (deterministica)
# ============================================================
_NEG = (r"(?:no|not|without|absence of|absent|denies|negative for|free of|"
        r"rules? out|ruled out|reduces?(?: the)?(?: likelihood| suspicion)?|"
        r"unlikely|within normal limits|unremarkable|normal|non-)")

_ABNORMAL_KW = {
    "OBS_QRS_LOW_VOLTAGE_LIMB":       ["low voltage", "low amplitude", "reduced amplitude"],
    "OBS_QRS_LOW_VOLTAGE_PRECORDIAL": ["low voltage", "low amplitude", "reduced amplitude"],
    "OBS_ST_SEGMENT_DEPRESSION":      ["st depression", "st segment depression", "depressed st", "st-segment depression"],
    "OBS_ST_DEPRESSION_LEFT_SIDED":   ["st depression", "st segment depression"],
    "OBS_ST_DEPRESSION_V1_V2":        ["st depression"],
    "OBS_PATHOLOGICAL_Q_WAVE_DEPTH":  ["q wave", "pathological q", "abnormal q", "significant q"],
    "OBS_PATHOLOGICAL_Q_WAVE_V2_V3":  ["q wave"],
    "OBS_PATHOLOGICAL_Q_WAVE_OTHER_LEADS": ["q wave"],
    "OBS_T_WAVE_INVERSION":           ["t wave inversion", "inverted t", "t-wave inversion"],
    "OBS_T_WAVE_INVERSION_LEFT_SIDED": ["t wave inversion", "inverted t"],
    "OBS_T_WAVE_INVERSION_V1_V2":     ["t wave inversion", "inverted t"],
    "OBS_ST_SEGMENT_ELEVATION":       ["st elevation", "elevated st", "st segment elevation"],
    "OBS_QRS_DURATION_WIDE":          ["wide qrs", "widened qrs", "prolonged qrs", "qrs prolong", "broad qrs"],
    "OBS_R_PRIME_PRESENT_V1_V2":      ["rsr", "r prime", "r'"],
    "OBS_R_PRIME_GT_R_V1_V2":         ["rsr", "r prime", "r'"],
}

def _affirmed(report_low, kws):
    for kw in kws:
        start = 0
        while True:
            i = report_low.find(kw, start)
            if i == -1:
                break
            left = report_low[max(0, i - 55):i]
            if not re.search(_NEG + r"[^.;:]*$", left):
                return True
            start = i + len(kw)
    return False

def negation_filter(ids, report, verbose=False):
    rl = report.lower()
    kept, dropped = [], []
    for _id in ids:
        kws = _ABNORMAL_KW.get(_id)
        if kws is None or _affirmed(rl, kws):
            kept.append(_id)
        else:
            dropped.append(_id)
    if verbose and dropped:
        print("  [negation-filter] scartate:", dropped)
    return kept, dropped

# ============================================================
# GENERAZIONE — chat template (generico per LLM instruct)
# ============================================================
def run_local_llm(prompt, max_new_tokens=400):
    map_tokenizer.padding_side = "left"
    # Usiamo solo il ruolo 'user' -> compatibile con i modelli instruct via chat template.
    messages = [{"role": "user", "content": prompt}]
    text = map_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = map_tokenizer(text, return_tensors="pt").to(map_model.device)
    with torch.no_grad():
        output = map_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # deterministico per il mapping
            repetition_penalty=1.1,
            eos_token_id=_TERMINATORS,
            pad_token_id=map_tokenizer.pad_token_id,
        )
    # decodifica SOLO i token nuovi (esclude il prompt), altrimenti il testo
    # conterrebbe anche l'esempio del prompt e verrebbe ripescato quello.
    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    return map_tokenizer.decode(gen_tokens, skip_special_tokens=True)

# ============================================================
# ESTRAZIONE ID ROBUSTA
# ============================================================
def _extract_ids(raw_text, valid_set):
    """Estrae gli ID: match esatto sul vocabolario reale, con RECUPERO degli ID
    'sporcati' da un suffisso (es. 'OBS_..._LEAD_I' -> 'OBS_...'). Deduplica in
    ordine ed e' robusto anche se il JSON e' troncato/malformato; il reasoning in
    linguaggio naturale prima del JSON non disturba (si tiene solo cio' che e' tra
    virgolette e combacia col vocabolario)."""
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
    # NB: .replace (NON .format) -> le graffe del JSON negli esempi restano singole.
    out_obs_raw = run_local_llm(
        PROMPT_OBS_LOCAL.replace("{OBS_LIST}", OBS_LIST_STR).replace("{text}", report_text)
    )
    out_diag_raw = run_local_llm(
        PROMPT_DIAG_LOCAL.replace("{DIAG_LIST}", DIAG_LIST_STR).replace("{text}", report_text)
    )
    oss_GEM = _extract_ids(out_obs_raw, VALID_OBS)
    diag_GEM = _extract_ids(out_diag_raw, VALID_DIAG)
    # rete di sicurezza deterministica: scarta le OBS anormali solo-negate.
    oss_GEM, _ = negation_filter(oss_GEM, report_text, verbose=verbose)
    if verbose:
        print("=== RAW LLM (osservazioni) ===\n", out_obs_raw, "\n")
        print("=== RAW LLM (diagnosi) ===\n", out_diag_raw, "\n")
    return oss_GEM, diag_GEM
