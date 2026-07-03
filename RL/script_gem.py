#!/usr/bin/env python
# coding: utf-8

# **REWARD JHA25**: definizione reward

# In[ ]:


def compute_claim_metrics(oss_RE, diag_RE, oss_GEM, diag_GEM):
    """
    Calcola precision, recall e F1 sui claim clinici,
    seguendo la definizione di Jha25 (DocLens claim-level metrics).
    """

    # Reference claims (rule engine)
    R = set(oss_RE + diag_RE)

    # Output claims (GEM)
    O = set(oss_GEM + diag_GEM)

    # ---- Recall (Completezza) ----
    # proporzione di claim veri coperti dal modello
    if len(R) == 0:
        recall = 1.0
    else:
        tp_recall = sum(1 for r in R if r in O)
        recall = tp_recall / len(R)

    # ---- Precision (Factualità) ----
    # proporzione di claim generati supportati dal rule engine
    if len(O) == 0:
        precision = 1.0
    else:
        tp_prec = sum(1 for o in O if o in R)
        precision = tp_prec / len(O)

    # ---- F1 ----
    eps = 1e-8
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def compute_reward_jha25(
    oss_RE, diag_RE, oss_GEM, diag_GEM,
    scale=10.0,
    gating_threshold=0.6
):
    """
    Reward in stile Jha25:
    - precision e recall sui claim
    - F1 claim-based
    - scaling a [0, 10]
    - reward gating (F1 < 0.6 → reward = 0)
    """

    precision, recall, f1 = compute_claim_metrics(
        oss_RE, diag_RE, oss_GEM, diag_GEM
    )

    # ---- Reward gating ----
    if f1 < gating_threshold:
        return 0.0

    # ---- Reward scalato ----
    return scale * f1


# In[ ]:


# Monta Drive se serve (in Colab)
from google.colab import drive
drive.mount('/content/drive')

# In[ ]:


# Installazione UNICA e coerente delle dipendenze per il training RL.
# NIENTE piu' disinstallazioni a meta' notebook: le versioni sono fissate
# una volta sola, qui, e non vengono piu' toccate.
#
# NB torch: NON si pinna piu' a torch==2.1.2. Quel pin risale a quando GEM
# e' stato sviluppato, ma i runtime Colab attuali usano Python 3.12 e per
# quella versione di Python non esiste alcuna wheel di torch 2.1.2 (la piu'
# vecchia disponibile e' la 2.2.0): "pip install torch==2.1.2" fallisce
# sempre con "No matching distribution found", lasciando silenziosamente
# installato qualunque torch fosse gia' presente nell'immagine Colab. Meglio
# non pinnarlo affatto e usare cio' che Colab fornisce di serie (di solito
# gia' con CUDA funzionante), stampandone la versione reale per verifica.
!pip install -q wfdb ftfy bitsandbytes
!pip install -q transformers==4.38.1 accelerate==0.21.0 peft==0.7.1

import torch
print(
    "Dipendenze RL installate (transformers 4.38.1 / accelerate 0.21.0 / "
    f"peft 0.7.1 / torch {torch.__version__}, CUDA disponibile: {torch.cuda.is_available()})"
)


# In[ ]:


# MAPPPING CON QWEN
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM   # MODIFICATO 27/06
import json
import re
from pathlib import Path

# ============================================================
# LISTE OBS / DIAG
# ============================================================

BASE = Path(__file__).resolve().parents[1]

_MAPPING_DIR = BASE / "mapping"

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

# MODIFICATO 27/06 — modello Instruct, perfetto per mapping
MAPPING_MODEL = "Qwen/Qwen2.5-3B-Instruct"

print("🔄 Loading Qwen 2.5 3B Instruct (FP16, no quantization)...  # MODIFICATO 27/06")

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
    map_tokenizer.padding_side = "left"   # MODIFICATO 27/06

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


# **con PPO**

# In[ ]:


"""
train_rl_gem PPO -- versione corretta (fix multimodale + fix reward/value).

Cosa e' cambiato rispetto alla versione precedente:
- path unificati sotto BASE (niente piu' "Tiro" vs "TIRO" vs
  "Tiro/models/GEM_model")
- rimossi gli import fragili `from reward_engine import ...` e
  `from mapping_local import ...`: le funzioni che servono
  (compute_reward_jha25, compute_claim_metrics, extract_claims_from_report_local,
  OBS_LIST_STR, DIAG_LIST_STR) sono gia' definite piu' sopra nello stesso
  notebook (celle "REWARD JHA25" e "MAPPING SEMANTICO"). Prima venivano
  importate da file .py esterni che il notebook non scriveva mai e che
  potevano essere disallineati rispetto al codice mostrato qui.
- l'assert su grid_pinpoints ora stampa informazioni di debug utili invece
  di limitarsi a fallire silenziosamente.
- train_rl_multimodal calcola ORA anche una valutazione "prima" (baseline,
  pesi originali) e "dopo" (post-RL, dopo il training PPO) sullo stesso
  modello caricato in memoria, cosi' si puo' confrontare l'impatto del RL.

FIX (revisione RL):
1. `collate_fn` tokenizzava i prompt con il tokenizer "piatto"
   (`tokenizer(prompts, ...)`), che NON converte il segnaposto testuale
   "<image>" nel sentinella speciale IMAGE_TOKEN_INDEX (-200) atteso da
   `prepare_inputs_labels_for_multimodal` (GEM/llava/model/llava_arch.py).
   Senza quel sentinella, `num_images` risultava sempre 0 e la fusione
   multimodale veniva SILENZIOSAMENTE saltata: il modello generava (e
   veniva aggiornato) senza mai vedere l'ECG o l'immagine, rendendo la
   reward quasi priva di senso. Ora si usa `tokenizer_image_token`, la
   stessa funzione usata dallo script ufficiale di inferenza di GEM
   (llava/eval/model_ecg_resume.py), che inserisce correttamente il
   sentinella.
2. Una volta che la fusione multimodale avviene davvero, il segnaposto
   "<image>" viene sostituito da centinaia di embedding (ecg + immagine):
   la sequenza che il modello elabora internamente e' quindi molto piu'
   lunga della sequenza di token originale, e NON e' piu' allineata 1:1
   con `input_ids`. Calcolare la posizione della risposta contando i token
   del solo prompt (`Lp`) e affettare logits/hidden_states con quell'indice
   (come faceva la versione precedente) puntava quindi a posizioni
   sbagliate. Ora la posizione dei token di risposta nella sequenza
   espansa si ricava passando esplicitamente `labels` (IGNORE_INDEX sul
   prompt, id reale sulla risposta) a `prepare_inputs_labels_for_multimodal`
   e leggendo dove, nei `new_labels` restituiti, il valore e' diverso da
   IGNORE_INDEX: questi indici sono per costruzione quelli giusti nella
   sequenza fusa, qualunque sia il numero di feature ecg/immagine inserite.
3. I token di risposta vengono presi direttamente dall'output di
   `model.generate(...)` (che restituisce SOLO i nuovi token quando si
   genera da `inputs_embeds`, come fa GEM quando `images is not None`)
   invece di essere ottenuti ri-tokenizzando il testo decodificato
   concatenato al prompt: si evita cosi' un secondo, piu' piccolo,
   disallineamento dovuto a merge diversi del tokenizer BPE sul testo
   unito rispetto al prompt e alla risposta tokenizzati separatamente.
4. `advantages`/`returns` (GAE) venivano calcolati a partire dagli stessi
   `values` con gradiente prodotti dalla forward pass di training, e MAI
   staccati dal grafo: il target di regressione del value loss dipendeva
   quindi dalle stesse predizioni che si voleva regredire (target mobile
   auto-referenziale), e il policy loss propagava gradiente anche dentro
   il value head tramite gli advantages. Ora GAE viene calcolato sotto
   `torch.no_grad()` su valori esplicitamente `.detach()`-ati: advantages
   e returns sono trattati come costanti, come nel PPO standard.
"""

import os
import sys
import json
import copy
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from PIL import Image
import wfdb

# FIX: sys.path deve includere GEM/ PRIMA di "from llava import ...",
# altrimenti l'import fallisce con ModuleNotFoundError ogni volta che questa
# cella gira in un runtime Colab fresco in cui la cella "Installa il
# pacchetto GEM" (pip install -e .) non e' ancora stata rieseguita in questa
# sessione. pip install -e in una sessione precedente non basta: ogni nuovo
# runtime Colab riparte da zero, quindi il fallback via sys.path deve essere
# in vigore PRIMA di provare a importare llava, non dopo.
os.environ["ACCELERATE_DISABLE_MAPPING"] = "1"
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

BASE = Path("/content/drive/MyDrive/TIRO")
sys.path.append(str(BASE / "GEM"))
sys.path.append(str(BASE / "mapping"))
sys.path.append(str(BASE))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

# --- PATCH cache_position (compat transformers >= 4.38) ---
# transformers 4.38 passa 'cache_position' a forward() durante generate(), ma
# la forward di GEM (LlavaLlamaForCausalLM, basata su una LLaVA piu' vecchia)
# non lo accetta -> TypeError 'unexpected keyword argument cache_position'.
# Lo scartiamo prima di chiamare la forward originale (LlamaModel ricalcola
# cache_position internamente quando e' None, quindi e' sicuro ignorarlo).
# IMPORTANTE: si usa functools.wraps cosi' inspect.signature() segue __wrapped__
# e vede la firma ORIGINALE (con attention_mask, ecgs, images, position_ids...):
# altrimenti _validate_model_kwargs di generate() segnalerebbe 'attention_mask
# not used'. E attention_mask SERVE: la 4.38 ci ricava position_ids dentro
# prepare_inputs_for_generation (senza, position_ids resta None -> crash).
import functools as _functools
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as _LlavaLlamaForCausalLM
# salva la forward ORIGINALE una sola volta; ri-eseguendo la cella si riparte
# sempre da quella (niente wrapper impilati).
if not hasattr(_LlavaLlamaForCausalLM, '_orig_forward_unpatched'):
    _LlavaLlamaForCausalLM._orig_forward_unpatched = _LlavaLlamaForCausalLM.forward
_orig_llava_forward = _LlavaLlamaForCausalLM._orig_forward_unpatched
@_functools.wraps(_orig_llava_forward)
def _forward_no_cache_position(self, *args, cache_position=None, **kwargs):
    return _orig_llava_forward(self, *args, **kwargs)
_LlavaLlamaForCausalLM.forward = _forward_no_cache_position
print('[patch] forward: cache_position ignorato, firma originale preservata (compat 4.38)')

# Controllo esplicito: queste funzioni devono essere gia' presenti nel
# kernel (definite nelle celle precedenti). Se manca qualcosa, meglio un
# errore chiaro subito che un NameError criptico a meta' training.
_REQUIRED_NAMES = [
    "compute_reward_jha25",
    "compute_claim_metrics",
    "extract_claims_from_report_local",
    "OBS_LIST_STR",
    "DIAG_LIST_STR",
]
_missing = [n for n in _REQUIRED_NAMES if n not in globals()]
if _missing:
    raise RuntimeError(
        "Mancano queste definizioni nel notebook: "
        + ", ".join(_missing)
        + ". Esegui prima le celle 'REWARD JHA25' (compute_reward_jha25, "
        "compute_claim_metrics) e 'MAPPING SEMANTICO' / 27/06 "
        "(extract_claims_from_report_local, OBS_LIST_STR, DIAG_LIST_STR)."
    )


# ============================================================
# DATASET RL MULTIMODALE
# ============================================================
class RLMultiModalDataset(Dataset):
    def __init__(self, rl_json_path, ptbxl_json_path, image_root, ecg_root,
                 image_processor, model_config, ecg_seq_len=5000):
        super().__init__()

        with open(rl_json_path, "r") as f:
            self.rl_data = json.load(f)

        with open(ptbxl_json_path, "r") as f:
            ptbxl = json.load(f)

        self.index = {x["id"]: x for x in ptbxl}
        self.image_root = Path(image_root)
        self.ecg_root = Path(ecg_root)
        self.image_processor = image_processor
        # model_config serve a process_images per rispettare image_aspect_ratio
        # ("ori"/"pad"/"anyres"): questo modello usa "ori".
        self.model_config = model_config
        self.ecg_seq_len = ecg_seq_len

    def __len__(self):
        return len(self.rl_data)

    def load_ecg(self, rel_path):
        record_path = self.ecg_root / rel_path
        ecg = wfdb.rdsamp(str(record_path))[0]
        ecg[np.isnan(ecg)] = 0
        ecg[np.isinf(ecg)] = 0
        ecg = torch.tensor(ecg.T, dtype=torch.float32)

        c, L = ecg.shape
        if L < self.ecg_seq_len:
            new = torch.zeros((c, self.ecg_seq_len))
            new[:, :L] = ecg
            ecg = new
        else:
            ecg = ecg[:, :self.ecg_seq_len]
        return ecg

    def load_image(self, rel_path):
        img_path = self.image_root / rel_path
        img = Image.open(img_path).convert("RGB")
        # FIX: si usa la STESSA pipeline dell'inferenza ufficiale GEM
        # (model_ecg_resume.py: process_images([img], image_processor, model.config)),
        # che sceglie il preprocessing in base a model_config.image_aspect_ratio.
        # La versione precedente chiamava sempre process_anyres_image + grid_pinpoints,
        # valido SOLO in modalita' 'anyres'. Questo modello ha image_aspect_ratio='ori'
        # -> process_images fa un normale preprocess CLIP e restituisce (3, H, W).
        image = process_images([img], self.image_processor, self.model_config)[0]
        return image

    def __getitem__(self, idx):
        item = self.rl_data[idx]
        ecg_id = item["id"]

        entry = self.index.get(ecg_id)
        if entry is None:
            raise KeyError(
                f"id '{ecg_id}' presente nel rl_dataset ma assente in "
                f"ptbxl_subset.json: controlla che i due file siano coerenti."
            )

        ecg = self.load_ecg(entry["ecg"])
        image = self.load_image(entry["image"])

        return {
            "id": ecg_id,
            "prompt": item["prompt"],
            "rule_obs": item["rule_observations"],
            "rule_diag": item["rule_diagnoses"],
            "ecg": ecg,
            "image": image,
        }


def collate_fn(batch, tokenizer):
    """
    FIX: usa tokenizer_image_token (non il tokenizer "piatto") per costruire
    input_ids. Il prompt contiene il segnaposto testuale "<image>", che deve
    diventare il sentinella IMAGE_TOKEN_INDEX (-200) in input_ids: e' l'unico
    modo con cui prepare_inputs_labels_for_multimodal riconosce dove inserire
    le feature ecg/immagine. tokenizer_image_token non supporta il batching
    nativo, quindi si tokenizza ogni prompt singolarmente e si fa padding
    manuale a destra (il config del modello ha tokenizer_padding_side="right").
    """
    prompts = [b["prompt"] for b in batch]

    ids_list = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for p in prompts
    ]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(x.shape[0] for x in ids_list)

    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, :ids.shape[0]] = ids
        attention_mask[i, :ids.shape[0]] = 1

    # FIX DTYPE: entrambe le tower di GEM ricastano il proprio output al dtype
    # dell'INPUT (clip_encoder.py: ecg_features.to(ecgs.dtype) e
    # image_features.to(images.dtype)). Se ecg/img sono fp32, l'output torna
    # fp32 e a valle il mm_projector (pesi fp16) da' 'mat1 and mat2 must have
    # the same dtype (Float vs Half)'. L'inferenza ufficiale passa infatti
    # ecg/img in .half(): li portiamo a fp16 qui, cosi' tutta la catena resta fp16.
    ecgs = torch.stack([b["ecg"] for b in batch]).to(torch.float16)
    images = torch.stack([b["image"] for b in batch]).to(torch.float16)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ecgs": ecgs,
        "images": images,
        "rule_obs": [b["rule_obs"] for b in batch],
        "rule_diag": [b["rule_diag"] for b in batch],
        "ids": [b["id"] for b in batch],
        "prompts": prompts,
    }


# ============================================================
# GENERAZIONE CON GEM (solo testo, senza grad)
# ============================================================
def generate_gem(model, tokenizer, input_ids, attention_mask, ecgs, images, max_new_tokens=512):
    model.eval()
    with torch.no_grad():
        # FIX: GEM (LlavaLlamaForCausalLM.generate) chiama il primo parametro
        # 'inputs', NON 'input_ids'. Passando input_ids=... come keyword finiva
        # in **kwargs e 'inputs' restava None -> prepare_inputs_labels_for_multimodal
        # riceveva input_ids=None -> 'NoneType' object has no attribute 'shape'.
        # Come nello script ufficiale (model_ecg_resume.py) i token vanno passati
        # come primo argomento POSIZIONALE. 'temperature' rimosso: con
        # do_sample=False e' ignorato (e in alcune versioni di transformers
        # temperature=0.0 solleva 'temperature must be strictly positive').
        # attention_mask SERVE con transformers 4.38: prepare_inputs_for_generation
        # ci ricava position_ids (senza, position_ids resta None -> crash su
        # cache_position). Ora e' di nuovo accettato perche' il patch della forward
        # preserva la firma (functools.wraps sopra).
        # inputs= (keyword) e NON posizionale: la generate di GEM chiama il
        # parametro 'inputs', e durante il training il modello e' un PeftModel,
        # la cui generate() NON accetta argomenti posizionali ('takes 1 positional
        # argument but 2 were given'). Come keyword funziona sia col modello nudo
        # (baseline) sia col PeftModel (training).
        out = model.generate(
            inputs=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            ecgs=ecgs.to(model.device),
            images=images.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    # NB: quando si genera da input costruiti con `images is not None`, GEM
    # (LlavaLlamaForCausalLM.generate) passa a HF generate() `inputs_embeds`
    # invece di `input_ids`: in quel caso l'output di generate() contiene
    # SOLO i token generati (HF non puo' anteporre l'input, perche' non
    # esistono input_ids corrispondenti a inputs_embeds). `out` sono quindi
    # esattamente i token di risposta: li riusiamo cosi' come sono nello
    # step di update PPO, invece di doverli recuperare ri-tokenizzando il
    # testo decodificato.
    reports = tokenizer.batch_decode(out, skip_special_tokens=True)
    return reports, out


# ============================================================
# GAE
# ============================================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    returns = []
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
        returns.insert(0, gae + values[t])

    return torch.stack(advantages), torch.stack(returns)


# ============================================================
# VALUTAZIONE (usata sia PRIMA che DOPO il training, stesso codice)
# ============================================================
@torch.no_grad()
def evaluate_policy(model, tokenizer, loader, label="eval"):
    model.eval()
    precisions, recalls, f1s, rewards = [], [], [], []

    import time as _time
    _nb = len(loader)
    for _bi, batch in enumerate(loader):
        _t0 = _time.time()
        print(f"[{label}] campione {_bi+1}/{_nb}: genero report con GEM "
              f"(max_new_tokens=512)...", flush=True)
        reports, _ = generate_gem(
            model, tokenizer,
            batch["input_ids"], batch["attention_mask"],
            batch["ecgs"], batch["images"],
        )
        print(f"[{label}] campione {_bi+1}: GEM ok in {_time.time()-_t0:.1f}s "
              f"(report: {len(reports[0]) if reports else 0} caratteri). "
              f"Estraggo claim con Qwen...", flush=True)
        for rep, oss_RE, diag_RE in zip(reports, batch["rule_obs"], batch["rule_diag"]):
            oss_GEM, diag_GEM = extract_claims_from_report_local(rep, OBS_LIST_STR, DIAG_LIST_STR)
            p, r, f1 = compute_claim_metrics(oss_RE, diag_RE, oss_GEM, diag_GEM)
            reward = compute_reward_jha25(oss_RE, diag_RE, oss_GEM, diag_GEM)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            rewards.append(reward)
        print(f"[{label}] campione {_bi+1}/{_nb} completato in "
              f"{_time.time()-_t0:.1f}s (f1={f1:.3f}, reward={reward:.2f})", flush=True)

    result = {
        "label": label,
        "n": len(rewards),
        "precision_mean": float(np.mean(precisions)) if precisions else float("nan"),
        "recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
        "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
    }
    print(
        f"[{label}] n={result['n']}  "
        f"precision={result['precision_mean']:.3f}  "
        f"recall={result['recall_mean']:.3f}  "
        f"f1={result['f1_mean']:.3f}  "
        f"reward={result['reward_mean']:.3f}"
    )
    return result


def _build_full_ids_and_labels(prompt_input_ids, prompt_attention_mask, response_ids, pad_id):
    """
    Concatena, per ogni esempio del batch, i token del prompt (gia' con il
    sentinella IMAGE_TOKEN_INDEX dentro, prodotti da collate_fn) con i token
    di risposta generati da generate_gem, SENZA ri-tokenizzare nulla.
    Costruisce anche `labels`: IGNORE_INDEX sul prompt, id reale sulla
    risposta. Dopo la fusione multimodale, le posizioni con label diverso da
    IGNORE_INDEX indicano esattamente dove si trovano i token di risposta
    nella sequenza espansa (vedi nota FIX 2 nel docstring del modulo).
    """
    full_ids_list, labels_list = [], []

    for i in range(prompt_input_ids.shape[0]):
        mask_i = prompt_attention_mask[i].bool()
        prompt_ids_i = prompt_input_ids[i][mask_i]

        resp_ids_i = response_ids[i]
        resp_mask_i = resp_ids_i != pad_id
        if resp_mask_i.any():
            last_real = resp_mask_i.nonzero()[-1].item()
            resp_ids_i = resp_ids_i[: last_real + 1]
        else:
            resp_ids_i = resp_ids_i[:0]

        full_ids_list.append(torch.cat([prompt_ids_i, resp_ids_i]))
        labels_list.append(torch.cat([
            torch.full_like(prompt_ids_i, IGNORE_INDEX),
            resp_ids_i.clone(),
        ]))

    max_len = max(x.shape[0] for x in full_ids_list)
    B = len(full_ids_list)
    device = prompt_input_ids.device

    full_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    full_attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
    full_labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
    for i in range(B):
        L = full_ids_list[i].shape[0]
        full_ids[i, :L] = full_ids_list[i].to(device)
        full_attn[i, :L] = 1
        full_labels[i, :L] = labels_list[i].to(device)

    return full_ids, full_attn, full_labels


# ============================================================
# TRAINING RL MULTIMODALE CON PPO (+ valutazione prima/dopo)
# ============================================================
def train_rl_multimodal(
    model_path,
    train_rl_json_path,
    eval_rl_json_path,
    ptbxl_path,
    image_folder,
    ecg_folder,
    output_dir,
    lr=1e-5,
    batch_size=1,
    num_epochs=1,
    ppo_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_eval_items=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # 1. CARICAMENTO MODELLO (device_map="auto", niente .to("cuda") a mano:
    #    evita i crash da tensori su "meta device" con modelli grandi)
    # =========================================================
    tokenizer, model, _, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava_llama",
        device_map="auto",
        load_4bit=True,   # FIX MEMORIA: GEM 7B fp16 (~14GB) non entra in T4 16GB
        load_8bit=False,  # insieme a Qwen -> offloading su CPU e lentezza estrema.
    )                     # In 4-bit GEM ~4-5GB: tutto in GPU, niente offload.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model.config.output_hidden_states = True

    # =========================================================
    # 2. VISION TOWER — diagnostica invece di un assert muto
    # =========================================================
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor

    # Il preprocessing delle immagini dipende da model.config.image_aspect_ratio:
    # 'anyres' -> process_anyres_image (richiede grid_pinpoints); 'pad'/'ori' ->
    # semplice preprocess CLIP. Questo GEM usa 'ori', quindi NON serve
    # grid_pinpoints. Non si impone piu' quell'attributo (l'assert precedente
    # falliva erroneamente proprio in modalita' 'ori'); si stampa solo la
    # modalita' per diagnostica. process_images(..., model.config) gestira' il
    # ramo giusto dentro RLMultiModalDataset.load_image.
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    print(f"[img] image_aspect_ratio={image_aspect_ratio!r} "
          f"image_processor={type(image_processor).__name__}")
    if image_aspect_ratio == "anyres" and not hasattr(image_processor, "grid_pinpoints"):
        raise AssertionError(
            "image_aspect_ratio='anyres' ma image_processor non ha 'grid_pinpoints': "
            "vision tower non inizializzato in modalita' anyres. Controlla config.json."
        )

    device = next(model.parameters()).device

    # =========================================================
    # 2b. FIX DEVICE ECG TOWER
    # La ecg_tower (ecg_coca) viene materializzata su CPU dentro __init__:
    # CLIPECGTower.load_model chiama get_ecg_encoder(device='cpu') e i suoi
    # pesi arrivano da un file .pt SEPARATO (cpt_wfep_epoch_20.pt), NON dal
    # checkpoint safetensors del modello. Percio' device_map='auto' non la
    # sposta su GPU insieme al resto: resta su cpu mentre input e resto del
    # modello sono su cuda -> dentro il forward della ecg tower si ha
    # 'Expected all tensors to be on the same device' (class_embedding su cpu,
    # input su cuda). class_embedding e' un nn.Parameter, quindi .to(device)
    # lo sposta davvero. (La vision_tower si carica gia' da sola con
    # device_map={'':'cuda'}, quindi non serve toccarla.)
    #
    # Si castra anche al DTYPE del modello (fp16): i pesi ecg_coca arrivano
    # dal .pt in fp32, ma il resto del modello e' fp16; senza questo cast, piu'
    # a valle prepare_inputs_labels_for_multimodal fa torch.cat([ecg_features
    # (fp32), image_features(fp16)]) e fallisce per dtype misto. GEM in
    # inferenza gira comunque tutto in fp16, quindi il cast e' coerente.
    model_dtype = torch.float16  # compute dtype: in 4-bit i param base sono uint8
    ecg_tower_mod = model.get_model().get_ecg_tower()
    if ecg_tower_mod is not None:
        inner = ecg_tower_mod.ecg_tower  # il transformer CoCa .ecg vero e proprio
        # Con low_cpu_mem_usage=True/device_map='auto' il modello viene creato
        # sotto init_empty_weights (parametri su 'meta'); i pesi della ecg_tower
        # arrivano da un .pt separato e NON dal checkpoint safetensors, quindi
        # accelerate non sempre li materializza: alcuni restano su 'meta' (senza
        # dati) e .to() fallisce con 'Cannot copy out of meta tensor; no data!'.
        # Se troviamo parametri meta, RIMATERIALIZZIAMO la ecg_tower dai suoi
        # pesi reali (il .pt il cui path e' in ecg_tower_mod.ecg_tower_name),
        # esattamente come fa get_ecg_encoder, ma direttamente sul device giusto.
        has_meta = any(p.is_meta for p in inner.parameters())
        if has_meta:
            ckpt_path = ecg_tower_mod.ecg_tower_name
            inner.to_empty(device=device)  # meta -> tensori reali (vuoti) su cuda
            _ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            _sd = _ck['state_dict']
            _sd = {k[len('module.ecg.'):]: v for k, v in _sd.items()
                   if k.startswith('module.ecg.')}
            missing, unexpected = inner.load_state_dict(_sd, strict=False)
            print(f"[ecg] ecg_tower rimaterializzata da {ckpt_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        # ora tutti i parametri sono reali: sposta su device + cast al dtype del
        # modello (fp16). Il cast evita, piu' a valle, torch.cat([ecg_features
        # (fp32), image_features(fp16)]) con dtype misto.
        inner.to(device=device, dtype=model_dtype)
        # aggiorna i device/dtype memorizzati dentro il wrapper CLIPECGTower
        try:
            sd = inner.state_dict()
            ecg_tower_mod.device = sd['class_embedding'].device
            ecg_tower_mod.dtype = sd['class_embedding'].dtype
        except Exception:
            pass
        print(f"[ecg] ecg_tower su {device} (dtype={model_dtype})", flush=True)

    # =========================================================
    # 2b-bis. UNIFORMA DTYPE A FP16
    # In 4-bit i moduli quantizzati calcolano in fp16 (bnb_4bit_compute_dtype),
    # ma i moduli NON quantizzati (skip_modules: mm_projector, vision_tower,
    # embed_tokens, lm_head, ecg_tower) transformers 4.38 puo' caricarli in fp32.
    # Nella catena multimodale una feature esce da un modulo 4-bit in fp16 ed
    # entra in uno skip in fp32 -> 'mat1 and mat2 must have the same dtype
    # (Float vs Half)'. Portiamo tutti i parametri float NON quantizzati a fp16
    # (i Params4bit sono uint8 e restano intatti).
    _n_cast = 0
    for _p in model.parameters():
        if _p.dtype == torch.float32:
            _p.data = _p.data.to(torch.float16)
            _n_cast += 1
    print(f"[dtype] parametri float32 -> float16: {_n_cast}", flush=True)

    # =========================================================
    # 2c. DIAGNOSTICA MEMORIA/DEVICE (stampa SUBITO, prima della eval lenta)
    # Se tra i device compare 'cpu' o 'meta', device_map='auto' ha fatto
    # OFFLOADING per mancanza di VRAM (probabile: GEM 7B fp16 ~14GB + Qwen 3B
    # ~6GB sulla stessa T4 da 16GB) -> generazione lentissima. In quel caso
    # la soluzione e' caricare GEM in 4-bit.
    from collections import Counter as _Counter
    _devs = _Counter(str(p.device) for p in model.parameters())
    print(f"[diag] parametri modello per device: {dict(_devs)}", flush=True)
    _hfmap = getattr(model, 'hf_device_map', None)
    if _hfmap:
        _mapdevs = _Counter(str(v) for v in _hfmap.values())
        print(f"[diag] hf_device_map (moduli per device): {dict(_mapdevs)}", flush=True)
    if torch.cuda.is_available():
        _free, _tot = torch.cuda.mem_get_info()
        print(f"[diag] GPU mem: usati {(_tot-_free)/1e9:.1f} GB / {_tot/1e9:.1f} GB "
              f"(liberi {_free/1e9:.1f} GB)", flush=True)

    # =========================================================
    # 3. VALUTAZIONE BASELINE (pesi originali, prima di qualunque LoRA/training)
    # =========================================================
    eval_ds = RLMultiModalDataset(
        rl_json_path=eval_rl_json_path,
        ptbxl_json_path=ptbxl_path,
        image_root=image_folder,
        ecg_root=ecg_folder,
        image_processor=image_processor,
        model_config=model.config,
    )
    if max_eval_items is not None:
        eval_ds.rl_data = eval_ds.rl_data[:max_eval_items]

    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    print("\n===== VALUTAZIONE BASELINE (prima del RL) =====")
    baseline_metrics = evaluate_policy(model, tokenizer, eval_loader, label="baseline (pre-RL)")

    # =========================================================
    # 4. LoRA SOLO sul LLM (QLoRA su base 4-bit)
    # =========================================================
    # NB: NON si usa prepare_model_for_kbit_training: fa un upcast a fp32 di
    # TUTTI i moduli non quantizzati (layernorm ma anche mm_projector/
    # vision_tower/ecg_tower), ri-rompendo la catena multimodale che deve
    # restare in fp16. Ci basta: congelare la base (lo fa get_peft_model) +
    # gradient checkpointing + input-grads, mantenendo tutto in fp16.
    for attr in ["vision_tower", "ecg_tower", "ecg_projector", "mm_projector"]:
        if hasattr(model, attr):
            getattr(model, attr).requires_grad_(False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    hidden_size = model.config.hidden_size
    # value_head in fp16: l'input (hidden_states) e' fp16, un nn.Linear fp32 di
    # default darebbe errore di dtype misto.
    model.value_head = nn.Linear(hidden_size, 1).to(device=device, dtype=torch.float16)

    # NIENTE old_model deepcopy: con base 4-bit (bitsandbytes) il deepcopy e'
    # problematico e raddoppierebbe la memoria. La policy di RIFERIMENTO per il
    # ratio PPO e' il modello base con l'adapter LoRA DISABILITATO
    # (model.disable_adapter()), pattern standard PEFT/TRL.

    # =========================================================
    # 5. DATASET DI TRAINING
    # =========================================================
    train_ds = RLMultiModalDataset(
        rl_json_path=train_rl_json_path,
        ptbxl_json_path=ptbxl_path,
        image_root=image_folder,
        ecg_root=ecg_folder,
        image_processor=image_processor,
        model_config=model.config,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    pad_id = tokenizer.pad_token_id

    # =========================================================
    # 6. LOOP DI TRAINING PPO
    # =========================================================
    for epoch in range(num_epochs):
        print(f"\n===== EPOCH {epoch + 1}/{num_epochs} =====")

        for batch in train_loader:
            model.train()

            reports, response_ids = generate_gem(
                model, tokenizer,
                batch["input_ids"], batch["attention_mask"],
                batch["ecgs"], batch["images"],
            )

            rewards_seq = []
            for rep, oss_RE, diag_RE in zip(reports, batch["rule_obs"], batch["rule_diag"]):
                oss_GEM, diag_GEM = extract_claims_from_report_local(rep, OBS_LIST_STR, DIAG_LIST_STR)
                r = compute_reward_jha25(oss_RE=oss_RE, diag_RE=diag_RE, oss_GEM=oss_GEM, diag_GEM=diag_GEM)
                rewards_seq.append(r)
            rewards_seq = torch.tensor(rewards_seq, dtype=torch.float32, device=model.device)

            full_ids, full_attn, full_labels = _build_full_ids_and_labels(
                batch["input_ids"].to(model.device),
                batch["attention_mask"].to(model.device),
                response_ids.to(model.device),
                pad_id,
            )

            # Fusione multimodale esplicita (la stessa che model.forward()
            # farebbe internamente quando inputs_embeds is None), cosi'
            # recuperiamo anche `new_labels` gia' espansi e allineati a
            # logits/hidden_states: e' l'unico modo corretto per sapere dove
            # sono finiti i token di risposta dopo l'inserimento delle
            # feature ecg/immagine.
            (
                _, new_position_ids, new_attention_mask, _,
                new_inputs_embeds, new_labels,
            ) = model.prepare_inputs_labels_for_multimodal(
                full_ids, None, full_attn, None, full_labels,
                batch["ecgs"].to(model.device), batch["images"].to(model.device),
            )

            outputs_new = model(
                inputs_embeds=new_inputs_embeds,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                output_hidden_states=True,
            )
            logits = outputs_new.logits
            hidden = outputs_new.hidden_states[-1]
            values_all = model.value_head(hidden).squeeze(-1)

            # policy di riferimento = modello base con LoRA disabilitato. La
            # fusione multimodale (towers ecg/vision) NON dipende dall'adapter,
            # quindi riuso new_inputs_embeds/attn/pos gia' calcolati: cambia solo
            # il forward del LLM (adapter off).
            with torch.no_grad(), model.disable_adapter():
                outputs_old = model(
                    inputs_embeds=new_inputs_embeds,
                    attention_mask=new_attention_mask,
                    position_ids=new_position_ids,
                    output_hidden_states=False,
                )
                logits_old = outputs_old.logits

            logprobs_resp, old_logprobs_resp, values_resp, rewards_tokens = [], [], [], []

            for i in range(full_ids.shape[0]):
                resp_positions = (new_labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                if resp_positions.numel() == 0:
                    continue

                tokens_resp = new_labels[i, resp_positions]
                # shift causale: la predizione del token alla posizione p
                # sta in logits[p-1] (posizioni gia' corrette per l'aver
                # incluso le feature ecg/immagine, vedi nota FIX 2 sopra)
                pred_positions = resp_positions - 1

                logits_i = logits[i, pred_positions, :]
                logits_old_i = logits_old[i, pred_positions, :]
                values_i = values_all[i, pred_positions]

                lp = torch.log_softmax(logits_i, dim=-1)
                lp_old = torch.log_softmax(logits_old_i, dim=-1)

                lp_chosen = lp.gather(1, tokens_resp.unsqueeze(-1)).squeeze(-1)
                lp_old_chosen = lp_old.gather(1, tokens_resp.unsqueeze(-1)).squeeze(-1)

                logprobs_resp.append(lp_chosen)
                old_logprobs_resp.append(lp_old_chosen)
                values_resp.append(values_i)
                # Reward SOLO sull'ultimo token della risposta (0 altrove): e' li' che
                # si conclude il report e ha senso attribuire il reward di sequenza.
                # Metterlo su ogni token fa vedere lo stesso reward finale ripetuto
                # ad ogni posizione, distorcendo il credit assignment del GAE.
                r_tok = torch.zeros_like(values_i)
                r_tok[-1] = rewards_seq[i]
                rewards_tokens.append(r_tok)

            if len(logprobs_resp) == 0:
                continue

            logprobs = torch.cat(logprobs_resp)
            old_logprobs = torch.cat(old_logprobs_resp)
            values_flat = torch.cat(values_resp)

            # FIX: GAE va calcolato su valori trattati come COSTANTI (come nel
            # PPO standard). Prima venivano passati i tensori `values_i` con
            # gradiente ancora attaccato al value_head appena calcolato nella
            # stessa forward pass: gli advantages finivano per propagare
            # gradiente nel value_head anche tramite il policy loss, e i
            # returns (target del value loss) dipendevano dalle stesse
            # predizioni che si voleva regredire (target auto-referenziale
            # e mobile). Qui si stacca esplicitamente il grafo prima del GAE.
            advantages_list, returns_list = [], []
            with torch.no_grad():
                for vals_i, rews_i in zip(values_resp, rewards_tokens):
                    adv_i, ret_i = compute_gae(rews_i.detach(), vals_i.detach())
                    advantages_list.append(adv_i)
                    returns_list.append(ret_i)

            advantages = torch.cat(advantages_list)
            returns = torch.cat(returns_list)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(logprobs - old_logprobs.detach())
            clip_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            value_loss = torch.nn.functional.mse_loss(values_flat, returns)
            # NB: questa NON e' la vera entropia della policy (richiederebbe la
            # softmax completa su tutto il vocabolario per ogni posizione, costosa
            # in memoria). E' un'approssimazione basata solo sul token generato:
            # -p(token_scelto)*log p(token_scelto). Con entropy_coef basso (0.01)
            # l'effetto pratico e' piccolo, ma non trattarla come un vero indicatore
            # di esplorazione della policy.
            entropy = -(torch.exp(logprobs) * logprobs).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(
                f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | "
                f"Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f} | "
                f"Entropy: {entropy.item():.4f} | Reward mean: {rewards_seq.mean().item():.4f}"
            )

        # (niente sync old_model: la reference e' sempre il base via disable_adapter)

        ckpt = Path(output_dir) / f"checkpoint-epoch-{epoch + 1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    # =========================================================
    # 7. VALUTAZIONE POST-RL (stesso identico eval set/loader di prima)
    # =========================================================
    print("\n===== VALUTAZIONE POST-RL (dopo il training PPO) =====")
    post_metrics = evaluate_policy(model, tokenizer, eval_loader, label="post-RL")

    print("\nRL multimodale con PPO completato.")

    return {
        "baseline": baseline_metrics,
        "post_rl": post_metrics,
        "checkpoint_dir": str(ckpt),
        "model": model,
        "tokenizer": tokenizer,
    }


# In[ ]:


# ============================================================
# SPLIT TRAIN/EVAL + ESECUZIONE TRAINING CON CONFRONTO BASELINE vs POST-RL
# ============================================================
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

random.seed(42)

rl_dataset_path = BASE / "RL" / "rl_dataset.json"
with open(rl_dataset_path) as f:
    full_rl_dataset = json.load(f)

print(f"Dataset RL totale: {len(full_rl_dataset)} esempi")

shuffled = full_rl_dataset[:]
random.shuffle(shuffled)

# 85% training / 15% valutazione (held-out, MAI vista in training).
# E' su questo 15% che si misura l'impatto reale del RL: se lo si valutasse
# sugli stessi esempi usati in training, i numeri sarebbero ottimistici e
# non direbbero nulla sulla generalizzazione del modello.
split_idx = max(1, int(len(shuffled) * 0.85))
train_items = shuffled[:split_idx]
eval_items = shuffled[split_idx:]

if len(eval_items) == 0:
    raise ValueError(
        "Il dataset RL e' troppo piccolo per ricavare un eval set separato: "
        f"solo {len(full_rl_dataset)} esempi totali. Servono almeno 2 esempi."
    )

print(f"Train: {len(train_items)}  |  Eval (held-out): {len(eval_items)}")

train_json_path = BASE / "RL" / "rl_dataset_train.json"
eval_json_path = BASE / "RL" / "rl_dataset_eval.json"

with open(train_json_path, "w") as f:
    json.dump(train_items, f)
with open(eval_json_path, "w") as f:
    json.dump(eval_items, f)

# Controllo rapido che il modello patchato ci sia davvero, prima di lanciare
# tutto (evita di scoprirlo a meta' caricamento)
model_path = str(BASE / "models" / "GEM_model")
assert Path(model_path).exists(), (
    f"Cartella modello non trovata: {model_path}. "
    "Controlla che il download (snapshot_download) e le patch a config.json "
    "siano stati fatti su questa stessa cartella."
)
print("File nel modello:", os.listdir(model_path) if "os" in dir() else __import__("os").listdir(model_path))

results = train_rl_multimodal(
    model_path=model_path,
    train_rl_json_path=str(train_json_path),
    eval_rl_json_path=str(eval_json_path),
    ptbxl_path=str(BASE / "data" / "ptbxl_subset.json"),
    image_folder=str(BASE / "data" / "ecg_images" / "gen_images"),
    ecg_folder=str(BASE / "data" / "ptbxl_subset"),  # FIX: era "ecg_timeseries/ptbxl_subset", cartella mai creata (il subset lo crea la cella PTBXL_SUBSET dentro data/ptbxl_subset)
    output_dir=str(BASE / "RL" / "checkpoints"),
    lr=1e-5,
    batch_size=1,
    num_epochs=1,
)


# ### **FINE**

# In[ ]:


# ============================================================
# CONFRONTO BASELINE vs POST-RL
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt

baseline = results["baseline"]
post_rl = results["post_rl"]

comparison_df = pd.DataFrame([baseline, post_rl]).set_index("label")
comparison_df["delta_reward"] = comparison_df["reward_mean"] - baseline["reward_mean"]
comparison_df["delta_f1"] = comparison_df["f1_mean"] - baseline["f1_mean"]

print(comparison_df[["n", "precision_mean", "recall_mean", "f1_mean", "reward_mean"]])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

metrics = ["precision_mean", "recall_mean", "f1_mean"]
axes[0].bar(
    [m.replace("_mean", "") for m in metrics],
    [baseline[m] for m in metrics],
    width=0.35, label="baseline (pre-RL)", align="edge",
)
axes[0].bar(
    [m.replace("_mean", "") for m in metrics],
    [post_rl[m] for m in metrics],
    width=-0.35, label="post-RL", align="edge",
)
axes[0].set_ylim(0, 1)
axes[0].set_title("Precision / Recall / F1 (eval set held-out)")
axes[0].legend()

axes[1].bar(["baseline (pre-RL)", "post-RL"],
            [baseline["reward_mean"], post_rl["reward_mean"]],
            color=["gray", "tab:blue"])
axes[1].set_title("Reward media (eval set held-out)")

plt.tight_layout()

out_path = BASE / "RL" / "confronto_baseline_vs_postRL.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"\nGrafico salvato in: {out_path}")
print(
    f"Variazione reward media: {comparison_df.loc['post-RL', 'delta_reward']:+.3f} "
    f"({baseline['reward_mean']:.3f} -> {post_rl['reward_mean']:.3f})"
)
print(
    f"Variazione F1 media:     {comparison_df.loc['post-RL', 'delta_f1']:+.3f} "
    f"({baseline['f1_mean']:.3f} -> {post_rl['f1_mean']:.3f})"
)


# **Nota:** questo confronto e' calcolato sullo stesso eval set held-out (mai usato in training) sia per il modello di partenza sia per il modello dopo il training PPO, con lo stesso identico codice di valutazione (`evaluate_policy`). Un reward/F1 piu' alto sul post-RL indica che il RL ha effettivamente migliorato l'aderenza dei report generati alle osservazioni/diagnosi del rule engine; un valore invariato o peggiore segnala che il training non ha (ancora) prodotto un effetto utile, ed e' un segnale per rivedere reward, learning rate o numero di epoche prima di dedurre altro.
