# DEFINIZIONE LOOP DI TRAINING (CON PPO)

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


os.environ["ACCELERATE_DISABLE_MAPPING"] = "1"
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"


BASE = Path(__file__).resolve().parent
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


# ============================================================
# IMPORT CROSS-FILE 
#   - reward_definition.py: le due reward + il selettore reward_fn_training
#   - mapping.py: estrazione claim con Qwen + i vocabolari OBS/DIAG
# ============================================================
from reward_definition import (
    compute_reward_jha25,
    compute_claim_metrics,
    compute_reward_strutturata,
    reward_fn_training,   # = la reward scelta in reward_definition (default: strutturata)
)
from mapping import (
    extract_claims_from_report_local,
    OBS_LIST_STR,
    DIAG_LIST_STR,
)

# Controllo esplicito: queste definizioni devono ora essere presenti
_REQUIRED_NAMES = [
    "compute_reward_jha25",
    "reward_fn_training",
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
def generate_gem(model, tokenizer, input_ids, attention_mask, ecgs, images, max_new_tokens=512,
                 do_sample=False, temperature=1.0):
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
        gen_kwargs = dict(
            inputs=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            ecgs=ecgs.to(model.device),
            images=images.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        if do_sample:
            # Rollout ON-POLICY: si campiona dalla distribuzione PIENA a
            # temperatura, SENZA top_p/top_k. Troncare renderebbe il rollout
            # off-policy rispetto ai logprob del PPO -> importance ratio distorto.
            # (I logprob del PPO sono calcolati alla STESSA temperatura, cioe'
            # logits/sample_temperature, quindi il rollout resta on-policy.) Si sovrascrivono
            # i valori del generation_config di GEM (temperature=0.7/top_p=0.8/
            # top_k=20) che altrimenti verrebbero applicati in sampling.
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 0
        out = model.generate(**gen_kwargs)
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
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
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
    diagnostics = []   # per-campione: referto GEM + claim Qwen + claim rule engine + metriche

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
        for rep, oss_RE, diag_RE, _exid in zip(reports, batch["rule_obs"], batch["rule_diag"], batch["ids"]):
            oss_GEM, diag_GEM = extract_claims_from_report_local(rep, OBS_LIST_STR, DIAG_LIST_STR)
            p, r, f1 = compute_claim_metrics(oss_RE, diag_RE, oss_GEM, diag_GEM)
            reward = reward_fn_training(oss_RE, diag_RE, oss_GEM, diag_GEM)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            rewards.append(reward)
            diagnostics.append({
                "id": _exid, "report": rep,
                "oss_GEM": oss_GEM, "diag_GEM": diag_GEM,
                "oss_RE": oss_RE, "diag_RE": diag_RE,
                "precision": p, "recall": r, "f1": f1, "reward": reward,
            })
            # DIAGNOSTICA LIVE: confronto claim estratti (GEM->Qwen) vs rule engine
            print(f"  [diag] id={_exid}  f1={f1:.3f}  reward={reward:.2f}", flush=True)
            print(f"    OBS  GEM/Qwen = {oss_GEM}", flush=True)
            print(f"    OBS  RULE-ENG = {oss_RE}", flush=True)
            print(f"    DIAG GEM/Qwen = {diag_GEM}  |  DIAG RULE-ENG = {diag_RE}", flush=True)
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
    return result, diagnostics


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
    kl_coef=0.2,       # peso della penalita' KL verso la policy di riferimento (base)
    sample_temperature=1.0,  # temperatura del rollout CAMPIONATO in training (1.0 = on-policy). L'eval resta greedy.
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
    # 2. VISION TOWER 
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
    baseline_metrics, baseline_diag = evaluate_policy(model, tokenizer, eval_loader, label="baseline (pre-RL)")

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
        # lm_head NON piu' in modules_to_save: era interamente addestrabile
        # (~150M param) e permetteva alla policy di stravolgere la generazione
        # in pochi step (nel run precedente -> elenco di osservazioni inventate).
        # Ora si addestrano SOLO gli adapter LoRA su q/k/v/o: capacita' ridotta,
        # generazione piu' stabile.
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
    training_log = []      # metriche per-step (per grafico e log)
    train_diag = []        # diagnostica per-campione durante il training
    ckpt = None
    _dbg_first = True   # stampa la diagnostica del grafo solo al PRIMO step
    for epoch in range(num_epochs):
        print(f"\n===== EPOCH {epoch + 1}/{num_epochs} =====")

        for batch in train_loader:
            model.train()

            # ROLLOUT campionato (do_sample=True): l'esplorazione e' il
            # prerequisito perche' il PPO impari. Con greedy il modello genera
            # sempre lo stesso report -> nessuna traiettoria alternativa -> nessun
            # apprendimento. L'eval (evaluate_policy) resta invece greedy.
            reports, response_ids = generate_gem(
                model, tokenizer,
                batch["input_ids"], batch["attention_mask"],
                batch["ecgs"], batch["images"],
                do_sample=True, temperature=sample_temperature,
            )

            rewards_seq = []
            for rep, oss_RE, diag_RE, _exid in zip(reports, batch["rule_obs"], batch["rule_diag"], batch["ids"]):
                oss_GEM, diag_GEM = extract_claims_from_report_local(rep, OBS_LIST_STR, DIAG_LIST_STR)
                r = reward_fn_training(oss_RE=oss_RE, diag_RE=diag_RE, oss_GEM=oss_GEM, diag_GEM=diag_GEM)
                rewards_seq.append(r)
                train_diag.append({
                    "epoch": epoch + 1, "id": _exid, "report": rep,
                    "oss_GEM": oss_GEM, "diag_GEM": diag_GEM,
                    "oss_RE": oss_RE, "diag_RE": diag_RE, "reward": r,
                })
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

            # FIX FLUSSO GRADIENTE: con gradient checkpointing, per costruire il
            # grafo all'indietro fino ai parametri interni (LoRA) il tensore di
            # INPUT deve richiedere gradiente. Passando inputs_embeds gia' pronti
            # si bypassa l'hook di enable_input_require_grads (che agisce solo sul
            # percorso input_ids->embeddings): senza questo il gradiente NON
            # arrivava alla LoRA (che restava congelata -> KL=0), mentre la value
            # head, alimentata direttamente, si aggiornava lo stesso.
            if not new_inputs_embeds.requires_grad:
                new_inputs_embeds.requires_grad_(True)

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

                # ON-POLICY: i logprob sono calcolati alla STESSA temperatura del
                # rollout campionato (logits / sample_temperature), per la policy
                # nuova E per la base/riferimento. Cosi' l'importance ratio e il KL
                # sono coerenti con la distribuzione da cui si e' campionato.
                # A sample_temperature=1.0 e' un no-op (divisione per 1).
                lp = torch.log_softmax(logits_i / sample_temperature, dim=-1)
                lp_old = torch.log_softmax(logits_old_i / sample_temperature, dim=-1)

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

            # Penalita' KL verso la policy di RIFERIMENTO (base, adapter LoRA
            # disabilitato -> gli stessi old_logprobs del ratio PPO): tiene la
            # policy vicina al modello originale ed evita che derivi. Stimatore
            # k3 di Schulman: sempre >= 0, bassa varianza.
            logratio = logprobs - old_logprobs.detach()
            approx_kl = (torch.exp(-logratio) - 1 + logratio).mean()

            loss = (policy_loss + value_coef * value_loss
                    - entropy_coef * entropy + kl_coef * approx_kl)

            optimizer.zero_grad()
            loss.backward()
            # DIAGNOSTICA: norma del gradiente dei SOLI parametri LoRA (dopo
            # backward, prima del clipping). ~0 = il gradiente non arriva alla
            # policy (blocco del flusso); > 0 = la policy si aggiorna davvero.
            _grad_lora = sum(
                (p.grad.detach().norm().item() ** 2)
                for n, p in model.named_parameters()
                if p.requires_grad and "lora_" in n.lower() and p.grad is not None
            ) ** 0.5

            # ---- DEBUG GRAFO (una sola volta, al primo step) ----
            # Serve SE il fix del requires_grad non bastasse: fotografa DOVE si
            # rompe il flusso. Interpretazione:
            #  - logits/logprobs/loss.requires_grad = False  -> grafo scollegato
            #    (la LoRA non e' nel percorso del forward): problema strutturale.
            #  - param LoRA con grad=None (tutti)            -> il gradiente non
            #    raggiunge la LoRA (checkpointing/inputs_embeds ancora bloccante).
            #  - grad LoRA presente ma tutto-zero + advantages ~0 -> il grafo va,
            #    ma il SEGNALE e' nullo (reward normalizzata via): serve cambiare
            #    il calcolo dell'advantage (es. GRPO), non il flusso.
            if _dbg_first:
                _dbg_first = False
                _lp = [(n, p) for n, p in model.named_parameters()
                       if p.requires_grad and "lora_" in n.lower()]
                _n_none = sum(1 for _, p in _lp if p.grad is None)
                _n_zero = sum(1 for _, p in _lp
                              if p.grad is not None and p.grad.detach().abs().sum().item() == 0.0)
                print("[DEBUG grafo] "
                      f"inputs_embeds.requires_grad={new_inputs_embeds.requires_grad} | "
                      f"logits.requires_grad={logits.requires_grad} | "
                      f"logprobs.requires_grad={logprobs.requires_grad} | "
                      f"loss.requires_grad={loss.requires_grad}", flush=True)
                print("[DEBUG grafo] "
                      f"param LoRA={len(_lp)} | grad=None: {_n_none} | grad tutto-zero: {_n_zero} | "
                      f"gradLoRA={_grad_lora:.3e}", flush=True)
                print("[DEBUG grafo] "
                      f"advantages: mean={advantages.mean().item():.3e} std={advantages.std().item():.3e} "
                      f"min={advantages.min().item():.3e} max={advantages.max().item():.3e} | "
                      f"n_token_risposta={advantages.numel()}", flush=True)

            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            training_log.append({
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "kl": float(approx_kl.item()),
                "grad_lora": float(_grad_lora),
                "reward_mean": float(rewards_seq.mean().item()),
            })
            print(
                f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | "
                f"Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f} | "
                f"Entropy: {entropy.item():.4f} | KL: {approx_kl.item():.6f} | gradLoRA: {_grad_lora:.3e} | Reward mean: {rewards_seq.mean().item():.4f}"
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
    post_metrics, post_diag = evaluate_policy(model, tokenizer, eval_loader, label="post-RL")

    print("\nRL multimodale con PPO completato.")

    # =========================================================
    # 8. SALVATAGGIO OUTPUT DELLA RUN (diagnostica, log, grafici)
    #    output_dir e' la cartella dedicata a questa run (TIRO/output/run_XXX).
    # =========================================================
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def _save_json(name, obj):
        with open(run_dir / name, "w") as _f:
            json.dump(obj, _f, indent=2, ensure_ascii=False)

    _save_json("diagnostica_baseline.json", baseline_diag)
    _save_json("diagnostica_post_rl.json", post_diag)
    _save_json("diagnostica_training.json", train_diag)
    _save_json("training_log.json", training_log)
    _save_json("metriche_riassunto.json", {"baseline": baseline_metrics, "post_rl": post_metrics})

    # log leggibile: referto GEM + claim Qwen + claim rule engine + reward, per campione
    def _write_readable(name, diag_list, titolo):
        lines = [f"===== {titolo} =====\n"]
        for d in diag_list:
            head = f"\n--- id: {d['id']}"
            if "epoch" in d:
                head += f" | epoch: {d['epoch']}"
            if "f1" in d:
                head += f" | f1: {d['f1']:.3f}"
            head += f" | reward: {d['reward']:.2f} ---\n"
            lines.append(head)
            lines.append("[REFERTO GEM]\n" + str(d["report"]).strip() + "\n")
            lines.append("[CLAIM GEM->QWEN]   obs=" + str(d["oss_GEM"]) + "  diag=" + str(d["diag_GEM"]) + "\n")
            lines.append("[CLAIM RULE ENGINE] obs=" + str(d["oss_RE"]) + "  diag=" + str(d["diag_RE"]) + "\n")
        with open(run_dir / name, "w") as _f:
            _f.writelines(lines)

    _write_readable("referti_e_claim_baseline.txt", baseline_diag, "BASELINE (pre-RL)")
    _write_readable("referti_e_claim_post_rl.txt", post_diag, "POST-RL")
    _write_readable("referti_e_claim_training.txt", train_diag, "TRAINING")

    # ---- RECAP DELLA RUN (configurazione + dettagli implementativi) ----
    # Salva un txt leggibile con tutto ciò che serve a ricostruire questa prova:
    # iperparametri, temperatura del rollout, dettagli di RL/reward, dati, risultati.
    import datetime as _dt
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    _lc = lora_config
    _info = f"""===== RECAP RUN =====
Timestamp        : {_dt.datetime.now().isoformat(timespec='seconds')}
Output dir       : {run_dir}
Model path       : {model_path}

--- DATI ---
Dataset RL       : train={len(train_ds.rl_data)}  |  eval/held-out={len(eval_ds.rl_data)}
train_rl_json    : {train_rl_json_path}
eval_rl_json     : {eval_rl_json_path}
ptbxl_index      : {ptbxl_path}
image_folder     : {image_folder}
ecg_folder       : {ecg_folder}

--- IPERPARAMETRI ---
lr                                    : {lr}
batch_size                            : {batch_size}
num_epochs                            : {num_epochs}
ppo_clip                              : {ppo_clip}
value_coef                            : {value_coef}
entropy_coef                          : {entropy_coef}
kl_coef                               : {kl_coef}
sample_temperature (rollout training) : {sample_temperature}

--- REWARD ---
Funzione reward usata (training + eval): {reward_fn_training.__name__}
  - compute_reward_jha25:       F-beta densa (scale=10, beta=0.5) su claim uniti (obs+diag).
  - compute_reward_strutturata: 4 quadranti (diagnosi x osservazioni), R_pos=10 P_o=6 P_d=3.
Metriche precision/recall/f1: claim-level stile Jha25 / DocLens (claim = set(obs+diag)).

--- RL / DETTAGLI IMPLEMENTATIVI ---
- GEM-7B caricato in 4-bit (QLoRA). LoRA: r={_lc.r}, alpha={_lc.lora_alpha}, dropout={_lc.lora_dropout}, target={list(_lc.target_modules)}.
- lm_head NON addestrabile (rimosso da modules_to_save): si allenano solo gli adapter LoRA + la value_head (fp16).
- Policy di RIFERIMENTO per il ratio/KL = modello base con adapter LoRA disabilitato (niente deepcopy).
- Rollout di TRAINING CAMPIONATO: do_sample=True, temperature={sample_temperature}, top_p=1.0, top_k=0
  (on-policy: sovrascrive il generation_config di GEM 0.7/0.8/20). VALUTAZIONI baseline/post-RL: GREEDY.
- Penalita' KL verso il base: stimatore k3 di Schulman (sempre >= 0), coef={kl_coef}.
- Reward assegnata SOLO sull'ultimo token della risposta; GAE gamma=1.0, lam=0.95;
  advantages normalizzati; values .detach() prima del GAE.
- Fusione multimodale via tokenizer_image_token (IMAGE_TOKEN_INDEX); posizioni di risposta
  ricavate dai labels espansi (!= IGNORE_INDEX), non da conteggi pre-fusione.
- Mapping claim: Qwen2.5-3B-Instruct (chat template, greedy, decodifica solo token nuovi);
  vocabolario dai file slim mapping/OBS_LIST.json (id osservazioni) e mapping/DIAG_LIST.json (diagnosis_id),
  con filtro _extract_ids sul vocabolario reale.

--- PARAMETRI ADDESTRABILI ---
trainable={_trainable:,}  |  totali={_total:,}  |  {100 * _trainable / _total:.3f}%

--- RISULTATI ---
baseline (pre-RL): precision={baseline_metrics['precision_mean']:.4f}  recall={baseline_metrics['recall_mean']:.4f}  f1={baseline_metrics['f1_mean']:.4f}  reward={baseline_metrics['reward_mean']:.4f}
post-RL          : precision={post_metrics['precision_mean']:.4f}  recall={post_metrics['recall_mean']:.4f}  f1={post_metrics['f1_mean']:.4f}  reward={post_metrics['reward_mean']:.4f}
variazione f1     : {post_metrics['f1_mean'] - baseline_metrics['f1_mean']:+.4f}
variazione reward : {post_metrics['reward_mean'] - baseline_metrics['reward_mean']:+.4f}
n. step di training: {len(training_log)}
===== fine recap =====
"""
    with open(run_dir / "run_info.txt", "w", encoding="utf-8") as _f:
        _f.write(_info)
    print("  - run_info.txt : recap configurazione + dettagli della run")

    # ---- GRAFICO 1: confronto baseline vs post-RL ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _m = ["precision_mean", "recall_mean", "f1_mean"]
    _lab = [x.replace("_mean", "") for x in _m]
    _fig, _ax = _plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].bar(_lab, [baseline_metrics[x] for x in _m], width=0.35, align="edge", label="baseline")
    _ax[0].bar(_lab, [post_metrics[x] for x in _m], width=-0.35, align="edge", label="post-RL")
    _ax[0].set_ylim(0, 1)
    _ax[0].set_title("Precision / Recall / F1 (eval held-out)")
    _ax[0].legend()
    _ax[1].bar(["baseline", "post-RL"],
               [baseline_metrics["reward_mean"], post_metrics["reward_mean"]],
               color=["gray", "tab:blue"])
    _ax[1].set_title("Reward media (eval held-out)")
    _fig.tight_layout()
    _fig.savefig(run_dir / "confronto_baseline_vs_postRL.png", dpi=150)
    _plt.close(_fig)

    # ---- GRAFICO 2: curve di training ----
    if training_log:
        _steps = list(range(1, len(training_log) + 1))
        _fig2, _ax2 = _plt.subplots(figsize=(9, 5))
        for _k in ["loss", "policy_loss", "value_loss", "entropy", "reward_mean"]:
            _ax2.plot(_steps, [t[_k] for t in training_log], marker="o", label=_k)
        _ax2.set_xlabel("step di training")
        _ax2.set_ylabel("valore")
        _ax2.set_title("Andamento del training PPO")
        _ax2.legend()
        _fig2.tight_layout()
        _fig2.savefig(run_dir / "curve_training.png", dpi=150)
        _plt.close(_fig2)

    print(f"\nOutput della run salvato in: {run_dir}")
    print("  - referti_e_claim_*.txt : referto GEM + claim Qwen + claim rule engine + reward")
    print("  - diagnostica_*.json, training_log.json, metriche_riassunto.json")
    print("  - confronto_baseline_vs_postRL.png, curve_training.png")

    return {
        "baseline": baseline_metrics,
        "post_rl": post_metrics,
        "baseline_diag": baseline_diag,
        "post_rl_diag": post_diag,
        "training_diag": train_diag,
        "training_log": training_log,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt) if ckpt is not None else None,
        "model": model,
        "tokenizer": tokenizer,
    }

