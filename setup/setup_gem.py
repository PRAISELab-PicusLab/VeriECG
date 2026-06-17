#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup GEM + ECG-CoCa environment
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download


# ============================================================
# 1. PATH DI BASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

GEM_DIR = BASE_DIR / "external" / "GEM"
MODEL_DIR = BASE_DIR / "models" / "GEM_model"
COCA_DIR = BASE_DIR / "models" / "ecg_coca" / "checkpoint"
DATA_DIR = BASE_DIR / "data"
BENCH_DIR = DATA_DIR / "ecg_bench"
TS_DIR = DATA_DIR / "ecg_timeseries" / "ptbxl"
IMG_DIR = DATA_DIR / "ecg_images" / "gen_images" / "ptb-xl-gen"

# Crea tutte le cartelle necessarie
for d in [GEM_DIR, MODEL_DIR, COCA_DIR, BENCH_DIR, TS_DIR, IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CLONA LA REPO GEM
# ============================================================

if not GEM_DIR.exists() or not any(GEM_DIR.iterdir()):
    os.system(f"git clone https://github.com/lanxiang1017/GEM.git {GEM_DIR}")
    print("✔️ GEM clonato")
else:
    print("ℹ️ GEM già presente, salto clone")


# ============================================================
# 3. SCARICA IL MODELLO GEM DA HUGGINGFACE
# ============================================================

print("⬇️ Download modello GEM...")
snapshot_download(
    repo_id="LANSG/GEM",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False
)
print("✔️ Modello GEM scaricato")


# ============================================================
# 4. SCARICA BENCHMARK ECG-GROUNDING
# ============================================================

print("⬇️ Download ECG-Grounding benchmark...")

bench_file = BENCH_DIR / "ecg-grounding-test-ptbxl.json"

path = hf_hub_download(
    repo_id="LANSG/ECG-Grounding",
    filename="ecg_jsons/ecg-grounding-test-ptbxl.json",
    repo_type="dataset"
)

shutil.copy(path, bench_file)
print(f"✔️ Benchmark salvato in {bench_file}")


# ============================================================
# 5. SCARICA CHECKPOINT ECG-CoCa
# ============================================================

print("⬇️ Download checkpoint ECG-CoCa...")

os.system(
    f"gdown 1wOKYfkb-Nep0WzYZz9-n66oTzp_4cky7 -O {COCA_DIR}/cpt_wfep_epoch_20.pt"
)

print("✔️ Checkpoint ECG-CoCa scaricato")


# ============================================================
# 6. CREA CONFIG coca_ViT-B-32.json
# ============================================================

print("📝 Creazione config coca_ViT-B-32.json...")

config_coca = {
    "embed_dim": 512,
    "ecg_cfg": {
        "layers": 12,
        "width": 768,
        "head_width": 64,
        "mlp_ratio": 4.0,
        "patch_size": 50,
        "seq_length": 5000,
        "lead_num": 12,
        "attentional_pool": False,
        "output_tokens": True
    },
    "text_cfg": {
        "context_length": 76,
        "vocab_size": 49408,
        "width": 512,
        "heads": 8,
        "layers": 12,
        "embed_cls": True,
        "output_tokens": True
    },
    "multimodal_cfg": {
        "context_length": 76,
        "vocab_size": 49408,
        "width": 512,
        "heads": 8,
        "layers": 12,
        "attn_pooler_heads": 8
    },
    "custom_text": True
}

config_path = GEM_DIR / "ecg_coca" / "open_clip" / "model_configs"
config_path.mkdir(parents=True, exist_ok=True)

with open(config_path / "coca_ViT-B-32.json", "w") as f:
    json.dump(config_coca, f, indent=2)

print("✔️ Config creato")


# ============================================================
# 7. PATCH CONFIG.JSON DEL MODELLO
# ============================================================

print("📝 Patch config.json (mm_ecg_tower)...")

model_cfg = MODEL_DIR / "config.json"
with open(model_cfg) as f:
    cfg = json.load(f)

cfg["mm_ecg_tower"] = str(COCA_DIR / "cpt_wfep_epoch_20.pt")

with open(model_cfg, "w") as f:
    json.dump(cfg, f, indent=2)

print("✔️ config.json aggiornato")


# ============================================================
# 8. PATCH file_utils.py (weights_only=False)
# ============================================================

print("🩹 Patch file_utils.py...")

file_utils = GEM_DIR / "ecg_coca" / "training" / "file_utils.py"
os.system(
    f"sed -i 's|torch.load(f, map_location=map_location)|torch.load(f, map_location=map_location, weights_only=False)|g' {file_utils}"
)

print("✔️ file_utils.py patchato")


# ============================================================
# 9. PATCH model_ecg_resume.py (load_4bit=True)
# ============================================================

print("🩹 Patch model_ecg_resume.py...")

resume_file = GEM_DIR / "llava" / "eval" / "model_ecg_resume.py"
os.system(
    f"sed -i 's/load_pretrained_model(model_path, args.model_base, model_name)/load_pretrained_model(model_path, args.model_base, model_name, load_4bit=True)/' {resume_file}"
)

print("✔️ model_ecg_resume.py patchato")


# ============================================================
# 10. PATCH builder.py (skip modules)
# ============================================================

print("🩹 Patch builder.py...")

builder_file = GEM_DIR / "llava" / "model" / "builder.py"

with open(builder_file, "r") as f:
    content = f.read()

old = """    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )"""

new = """    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            llm_int8_skip_modules=["ecg_tower", "mm_projector", "vision_tower", "lm_head", "embed_tokens"]
        )"""

if old in content:
    content = content.replace(old, new)
    with open(builder_file, "w") as f:
        f.write(content)
    print("✔️ builder.py patchato")
else:
    print("⚠️ Patch non trovata, verifica manuale necessaria")
