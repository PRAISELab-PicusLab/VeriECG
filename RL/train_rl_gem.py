"""
train_rl_gem.py

Pipeline RL multimodale per GEM:
- Input: immagine ECG + segnale ECG + prompt
- GEM genera un referto naturale (passaggio 1)
- LLM locale (llama3.3_70b) estrae i claim
- Confronto con rule engine → reward
- Costruzione prompt JSON strutturato
- GEM genera un nuovo referto (passaggio 2)
- Loss RL = -reward
- Backprop su GEM

Richiede:
- ptbxl_subset.json (dataset multimodale originale)
- rl_dataset.json (dataset RL con id, prompt, rule_obs, rule_diag)
- mapping_local.py (llama3.3_70b)
- reward_engine.py
"""

import sys
sys.path.append("/content/drive/MyDrive/Tiro/GEM")


import os
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from PIL import Image
import wfdb
import numpy as np

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_anyres_image

# reward
from reward_engine import compute_reward_jha25

# mapping locale
from mapping_local import (
    extract_claims_from_report_local,
    OBS_LIST_STR,
    DIAG_LIST_STR,
)

# ============================================================
# DATASET RL MULTIMODALE
# ============================================================

class RLMultiModalDataset(Dataset):
    def __init__(self, rl_json_path, ptbxl_json_path, image_root, ecg_root, image_processor, ecg_seq_len=5000):
        super().__init__()

        # dataset RL (id, prompt, rule_obs, rule_diag)
        with open(rl_json_path, "r") as f:
            self.rl_data = json.load(f)

        # dataset multimodale originale
        with open(ptbxl_json_path, "r") as f:
            ptbxl = json.load(f)

        # mappa id → (ecg_file, image_file)
        self.index = {x["id"]: x for x in ptbxl}

        self.image_root = Path(image_root)
        self.ecg_root = Path(ecg_root)
        self.image_processor = image_processor
        self.ecg_seq_len = ecg_seq_len

    def __len__(self):
        return len(self.rl_data)

    def load_ecg(self, rel_path):
        """Carica segnale ECG da file .hea usando wfdb."""
        record_path = self.ecg_root / rel_path
        ecg = wfdb.rdsamp(str(record_path))[0]
        ecg[np.isnan(ecg)] = 0
        ecg[np.isinf(ecg)] = 0
        ecg = torch.tensor(ecg.T, dtype=torch.float32)  # (12, L)

        # padding/truncation
        c, L = ecg.shape
        if L < self.ecg_seq_len:
            new = torch.zeros((c, self.ecg_seq_len))
            new[:, :L] = ecg
            ecg = new
        else:
            ecg = ecg[:, :self.ecg_seq_len]

        return ecg

    def load_image(self, rel_path):
        """Carica immagine ECG."""
        img_path = self.image_root / rel_path
        img = Image.open(img_path).convert("RGB")
        img = self.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
        return img

    def __getitem__(self, idx):
        item = self.rl_data[idx]
        ecg_id = item["id"]

        # multimodale
        entry = self.index[ecg_id]
        ecg_file = entry["ecg"]
        image_file = entry["image"]

        ecg = self.load_ecg(ecg_file)
        image = self.load_image(image_file)

        return {
            "id": ecg_id,
            "prompt": item["prompt"],
            "rule_obs": item["rule_observations"],
            "rule_diag": item["rule_diagnoses"],
            "ecg": ecg,
            "image": image,
        }


def collate_fn(batch, tokenizer):
    prompts = [b["prompt"] for b in batch]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    ecgs = torch.stack([b["ecg"] for b in batch])
    images = torch.stack([b["image"] for b in batch])

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "ecgs": ecgs,
        "images": images,
        "rule_obs": [b["rule_obs"] for b in batch],
        "rule_diag": [b["rule_diag"] for b in batch],
        "ids": [b["id"] for b in batch],
        "prompts": prompts,
    }

# ============================================================
# GENERAZIONE CON GEM
# ============================================================

def generate_gem(model, tokenizer, input_ids, attention_mask, ecgs, images, max_new_tokens=512):
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            ecgs=ecgs.to(model.device),
            images=images.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)

# ============================================================
# TRAINING RL MULTIMODALE
# ============================================================

def train_rl_multimodal(
    model_path,
    rl_dataset_path,
    ptbxl_path,
    image_folder,
    ecg_folder,
    output_dir,
    lr=1e-5,
    batch_size=1,
    num_epochs=1,
):

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # Carica GEM
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name="llava"
    )
    model.config.use_cache = False

    # inizializza vision tower
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor

    # ------------------------------
    # Dataset multimodale RL
    # ------------------------------
    dataset = RLMultiModalDataset(
        rl_json_path=rl_dataset_path,
        ptbxl_json_path=ptbxl_path,
        image_root=image_folder,
        ecg_root=ecg_folder,
        image_processor=image_processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # ------------------------------
    # Ottimizzatore
    # ------------------------------
    optimizer = AdamW(model.parameters(), lr=lr)

    # ------------------------------
    # Loop RL
    # ------------------------------
    for epoch in range(num_epochs):
        print(f"\n===== EPOCH {epoch+1}/{num_epochs} =====")

        for batch in loader:
            # --------------------------
            # 1) Primo passaggio: GEM genera report naturale
            # --------------------------
            reports = generate_gem(
                model,
                tokenizer,
                batch["input_ids"],
                batch["attention_mask"],
                batch["ecgs"],
                batch["images"],
            )

            # --------------------------
            # 2) Mapping claim con Dracarys
            # --------------------------
            rewards = []
            for rep, oss_RE, diag_RE in zip(reports, batch["rule_obs"], batch["rule_diag"]):
                oss_GEM, diag_GEM = extract_claims_from_report_local(
                    rep, OBS_LIST_STR, DIAG_LIST_STR
                )
                r = compute_reward_jha25(
                    oss_RE=oss_RE,
                    diag_RE=diag_RE,
                    oss_GEM=oss_GEM,
                    diag_GEM=diag_GEM,
                )
                rewards.append(r)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=model.device)

            # --------------------------
            # 3) Costruzione prompt JSON strutturato
            # --------------------------
            json_prompts = []
            for rep, oss_RE, diag_RE, r in zip(reports, batch["rule_obs"], batch["rule_diag"], rewards):
                json_prompts.append(
                    json.dumps({
                        "ecg_image": "IMAGE_INPUT",
                        "ecg_signal": "ECG_INPUT",
                        "previous_report": rep,
                        "rule_engine_report": {
                            "observations": oss_RE,
                            "diagnoses": diag_RE
                        },
                        "reward": float(r.item()),
                        "instruction": "Generate an improved ECG report based on the rule engine output."
                    })
                )

            enc2 = tokenizer(
                json_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )

            # --------------------------
            # 4) Secondo passaggio: GEM genera nuovo report
            # --------------------------
            new_reports = generate_gem(
                model,
                tokenizer,
                enc2["input_ids"],
                enc2["attention_mask"],
                batch["ecgs"],
                batch["images"],
            )

            # --------------------------
            # 5) Loss RL = -reward
            # --------------------------
            loss = -rewards.mean()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(f"Loss RL: {loss.item():.4f} | Reward: {rewards.mean().item():.4f}")

        # Salvataggio epoch
        ckpt = Path(output_dir) / f"checkpoint-epoch-{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    print("\n🎉 RL multimodale completato!")
