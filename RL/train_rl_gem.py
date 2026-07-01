"""
train_rl_gem PPO — versione A100 (CUDA, senza Accelerate)
"""

# ============================================================
# AMBIENTE
# ============================================================

import os
os.environ["ACCELERATE_DISABLE_MAPPING"] = "1"
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

# ============================================================

BASE = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE / "models" / "GEM_model"
PTBXL_PATH = BASE / "data" / "ptbxl_subset.json"
RL_DATASET = BASE / "progetto" / "RL" / "rl_dataset.json"

IMAGE_FOLDER = BASE / "data" / "ecg_images" / "gen_images"
ECG_FOLDER = BASE / "data" / "ptbxl_subset"
OUTPUT_DIR = BASE / "RL" / "checkpoints"

import sys
sys.path.append(str(BASE / "GEM"))
sys.path.append(str(BASE / "mapping"))
sys.path.append(str(BASE))

# ============================================================
# LIBRERIE
# ============================================================

import json
from pathlib import Path
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model

from PIL import Image
import wfdb
import numpy as np

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_anyres_image

from reward_engine import compute_reward_jha25
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

        with open(rl_json_path, "r") as f:
            self.rl_data = json.load(f)

        with open(ptbxl_json_path, "r") as f:
            ptbxl = json.load(f)

        self.index = {x["id"]: x for x in ptbxl}

        self.image_root = Path(image_root)
        self.ecg_root = Path(ecg_root)
        self.image_processor = image_processor
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

        patches = process_anyres_image(
            img,
            self.image_processor,
            self.image_processor.grid_pinpoints
        )
        return patches

    def __getitem__(self, idx):
        item = self.rl_data[idx]
        ecg_id = item["id"]

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
# GENERAZIONE CON GEM (solo testo, senza grad)
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
# TRAINING RL MULTIMODALE CON PPO
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
    ppo_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):

    os.makedirs(output_dir, exist_ok=True)

    # usa il builder LLaVA/GEM
    tokenizer, model, _, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava_llama",
    )
    model = model.to("cuda")
    model.config.use_cache = False
    model.config.output_hidden_states = True

    # vision tower + image processor
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor

    assert hasattr(image_processor, "grid_pinpoints")

    # congela Vision Tower, ECG Tower, mm_projector
    if hasattr(model, "vision_tower"):
        for p in model.vision_tower.parameters():
            p.requires_grad = False
    if hasattr(model, "ecg_tower"):
        for p in model.ecg_tower.parameters():
            p.requires_grad = False
    if hasattr(model, "ecg_projector"):
        for p in model.ecg_projector.parameters():
            p.requires_grad = False
    if hasattr(model, "mm_projector"):
        for p in model.mm_projector.parameters():
            p.requires_grad = False

    # LoRA solo sul LLM
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

    # Value head
    hidden_size = model.config.hidden_size
    model.value_head = nn.Linear(hidden_size, 1).cuda()

    # Old policy
    old_model = copy.deepcopy(model)
    old_model.eval()
    old_model.config.output_hidden_states = True
    for p in old_model.parameters():
        p.requires_grad = False

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

    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    # ============================================================
    # LOOP EPOCH
    # ============================================================

    for epoch in range(num_epochs):
        print(f"\n===== EPOCH {epoch+1}/{num_epochs} =====")

        for batch in loader:
            model.train()

            # 1) Generazione report
            reports = generate_gem(
                model,
                tokenizer,
                batch["input_ids"],
                batch["attention_mask"],
                batch["ecgs"],
                batch["images"],
            )

            # 2) Reward per sequenza
            rewards_seq = []
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
                rewards_seq.append(r)

            rewards_seq = torch.tensor(rewards_seq, dtype=torch.float32, device=model.device)

            # 3) Costruzione prompt+risposta
            full_texts = []
            for prompt, rep in zip(batch["prompts"], reports):
                full_texts.append(prompt + "\n" + rep)

            enc_full = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            enc_prompt = tokenizer(
                batch["prompts"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)
            prompt_lengths = enc_prompt["attention_mask"].sum(dim=1).tolist()

            # 4) Forward nuova policy (logits + value)
            outputs_new = model(
                input_ids=enc_full["input_ids"],
                attention_mask=enc_full["attention_mask"],
                ecgs=batch["ecgs"].to(model.device),
                images=batch["images"].to(model.device),
            )

            logits = outputs_new.logits
            hidden = outputs_new.hidden_states[-1]
            values_all = model.value_head(hidden).squeeze(-1)

            # 5) Forward old policy (solo logprobs)
            with torch.no_grad():
                outputs_old = old_model(
                    input_ids=enc_full["input_ids"],
                    attention_mask=enc_full["attention_mask"],
                    ecgs=batch["ecgs"].to(old_model.device),
                    images=batch["images"].to(old_model.device),
                )
                logits_old = outputs_old.logits

            # 6) Slicing su token della risposta
            logprobs_resp = []
            old_logprobs_resp = []
            values_resp = []
            rewards_tokens = []

            for i in range(len(full_texts)):
                Lp = prompt_lengths[i]

                tokens_resp = enc_full["input_ids"][i, Lp:]
                if tokens_resp.numel() == 0:
                    continue

                logits_i = logits[i, Lp-1: Lp-1 + tokens_resp.shape[0], :]
                logits_old_i = logits_old[i, Lp-1: Lp-1 + tokens_resp.shape[0], :]
                values_i = values_all[i, Lp-1: Lp-1 + tokens_resp.shape[0]]

                lp = torch.log_softmax(logits_i, dim=-1)
                lp_old = torch.log_softmax(logits_old_i, dim=-1)

                lp_chosen = lp.gather(1, tokens_resp.unsqueeze(-1)).squeeze(-1)
                lp_old_chosen = lp_old.gather(1, tokens_resp.unsqueeze(-1)).squeeze(-1)

                logprobs_resp.append(lp_chosen)
                old_logprobs_resp.append(lp_old_chosen)
                values_resp.append(values_i)

                r_seq = rewards_seq[i]
                rewards_tokens.append(torch.full_like(values_i, r_seq))

            if len(logprobs_resp) == 0:
                continue

            logprobs = torch.cat(logprobs_resp)
            old_logprobs = torch.cat(old_logprobs_resp)
            values_flat = torch.cat(values_resp)
            rewards_flat = torch.cat(rewards_tokens)

            # 7) GAE
            advantages_list = []
            returns_list = []

            for vals_i, rews_i in zip(values_resp, rewards_tokens):
                adv_i, ret_i = compute_gae(rews_i, vals_i)
                advantages_list.append(adv_i)
                returns_list.append(ret_i)

            advantages = torch.cat(advantages_list)
            returns = torch.cat(returns_list)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 8) Loss PPO
            ratio = torch.exp(logprobs - old_logprobs)
            clip_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)

            policy_loss = -torch.min(
                ratio * advantages,
                clip_ratio * advantages
            ).mean()

            value_loss = torch.nn.functional.mse_loss(values_flat, returns)

            entropy = -(torch.exp(logprobs) * logprobs).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(
                f"Epoch {epoch+1} | "
                f"Loss: {loss.item():.4f} | "
                f"Policy: {policy_loss.item():.4f} | "
                f"Value: {value_loss.item():.4f} | "
                f"Entropy: {entropy.item():.4f} | "
                f"Reward mean: {rewards_seq.mean().item():.4f}"
            )

        old_model.load_state_dict(model.state_dict())
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        ckpt = Path(output_dir) / f"checkpoint-epoch-{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    print("\n🎉 RL multimodale con PPO completato!")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[1]

    train_rl_multimodal(
        model_path=str(BASE / "models" / "GEM_model"),
        rl_dataset_path=str(BASE / "RL" / "rl_dataset.json"),
        ptbxl_path=str(BASE / "data" / "ptbxl_subset.json"),
        image_folder=str(BASE / "data" / "ecg_images" / "gen_images"),
        ecg_folder=str(BASE / "data" / "ptbxl_subset"),
        output_dir=str(BASE / "RL" / "checkpoints"),
        lr=1e-5,
        batch_size=1,
        num_epochs=1,
    )

