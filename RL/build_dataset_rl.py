#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# ===== PATH BASE =====
BASE = Path(__file__).resolve().parents[1]

GEM_STRUCTURED = BASE / "data" / "output_gem_structured.json" # DA MODIFICARE
RULE_OUTPUT    = BASE / "rule_engine" / "rule_output.json"
RL_DATASET     = BASE / "RL" / "rl_dataset.json"

# ===== LOAD =====
def load_json(path):
    with open(path) as f:
        return json.load(f)

gem_structured = load_json(GEM_STRUCTURED)
rule_output = load_json(RULE_OUTPUT)

# indicizzazione per id
rule_dict = {x["id"]: x for x in rule_output}

# ===== REWARD =====
def compute_reward(pred_obs, pred_dx, gt_obs, gt_dx, alpha=0.7, beta=0.3):

    pred_obs = set(pred_obs)
    gt_obs = set(gt_obs)

    TP = len(pred_obs & gt_obs)
    FP = len(pred_obs - gt_obs)
    FN = len(gt_obs - pred_obs)

    r_obs = (TP - FP - FN) / (len(gt_obs) + 1e-8)

    r_dx = 1 if set(pred_dx) == set(gt_dx) else -1

    return alpha * r_obs + beta * r_dx

# ===== COSTRUZIONE DATASET RL =====
rl_dataset = []

for item in gem_structured:

    ecg_id = item["id"]
    pred_obs = item.get("observations", [])
    pred_dx = item.get("diagnoses", [])

    gt = rule_dict.get(ecg_id)
    if not gt:
        continue

    gt_obs = gt.get("observations", [])
    gt_dx = gt.get("diagnoses", [])

    reward = compute_reward(pred_obs, pred_dx, gt_obs, gt_dx)

    rl_dataset.append({
        "id": ecg_id,
        "gem_observations": pred_obs,
        "gem_diagnoses": pred_dx,
        "rule_observations": gt_obs,
        "rule_diagnoses": gt_dx,
        "reward": reward
    })

# ===== SAVE =====
with open(RL_DATASET, "w") as f:
    json.dump(rl_dataset, f, indent=2)

print("Dataset RL creato!")
print("📂 Salvato in:", RL_DATASET)
