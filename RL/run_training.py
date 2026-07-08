# ============================================================
# SPLIT TRAIN/EVAL + ESECUZIONE TRAINING CON CONFRONTO BASELINE vs POST-RL
# ============================================================
import os
import json
import random
from pathlib import Path

# training_rl.py definisce train_rl_multimodal e, importandolo, tira dentro
# reward_definition.py (le due reward + reward_fn_training) e mapping.py
# (estrazione claim con l'LLM mapper -> CARICA il modello mapper)

from training_rl import train_rl_multimodal


BASE = Path(__file__).resolve().parent

random.seed(42)

rl_dataset_path = BASE / "RL" / "rl_ptbxl_dataset.json"
train_json_path = BASE / "RL" / "rl_ptbxl_dataset_train.json"
eval_json_path = BASE / "RL" / "rl_ptbxl_dataset_eval.json"

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


with open(train_json_path, "w") as f:
    json.dump(train_items, f)
with open(eval_json_path, "w") as f:
    json.dump(eval_items, f)

# Controllo rapido che il modello patchato ci sia davvero, prima di lanciare
# tutto (evita di scoprirlo a meta' caricamento)
model_path = str(BASE / "models" /"GEM_model")
assert Path(model_path).exists(), (
    f"Cartella modello non trovata: {model_path}. "
    "Controlla che il download (snapshot_download) e le patch a config.json "
    "siano stati fatti su questa stessa cartella."
)
print("File nel modello:", os.listdir(model_path))

# Cartella di output dedicata a QUESTA run: TIRO/output/run_XXX (id numerico
# progressivo), cosi' ogni prova ha referti/claim/grafici separati.
output_root = BASE / "output"
output_root.mkdir(parents=True, exist_ok=True)
_existing = [int(p.name.split("_")[-1]) for p in output_root.glob("run_*")
             if p.name.split("_")[-1].isdigit()]
run_id = max(_existing, default=0) + 1
run_dir = output_root / f"run_{run_id:03d}"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Output di questa run in: {run_dir}")

results = train_rl_multimodal(
    model_path=model_path,
    train_rl_json_path=str(train_json_path),
    eval_rl_json_path=str(eval_json_path),
    ptbxl_path=str(BASE / "data" / "ptbxl_subset.json"),
    image_folder=str(BASE / "data" / "ecg_images" / "gen_images"),
    ecg_folder=str(BASE / "data" / "ptbxl_subset"),  # FIX: era "ecg_timeseries/ptbxl_subset", cartella mai creata (il subset lo crea la cella PTBXL_SUBSET dentro data/ptbxl_subset)
    output_dir=str(run_dir),  # tutta la diagnostica/grafici/checkpoint di questa run vanno qui
    lr=5e-5,
    batch_size=6,   # >1: la normalizzazione degli advantage confronta piu report (evita il caso degenere batch_size=1)
    num_epochs=3,
    kl_coef=0.3,
)


# ============================================================
# CONFRONTO BASELINE vs POST-RL
# (spostato qui da training_rl.py: usa 'results', prodotto dalla chiamata sopra.
#  train_rl_multimodal salva gia' i grafici internamente in run_dir; questo
#  blocco stampa la tabella di confronto e ri-salva il grafico riassuntivo.)
# ============================================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend non-interattivo: salva su file, niente finestra (fuori da Colab)
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

out_path = Path(results["run_dir"]) / "confronto_baseline_vs_postRL.png"
plt.savefig(out_path, dpi=150)

print(f"\nGrafico salvato in: {out_path}")
print(
    f"Variazione reward media: {comparison_df.loc['post-RL', 'delta_reward']:+.3f} "
    f"({baseline['reward_mean']:.3f} -> {post_rl['reward_mean']:.3f})"
)
print(
    f"Variazione F1 media:     {comparison_df.loc['post-RL', 'delta_f1']:+.3f} "
    f"({baseline['f1_mean']:.3f} -> {post_rl['f1_mean']:.3f})"
)
