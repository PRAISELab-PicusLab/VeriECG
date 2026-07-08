# ============================================================
# COSTRUZIONE DEL DATASET RL  (rl_dataset)
# ============================================================
# Cosa fa: per ogni ECG accoppia il PROMPT di generazione con la GROUND TRUTH
# del rule engine (osservazioni + diagnosi). E' il file che alimenta il PPO:
# durante il training GEM genera un report, l'LLM mapper ne estrae i claim, e la reward
# li confronta con rule_observations/rule_diagnoses di questo file.
#
# INPUT:
#   1) rule_engine/output_rule_engine.json = GROUND TRUTH.
#      Prodotto da run_rule_engine.py: per ogni ECG, osservazioni e diagnosi
#      rilevate dal rule engine. E' MULTI-RIGA per id (una riga per diagnosi),
#      quindi va AGGREGATO per id (unione), altrimenti si perdono diagnosi/oss.
#   2) data/ptbxl_subset.json = INDICE DEI DATI.
#      Lista di {id, ecg, image, ...}: dice dove sono segnale e immagine di ogni
#      ECG. Serve per NON inserire nel dataset id di cui mancano i file (il
#      DataLoader andrebbe in KeyError). Si tengono solo gli id presenti qui.
#
# OUTPUT:
#   RL/<nome>.json: lista di {id, prompt, rule_observations, rule_diagnoses}.
import json
from pathlib import Path
from collections import Counter

# Percorsi RELATIVI alla posizione dello script: scripts/ sta dentro TIRO/,
# quindi BASE = cartella TIRO (due livelli sopra questo file). Niente Drive,
# niente variabili d'ambiente.
BASE = Path(__file__).resolve().parent

RULE_OUTPUT_PATH = BASE / "rule_engine" / "output_rule_engine.json"
PTBXL_SUBSET_PATH = BASE / "data" / "ptbxl_subset.json"
# Output generato AUTOMATICAMENTE come rl_<nome_file_input>:
# es. ptbxl_subset.json -> rl_ptbxl_subset.json
RL_OUTPUT_PATH = BASE / "RL" / f"rl_{PTBXL_SUBSET_PATH.name}"

# Stesso PROMPT di ptbxl_subset.json, cosi' il campo "prompt" e' coerente con
# quello che il modello si aspetta in generazione.
PROMPT = (
    "Interpret the provided ECG image, identify key features and "
    "abnormalities in each lead, and generate a clinical diagnosis that is "
    "supported by the observed evidence"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


rule_output = load_json(RULE_OUTPUT_PATH)
valid_ids = {x["id"] for x in load_json(PTBXL_SUBSET_PATH)}

# ---- AGGREGAZIONE ground truth per id (FIX multi-riga per diagnosi) ----
rule_gt = {}
for row in rule_output:
    agg = rule_gt.setdefault(row["id"], {"observations": set(), "diagnoses": set()})
    agg["observations"].update(row.get("observations", []))
    agg["diagnoses"].update(row.get("diagnoses", []))

# ============================================================
# COSTRUZIONE
# ============================================================
rl_dataset = []
skipped = 0
for rid, agg in rule_gt.items():
    if rid not in valid_ids:
        skipped += 1          # niente ecg/immagine nell'indice -> salta
        continue

    rl_dataset.append({
        "id": rid,
        "prompt": f"<image>\n{PROMPT}",
        "rule_observations": sorted(agg["observations"]),
        "rule_diagnoses": sorted(agg["diagnoses"]),
    })

# ============================================================
# SALVATAGGIO
# NB: run_training.py legge il dataset RL da RL/rl_dataset.json e poi lo splitta
# in train/eval. Fai puntare quel percorso a QUESTO file (o rinominalo).
# ============================================================
RL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(rl_dataset, f, indent=2, ensure_ascii=False)

print(f"Dataset RL creato: {len(rl_dataset)} esempi -> {RL_OUTPUT_PATH}")
if skipped:
    print(f"  ({skipped} id del rule engine saltati: assenti in ptbxl_subset.json)")
print("  diagnosi nella ground truth:",
      dict(Counter(d for x in rl_dataset for d in x["rule_diagnoses"])))
