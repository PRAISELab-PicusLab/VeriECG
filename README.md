## FILE DA AGGIORNARE

nella cartella update ci sono dei file da sostituire a quelli di GEM, ciascun file è messo nella cartella con il percorso all'interno di GEM:
- in update/models-GEM_model c'è il file config.json che va sostituito al file config.json all'interno della cartella models/GEM_model
- in update/llava-model c'è il file builder.py da sostituire al file builder.py all'interno della cartella GEM/llava/model
- in update/llava-model-multimodal_encoder ci sono i file builder.py e clip_encoder.py da sostituire ai rispettivi file all'interno di GEM/llava/model/multimodal_encoder


---

## FASE 1 — RULE ENGINE  (ambiente FeatureDB)

**INPUT:**
- `data/ptbxl_subset/` —  2041 ECG 
- `rule_engine/Clinical_Observation_Dictionary.json`, `rule_engine/Diagnosis_Dictionary.json`

**ESEGUIRE (in ordine):**
1. creare l'ambiente virtuale FeatureDB con `rule_engine/requirements_featuredb.txt`
2. `python rule_engine/estrazione_feature_deterministiche_da_segnale.py`
3. `python rule_engine/run_rule_engine.py`

**OUTPUT:**
- `rule_engine/features_di_ptbxl_subset.json` — feature deterministiche
- `rule_engine/output_rule_engine.json` — **ground truth** (osservazioni + diagnosi per ECG)

---

## FASE 2 — RL  (ambiente RL, GPU)

**INPUT:**
- `rule_engine/output_rule_engine.json` (dalla Fase 1)
- `GEM/` e `models/GEM_model/` (modello GEM)

**ESEGUIRE (in ordine):**
1. creare l'ambiente virtuale RL con `requirements.txt` (+ `torch` CUDA)
2. `python build_dataset_RL.py`
3. `python run_training.py`
   > Parametri del training in fondo a `run_training.py`: `lr`, `batch_size`, `num_epochs`, `kl_coef`.

**OUTPUT (in `output/run_XXX/`):**
- `referti_e_claim_*.txt` — referto GEM + claim estratti + claim rule engine + reward
- `training_log.json` — loss / KL / reward per step
- `confronto_baseline_vs_postRL.png`, `curve_training.png`
- `run_info.txt` — riepilogo iperparametri e risultati

---
