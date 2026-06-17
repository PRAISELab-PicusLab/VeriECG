#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from rule_engine import (
    evaluate_observations_per_beat,
    evaluate_observations_global,
    apply_diagnosis_reasoning
)

RAW_ECG_PATH = "ecg_features_subset_prova.json"
OBS_DEF_PATH = "Clinical_Observation_Dictionary.json"
DIAG_DEF_PATH = "Diagnosis_Dictionary.json"

OUT_OBS = "output_observations_by_lead_subset_prova_finale.json"
OUT_DIAG = "output_diagnoses_subset_prova_finale.json"
OUT_ALIGNED = "rule_output_aligned.json"


# ==========================================================
# LOAD
# ==========================================================

def load(path):
    with open(path, "r") as f:
        return json.load(f)


# ==========================================================
# SCRITTURA OSSERVAZIONI
# ==========================================================

def write_observations(data, path):
    lines = ["{"]

    ecg_items = list(data.items())
    for i, (ecg_id, obs_dict) in enumerate(ecg_items):
        lines.append(f'  "{ecg_id}": {{')

        obs_items = list(obs_dict.items())
        for j, (obs_id, leads) in enumerate(obs_items):
            lines.append(f'    "{obs_id}": {{')

            lead_items = list(leads.items())
            for k, (lead, info) in enumerate(lead_items):
                beats_inline = ",".join(str(b) for b in info["beats"])
                comma = "," if k < len(lead_items) - 1 else ""
                lines.append(
                    f'      "{lead}": {{ "count": {info["count"]}, "beats": [{beats_inline}] }}{comma}'
                )

            comma_obs = "," if j < len(obs_items) - 1 else ""
            lines.append(f"    }}{comma_obs}")

        comma_ecg = "," if i < len(ecg_items) - 1 else ""
        lines.append(f"  }}{comma_ecg}")

    lines.append("}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ==========================================================
# AGGIUNTA REQUIRED/OPTIONAL SODDISFATTI
# ==========================================================

def attach_required_satisfied_to_each_diagnosis(diagnoses, beat_obs, global_obs, diag_defs):

    satisfied = set(beat_obs.keys()) | set(global_obs.keys())
    enriched = []

    for diag in diagnoses:


        diag_name = diag["diagnosis"]

        #  se non è stringa, converti
      #  if not isinstance(diag_name, str):
           # diag_name = str(diag_name)

        diag_def = next(
            d for d in diag_defs
            if d["diagnosis_name"].lower().strip() == diag_name.lower().strip()
        )

        required = diag_def.get("required_observations", [])
        optional = diag_def.get("optional_observations", [])

        required_satisfied = [obs for obs in required if obs in satisfied]
        optional_satisfied = [obs for obs in optional if obs in satisfied]

        enriched.append({
            **diag,
            "diagnosis": diag_name,
            "obs_satisfied": sorted(required_satisfied + optional_satisfied)
        })

    return enriched


# ==========================================================
# OUTPUT ALLINEATO PER RL
# ==========================================================

def normalize_diag(d):
    return d.upper().replace(" ", "_")


def export_aligned_rule_output(all_diag, output_path="rule_output_aligned.json"):
    final_output = []

    for ecg_id, data in all_diag.items():

        diag_block = data["diagnoses"][0]

        observations = diag_block["obs_satisfied"]

        diagnosis_name = diag_block["diagnosis"]
        diagnosis_id = normalize_diag(diagnosis_name)

        final_output.append({
            "id": ecg_id,
            "observations": observations,
            "diagnoses": [diagnosis_id]
        })

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print("📂 Creato file allineato:", output_path)


# ==========================================================
# MAIN
# ==========================================================

def main():
    ecgs = load(RAW_ECG_PATH)
    obs_defs = load(OBS_DEF_PATH)
    diag_defs = load(DIAG_DEF_PATH)

    all_obs = {}
    all_diag = {}

    for ecg_id, ecg in ecgs.items():

        beat_obs = evaluate_observations_per_beat(ecg, obs_defs)
        global_obs = evaluate_observations_global(ecg, obs_defs)

        diagnoses, diagnosis_audit = apply_diagnosis_reasoning(
            ecg, beat_obs, global_obs, obs_defs, diag_defs
        )

        all_obs[ecg_id] = beat_obs

        diagnoses_with_required = attach_required_satisfied_to_each_diagnosis(
            diagnoses, beat_obs, global_obs, diag_defs
        )

        all_diag[ecg_id] = {
            "diagnoses": diagnoses_with_required
        }

    # ================================
    # OUTPUT OSSERVAZIONI
    # ================================
    write_observations(all_obs, OUT_OBS)

    # ================================
    # OUTPUT DIAGNOSI
    # ================================
    with open(OUT_DIAG, "w") as f:
        json.dump(all_diag, f, indent=2)

    # ================================
    # OUTPUT ALLINEATO PER RL
    # ================================
    export_aligned_rule_output(all_diag, OUT_ALIGNED)

    print("✅ Rule engine completato correttamente")
    print("📂 Salvati:", OUT_OBS, OUT_DIAG, OUT_ALIGNED)


if __name__ == "__main__":
    main()
