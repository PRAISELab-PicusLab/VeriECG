import json
from pathlib import Path
from rule_engine import (
    evaluate_observations_per_beat,
    evaluate_observations_global,
    apply_diagnosis_reasoning
)


BASE = Path(__file__).resolve().parent

RAW_ECG_PATH = BASE / "features_di_ptbxl_subset.json"
OBS_DEF_PATH = BASE / "Clinical_Observation_Dictionary.json"
DIAG_DEF_PATH = BASE / "Diagnosis_Dictionary.json"

OUT_OBS = BASE / "output_observations.json"
OUT_DIAG = BASE / "output_diagnoses.json"
OUT_ALIGNED = BASE / "output_rule_engine.json"


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

    # osservazioni effettivamente soddisfatte nell'ECG
    satisfied = set(beat_obs.keys()) | set(global_obs.keys())
    enriched = []

    for diag in diagnoses:

        diag_name = diag["diagnosis"].strip().lower()

        # trova la definizione della diagnosi
        diag_def = next(
            d for d in diag_defs
            if d["diagnosis_name"].strip().lower() == diag_name
        )

        required = diag_def.get("required_observations", [])
        optional = diag_def.get("optional_observations", [])

        # FILTRATE per diagnosi
        required_satisfied = [obs for obs in required if obs in satisfied]
        optional_satisfied = [obs for obs in optional if obs in satisfied]

        enriched.append({
            **diag,
            "diagnosis": diag_def["diagnosis_name"],
            "diagnosis_id": diag_def["diagnosis_id"],
            "obs_satisfied": sorted(required_satisfied + optional_satisfied)
        })

    return enriched


# ==========================================================
# OUTPUT ALLINEATO PER RL
# ==========================================================

def export_aligned_rule_output(all_diag, output_path="rule_output_aligned.json"):
    final_output = []

    for ecg_id, data in all_diag.items():

        for diag_block in data["diagnoses"]:

            # Usa il diagnosis_id CANONICO (non piu' normalize_diag(nome)): e' la
            # stessa convenzione di DIAG_LIST.json che vede il mapper, cosi' la reward
            # confronta gli stessi identificatori. NB: mapping/genera_liste_da_rule_engine.py
            # produce DIAG_LIST.json con questi diagnosis_id.
            diagnosis_id = diag_block["diagnosis_id"]

            final_output.append({
                "id": ecg_id,
                "observations": diag_block["obs_satisfied"],
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
