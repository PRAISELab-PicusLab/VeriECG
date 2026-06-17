import ast
import statistics

# ==========================================================
# UTILS
# ==========================================================

def parse_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def parse_range(value):
    # ✅ Caso nuovo: value è una lista [min, max]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])

    # ✅ Caso legacy: value è stringa "20-40"
    if isinstance(value, str):
        value = value.strip().replace("â€“", "-").replace("–", "-")
        low, high = value.split("-")
        return float(low), float(high)

    raise ValueError(f"Invalid range value: {value}")



def operator_match(x, operator, threshold):
    if x is None:
        return False

    try:
        if operator == "==":
            return str(x) == str(threshold)

        x = float(x)

        if operator == ">":
            return x > threshold
        if operator == "<":
            return x < threshold
        if operator == ">=":
            return x >= threshold
        if operator == "<=":
            return x <= threshold
        if operator == "between":
            lo, hi = threshold
            return lo <= x <= hi

    except Exception:
        return False

    return False


# ==========================================================
# SEMANTICA CLINICA
# ==========================================================

def evaluate_presence_semantics(obs_def, hit_count, total_beats):
    semantics = obs_def.get("presence_semantics", "exists")

    if semantics == "exists":
        return hit_count >= 1

    if semantics == "all":
        return hit_count == total_beats

    if semantics == "fraction":
        min_fraction = obs_def.get("min_fraction", 0.5)
        return (hit_count / total_beats) >= min_fraction

    return False


# ==========================================================
# BEAT-LEVEL OBSERVATIONS
# ==========================================================

def evaluate_observations_per_beat(ecg, obs_defs):
    results = {}

    for obs in obs_defs:
        if obs.get("evaluation_level") != "beat":
            continue

        obs_id = obs["id"]
        feature = obs["feature"]
        operator = obs["operator"]
        value = obs["value"]
        lead_constraint = obs.get("lead", "")

        if operator == "between":
            threshold = parse_range(value)
        elif operator in [">", "<", ">=", "<="]:
            threshold = float(value)
        elif operator == "==":
            threshold = value
        else:
            continue

        obs_hits = {}

        for lead, lead_data in ecg["Leads"].items():
            if lead_constraint not in ["", "All", "Any"] and lead not in lead_constraint:
                continue
            if feature not in lead_data:
                continue

            values = parse_list(lead_data[feature])
            if not values:
                continue

            hits = [i for i, v in enumerate(values) if operator_match(v, operator, threshold)]

            if hits:
                obs_hits[lead] = {"count": len(hits), "beats": hits}

        if obs_hits:
            results[obs_id] = obs_hits

    return results


# ==========================================================
# GLOBAL OBSERVATIONS
# ==========================================================

def evaluate_observations_global(ecg, obs_defs):
    """
    Valuta le osservazioni GLOBALI (scalari ECG-level),
    come Heart Rate, RR mean, QTc, ecc.
    """
    results = {}

    for obs in obs_defs:
        if obs.get("evaluation_level") != "global":
            continue

        obs_id = obs["id"]
        feature = obs["feature"]
        operator = obs["operator"]
        value = obs["value"]

        # ✅ la feature deve essere direttamente in ecg
        if feature not in ecg:
            continue

        x = ecg[feature]

        try:
            if operator == "between":
                lo, hi = parse_range(value)
                ok = lo <= x <= hi
            elif operator in [">", "<", ">=", "<="]:
                ok = operator_match(x, operator, float(value))
            elif operator == "==":
                ok = operator_match(x, operator, value)
            else:
                ok = False
        except Exception:
            ok = False

        if ok:
            results[obs_id] = True

    return results


# ==========================================================
# AUDIT LIVELLO 2 — DIAGNOSI
# ==========================================================

def audit_diagnosis(diag, beat_obs, global_obs):
    audit = {"required": {}, "optional": {}}

    for obs in diag.get("required_observations", []):
        if obs in beat_obs:
            audit["required"][obs] = "TRUE (beat)"
        elif obs in global_obs:
            audit["required"][obs] = "TRUE (global)"
        else:
            audit["required"][obs] = "FALSE"

    for obs in diag.get("optional_observations", []):
        if obs in beat_obs:
            audit["optional"][obs] = "TRUE (beat)"
        elif obs in global_obs:
            audit["optional"][obs] = "TRUE (global)"
        else:
            audit["optional"][obs] = "FALSE"

    return audit


# ==========================================================
# DIAGNOSIS ENGINE FINALE
# ==========================================================

def apply_diagnosis_reasoning(ecg, beat_obs, global_obs, obs_defs, diag_defs):
    diagnoses = []
    diagnosis_audit = {}

    # Mappa OBS_ID -> definizione OBS
    obs_def_map = {obs["id"]: obs for obs in obs_defs}

    # Determina il numero totale di battiti (da qualunque lead beat-level)
    total_beats = None
    for lead_data in ecg["Leads"].values():
        for v in lead_data.values():
            lst = parse_list(v)
            if lst:
                total_beats = len(lst)
                break
        if total_beats:
            break

    if not total_beats:
        return diagnoses, diagnosis_audit

    # Loop su tutte le diagnosi
    for diag in diag_defs:

        diagnosis_audit[diag["diagnosis_name"]] = audit_diagnosis(
            diag, beat_obs, global_obs
        )

        req = diag.get("required_observations", [])
        opt = diag.get("optional_observations", [])
        req_ok = 0

        # ===== VALUTAZIONE DELLE REQUIRED OBS =====
        for obs_id in req:
            obs_def = obs_def_map.get(obs_id)
            if obs_def is None:
                continue

            level = obs_def.get("evaluation_level")

            # --- OBS global ---
            if level == "global":
                if obs_id in global_obs:
                    req_ok += 1

            # --- OBS beat-level ---
            elif level == "beat":
                if obs_id in beat_obs:
                    hit_count = sum(
                        v["count"] for v in beat_obs[obs_id].values()
                    )
                    if evaluate_presence_semantics(
                        obs_def, hit_count, total_beats
                    ):
                        req_ok += 1

        # ===== SEMANTICA DELLE REQUIRED (STRICT vs MAJORITY) =====
        required_semantics = diag.get("required_semantics", "all")
        print(
        f"[DEBUG] {diag['diagnosis_name']} | "
        f"semantics={required_semantics} | "
        f"req_ok={req_ok}/{len(req)}"
        )

        if required_semantics == "all":
            # comportamento precedente (strict)
            if req_ok != len(req):
                continue

        elif required_semantics == "majority":
            # nuova semantica majority
            min_fraction = diag.get("required_min_fraction", 0.5)

            if len(req) == 0:
                continue  # sicurezza

            if (req_ok / len(req)) < min_fraction:
                continue

        # ===== VALUTAZIONE OPTIONAL OBS =====
        opt_ok = sum(
            1 for o in opt if o in beat_obs or o in global_obs
        )

        # ===== FILTRO FINALE =====
        # Mostra la diagnosi solo se almeno una OBS è vera
        if req_ok + opt_ok > 0:
            diagnoses.append({
                "diagnosis": diag["diagnosis_name"],
                "required_satisfied": req_ok,
                "required_total": len(req),
                "optional_satisfied": opt_ok
            })

    return diagnoses, diagnosis_audit