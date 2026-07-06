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
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])

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
# CALCOLO ROBUSTO DEI TOTAL_BEATS
# ==========================================================

def compute_total_beats(ecg):
    """
    Determina il numero reale di battiti usando feature affidabili.
    Se non esistono, usa la feature beat-level più lunga.
    """

    candidate_features = [
        "R_PEAK_POSITIONS",
        "QRS_ONSETS",
        "QRS_OFFSETS",
        "P_ONSETS",
        "P_OFFSETS",
        "T_ONSETS",
        "T_OFFSETS"
    ]

    # 1) Cerca feature affidabili
    for lead, lead_data in ecg["Leads"].items():
        for feat in candidate_features:
            if feat in lead_data:
                lst = parse_list(lead_data[feat])
                if lst:
                    return len(lst)

    # 2) Se non trovi nulla, prendi la feature beat-level più lunga
    max_len = 0
    for lead, lead_data in ecg["Leads"].items():
        for feat, values in lead_data.items():
            lst = parse_list(values)
            if isinstance(lst, list):
                max_len = max(max_len, len(lst))

    return max_len if max_len > 0 else None


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
    results = {}

    for obs in obs_defs:
        if obs.get("evaluation_level") != "global":
            continue

        obs_id = obs["id"]
        feature = obs["feature"]
        operator = obs["operator"]
        value = obs["value"]

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
# AUDIT
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
# DIAGNOSIS ENGINE
# ==========================================================

def apply_diagnosis_reasoning(ecg, beat_obs, global_obs, obs_defs, diag_defs):
    diagnoses = []
    diagnosis_audit = {}

    obs_def_map = {obs["id"]: obs for obs in obs_defs}

    total_beats = compute_total_beats(ecg)
    if not total_beats:
        return diagnoses, diagnosis_audit

    for diag in diag_defs:

        # Le diagnosi "di esclusione" (es. NORM) non dipendono da osservazioni
        # ma dall'ASSENZA di altre diagnosi: si valutano DOPO il loop principale.
        if diag.get("logic") == "exclusion":
            continue

        diagnosis_audit[diag["diagnosis_name"]] = audit_diagnosis(
            diag, beat_obs, global_obs
        )

        req = diag.get("required_observations", [])
        opt = diag.get("optional_observations", [])
        req_ok = 0

        # ===== REQUIRED =====
        for obs_id in req:
            obs_def = obs_def_map.get(obs_id)
            if obs_def is None:
                continue

            level = obs_def.get("evaluation_level")

            if level == "global":
                if obs_id in global_obs:
                    req_ok += 1

            elif level == "beat":
                if obs_id in beat_obs:
                    # FIX: frazione valutata PER SINGOLO LEAD, non sommando i
                    # battiti-positivi su tutti i lead. Sommare tra lead e dividere
                    # per total_beats (battiti di UN solo lead) gonfiava la frazione
                    # ~12x (es. 56/11 = 5.1 vs soglia 0.5), facendo passare sempre
                    # la soglia. max() = miglior lead -> frazione reale in [0,1].
                    hit_count = max(
                        v["count"] for v in beat_obs[obs_id].values()
                    )
                    if evaluate_presence_semantics(
                        obs_def, hit_count, total_beats
                    ):
                        req_ok += 1

        required_semantics = diag.get("required_semantics", "all")

        if required_semantics == "all":
            if req_ok != len(req):
                continue

        elif required_semantics == "majority":
            min_fraction = diag.get("required_min_fraction", 0.5)
            if len(req) == 0:
                continue
            if (req_ok / len(req)) < min_fraction:
                continue

        # ===== OPTIONAL =====
        opt_ok = sum(
            1 for o in opt if o in beat_obs or o in global_obs
        )

        # ===== FILTRO =====
        if req_ok + opt_ok > 0:
            diagnoses.append({
                "diagnosis": diag["diagnosis_name"],
                "required_satisfied": req_ok,
                "required_total": len(req),
                "optional_satisfied": opt_ok
            })

    # ==========================================================
    # DIAGNOSI DI ESCLUSIONE (es. NORM = ECG del tutto normale)
    # ==========================================================
    # NORM scatta solo se sono presenti tutte le "required_diagnoses"
    # (tipicamente SINUS_RHYTHM) e NESSUN'ALTRA diagnosi patologica e' scattata.
    name_to_id = {d["diagnosis_name"]: d.get("diagnosis_id") for d in diag_defs}
    fired_ids = {name_to_id.get(x["diagnosis"]) for x in diagnoses}

    for diag in diag_defs:
        if diag.get("logic") != "exclusion":
            continue

        diagnosis_audit[diag["diagnosis_name"]] = {
            "required": {}, "optional": {}
        }

        required_diag = set(diag.get("required_diagnoses", []))
        # scatta se: tutte le required_diagnoses presenti E l'insieme delle
        # diagnosi gia' scattate e' ESATTAMENTE quelle richieste (niente patologie).
        if required_diag and fired_ids == required_diag:
            diagnoses.append({
                "diagnosis": diag["diagnosis_name"],
                "required_satisfied": len(required_diag),
                "required_total": len(required_diag),
                "optional_satisfied": 0
            })

    return diagnoses, diagnosis_audit

