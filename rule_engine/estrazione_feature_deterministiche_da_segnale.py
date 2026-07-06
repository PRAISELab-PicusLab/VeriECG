import sys
import os
import json
import wfdb
import numpy as np
import math
from pathlib import Path
import traceback


# ==========================================================
# === PATH BASE DELLA REPOSITORY
# ==========================================================
# rule_engine/ sta dentro TIRO/, quindi parents[1] = TIRO. (Era parents[2] =
# Tirocinio, fuori da TIRO: incoerente con rule_engine/FeatureDB e data/ptbxl_subset,
# che sono relativi a TIRO.)
BASE = Path(__file__).resolve().parents[1]

# ==========================================================
# === PATH FEATUREDB
# ==========================================================
sys.path.append(str(BASE / "rule_engine" /"FeatureDB"))




# ==========================================================
# === IMPORT REALI FEATUREDB
# ==========================================================
from tools.SimpleFilter import smooth_avg1
from tools.NormalizeTools import normalize_sig_hist
from tools.QRSDetector import simple_qrs_detector
from tools.SingleBeatBounds import dwt_ecg_delineator
from features.ECGAvgBeat import extract_features

# ==========================================================
# === UTILS
# ==========================================================

GLOBAL_RR_MS = []

def clean(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def gem_list(values, ndigits=3):
    if len(values) == 0:
        return "[]"

    out = []
    for v in values:
        if v is None:
            out.append("None")                    # era "null": ast.literal_eval (usato dal rule engine) vuole None, non null
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out.append(f"{round(float(v), ndigits)}")
        else:
            out.append(repr(str(v)))              # stringhe QUOTATE (es. 'declination'): erano non quotate e illeggibili

    return "[" + ", ".join(out) + "]"

# ==========================================================
# === CALIBRAZIONE (lasciata invariata)
# ==========================================================
def clinical_adjustments(feats, bounds, beat, fs):
   # AMP_SCALE = 0.45
   # DUR_SCALE = 1.35
    QT_SHIFT_MS = 12
    '''
    feats["P"]["amplitude"] *= AMP_SCALE
    feats["QRS"]["QRSAmplitude"] *= AMP_SCALE
    feats["T"]["amplitude"] *= AMP_SCALE
    '''

    feats["P"]["duration"] = int(feats["P"]["duration"]) # * DUR_SCALE
    feats["QRS"]["QRSDuration"] = int(feats["QRS"]["QRSDuration"]) # * DUR_SCALE
    feats["T"]["duration"] = int(feats["T"]["duration"]) # * DUR_SCALE

    if bounds["ECG_T_Offset"] is not None and bounds["ECG_R_Onset"] is not None:
        corrected_t_offset = bounds["ECG_T_Offset"] + int(QT_SHIFT_MS * fs / 1000)
        feats["QT"] = int((corrected_t_offset - bounds["ECG_R_Onset"]) * 1000 / fs)
    else:
        feats["QT"] = None

    return feats

# ==========================================================
# === RICALCOLO CORRETTO DEL SEGMENTO ST (Fix 2)
# ==========================================================
def compute_st_from_baseline(beat, bounds, fs):
    """
    Ricalcolo corretto di ST Level / ST Duration / ST Form.

    Il calcolo di FeatureDB (_pr_or_st) misura l'ampiezza ST rispetto al PUNTO J
    (fine QRS), non rispetto alla linea isoelettrica, e prende l'estremo su tutta
    la finestra fino a inizio-T: cattura la pendenza ST-T, risulta sistematicamente
    negativa ("declination") su ~meta' dei battiti e produce outlier (fino a -32)
    sui lead a basso voltaggio (per via di normalize_sig_hist).

    Qui l'ST Level e' la deviazione del punto J (mediana su una finestra ~40 ms dopo J)
    rispetto al baseline isoelettrico stimato sul segmento PR (P_offset -> QRS_onset).
    Ritorna (st_level, st_duration_ms, st_form).
    """
    n = len(beat)

    def ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    j    = bounds.get("ECG_R_Offset")   # punto J = fine QRS
    qon  = bounds.get("ECG_R_Onset")    # inizio QRS
    poff = bounds.get("ECG_P_Offset")
    ton  = bounds.get("ECG_T_Onset")

    if not ok(j) or not ok(qon):
        return None, None, None
    j = int(j); qon = int(qon)
    if not (0 <= qon < j < n):
        return None, None, None

    # baseline isoelettrico: mediana del segmento PR; fallback: ~40 ms prima del QRS
    if ok(poff) and 0 <= int(poff) < qon:
        seg = beat[int(poff):qon]
    else:
        w = max(1, int(0.04 * fs))
        seg = beat[max(0, qon - w):qon]
    if len(seg) == 0:
        return None, None, None
    baseline = float(np.median(seg))

    # ST Level: mediana in una piccola finestra (~40 ms) subito dopo il punto J
    w2 = max(1, int(0.04 * fs))
    hi = min(j + w2, n)
    if hi <= j:
        return None, None, None
    st_level = float(np.median(beat[j:hi])) - baseline

    # Durata ST (J -> inizio T) e forma dal dislivello lungo il tratto ST
    if ok(ton) and j < int(ton) < n:
        te = int(ton)
        st_dur = int((te - j) * 1000 / fs)
        slope = (float(beat[te]) - baseline) - st_level
    else:
        st_dur = None
        slope = 0.0

    DEAD = 0.05
    if abs(slope) < DEAD:
        form = "horizontal"
    elif slope < 0:
        form = "declination"
    else:
        form = "upslope"

    return round(st_level, 3), st_dur, form


# ==========================================================
# === FILTRO BEAT MINIMALE (TECNICO)
# ==========================================================
def is_valid_beat_realistic(feats, bounds, rr_ms, fs):
    # >>> MODIFICA
    # SOLO controllo tecnico: il beat deve essere misurabile
    if bounds["ECG_R_Onset"] is None or bounds["ECG_R_Offset"] is None:
        return False
    if np.isnan(bounds["ECG_R_Onset"]) or np.isnan(bounds["ECG_R_Offset"]):
        return False
    if rr_ms < 200 or rr_ms > 3000:
        return False
    qrs_dur = feats["QRS"]["QRSDuration"]
    if qrs_dur is None or np.isnan(qrs_dur):
        return False
    return True

# ==========================================================
# === SEGMENTAZIONE BATTI (OK)
# ==========================================================
def extract_single_beats(ecg, ecg_mv, rpeaks, fs):

    beats = []
    win_pre  = int(0.3 * fs)
    win_post = int(0.5 * fs)

    for i in range(1, len(rpeaks)-1):
        start = rpeaks[i] - win_pre
        end   = rpeaks[i] + win_post
        if start < 0 or end > len(ecg):
            continue

        beat = ecg[start:end]          # normalizzato: per delineazione
        beat_mv = ecg_mv[start:end]    # mV grezzo: per misura ampiezze (Fix 5)
        r_local = rpeaks[i] - start
        rr_ms = (rpeaks[i] - rpeaks[i-1]) * 1000 / fs

        beats.append((beat, beat_mv, r_local, rr_ms))
        GLOBAL_RR_MS.append(rr_ms)

    return beats

# ==========================================================
# === PROCESSO DI UN LEAD
# ==========================================================
def process_lead(ecg, fs):

    # Fix 5: segnale in mV grezzo (solo smoothing, NIENTE normalize_sig_hist)
    # per misurare le ampiezze nella scala fisiologica corretta. wfdb PTB-XL
    # e' gia' in mV. La delineazione resta sul segnale normalizzato (le sue
    # soglie interne sono tarate su quella scala).
    ecg_mv = smooth_avg1(ecg.copy(), radius=3)

    ecg = ecg * 1000.0
    ecg = smooth_avg1(ecg, radius=3)
    ecg = normalize_sig_hist(ecg)

    rpeaks = simple_qrs_detector(ecg, fs)
    if len(rpeaks) < 5:
        return None

    beats = extract_single_beats(ecg, ecg_mv, rpeaks, fs)

    accepted = 0
    total = len(beats)

    out = {
        "P Amplitude (mV)": [],
        "P Duration (ms)": [],
        "PR Interval (ms)": [],
        "QRS Amplitude (mV)": [],
        "QRS Duration (ms)": [],
        "Q Amplitude (mV)": [],
        "Q Duration (ms)": [],
        "R Amplitude (mV)": [],
        "S Amplitude (mV)": [],
        "S Duration (ms)": [],
        "R' Amplitude (mV)": [],
        "R Wave Peak Time (ms)": [],
        "R'/R Ratio": [],
        "T Amplitude (mV)": [],
        "T Duration (ms)": [],
        "ST Duration (ms)": [],
        "ST Level (mV)": [],
        "ST Form": [],
        "QT Interval (ms)": [],
        "QTc Interval (ms)": []
    }

    for beat, beat_mv, r_local, rr_ms in beats:
        try:
            bounds = dwt_ecg_delineator(beat, r_local, fs)   # delineazione sul normalizzato

            # ================================
            # === R WAVE PEAK TIME (RWPT)
            # ================================
            rwpt = None

            qrs_start = bounds.get("ECG_R_Onset", None)

            if qrs_start is not None and not np.isnan(qrs_start):
                rwpt = (r_local - qrs_start) * 1000 / fs

            # Fix 5: ampiezze misurate sul battito in mV (non sul normalizzato)
            feats = extract_features(
                beat_mv,
                bounds["ECG_P_Onset"],
                bounds["ECG_P_Offset"],
                bounds["ECG_R_Onset"],
                bounds["ECG_R_Offset"],
                bounds["ECG_T_Onset"],
                bounds["ECG_T_Offset"],
                fs,
                rr_ms
            )

            feats = clinical_adjustments(feats, bounds, beat_mv, fs)

            if not is_valid_beat_realistic(feats, bounds, rr_ms, fs):
                continue

            accepted += 1
            qrs = feats["QRS"]

            # ================================
            # === R PRIME / R RATIO
            # ================================
            
            r_prime_over_r = None
            r_amp = qrs.get("RAmplitude", None)      # R
            r_prime_amp = qrs.get("R'Amplitude", None)  # R'
            
            if r_amp is not None and r_prime_amp is not None:
                if not np.isnan(r_amp) and not np.isnan(r_prime_amp):
                    if r_amp > 0:
                        r_prime_over_r = r_prime_amp / r_amp


            # >>> MODIFICA
            # Append SEMPRE (None se non esiste)
            out["P Amplitude (mV)"].append(clean(feats["P"]["amplitude"]))
            out["P Duration (ms)"].append(clean(feats["P"]["duration"]))
            out["PR Interval (ms)"].append(clean(feats["PR"].get("interval")))

            out["QRS Amplitude (mV)"].append(clean(qrs.get("QRSAmplitude")))
            out["QRS Duration (ms)"].append(clean(qrs.get("QRSDuration")))
            out["Q Amplitude (mV)"].append(clean(qrs.get("QAmplitude")))
            out["Q Duration (ms)"].append(clean(qrs.get("QDuration")))
            out["R Amplitude (mV)"].append(clean(qrs.get("RAmplitude")))
            out["S Amplitude (mV)"].append(clean(qrs.get("SAmplitude")))
            out["S Duration (ms)"].append(clean(qrs.get("SDuration")))
            out["R' Amplitude (mV)"].append(clean(qrs.get("R'Amplitude")))
            out["R Wave Peak Time (ms)"].append(clean(rwpt))
            out["R'/R Ratio"].append(clean(r_prime_over_r))

            # Fix 6: T Amplitude FIRMATA (negativa se onda T invertita) -> serve a
            # OBS_T_WAVE_INVERSION (< 0), criterio chiave dell'ischemia. FeatureDB
            # restituisce la magnitudine + il campo 'direction' ('+'/'-').
            t_amp = feats["T"].get("amplitude")
            if t_amp is not None and not (isinstance(t_amp, float) and np.isnan(t_amp)):
                t_amp = -abs(t_amp) if feats["T"].get("direction") == "-" else abs(t_amp)
            out["T Amplitude (mV)"].append(clean(t_amp))
            out["T Duration (ms)"].append(clean(feats["T"]["duration"]))

            # Fix 2: ST ricalcolato rispetto al baseline isoelettrico
            # (feats["ST"] di FeatureDB e' misurato rispetto al punto J -> inaffidabile)
            st_level, st_dur, st_form = compute_st_from_baseline(beat_mv, bounds, fs)
            out["ST Duration (ms)"].append(clean(st_dur))
            out["ST Level (mV)"].append(clean(st_level))
            out["ST Form"].append(st_form)

            qt = feats.get("QT")
            out["QT Interval (ms)"].append(clean(qt))

            if qt is not None:
                out["QTc Interval (ms)"].append(int(qt / math.sqrt(rr_ms / 1000)))
            else:
                out["QTc Interval (ms)"].append(None)

        except Exception:
            traceback.print_exc()
            continue

    print(f"Accepted beats: {accepted}/{total}")

    for k in out:
        out[k] = gem_list(out[k])

    return out

# ==========================================================
# === MAIN
# ==========================================================
ECG_DIR = BASE / "data" / "ptbxl_subset"
# Scrive direttamente dove run_rule_engine.py legge (TIRO/rule_engine/), evitando la copia manuale.
OUT_JSON = Path(__file__).resolve().parent / "features_di_ptbxl_subset.json"

LEAD_NAMES = [
    "Lead I","Lead II","Lead III",
    "Lead aVR","Lead aVL","Lead aVF",
    "Lead V1","Lead V2","Lead V3",
    "Lead V4","Lead V5","Lead V6"
]

def main():
    results = {}

    for file in os.listdir(ECG_DIR):
        if not file.endswith(".hea"):
            continue

        record_name = file.replace(".hea", "")
        record_path = ECG_DIR / record_name

        print(f"\nProcessing ECG: {record_name}")
        GLOBAL_RR_MS.clear()

        rec = wfdb.rdrecord(str(record_path))
        ecg = rec.p_signal
        fs = rec.fs

        ecg_result = {}
        for i in range(12):
            ecg_result[LEAD_NAMES[i]] = process_lead(ecg[:, i], fs)

        rr_valid = [rr for rr in GLOBAL_RR_MS if 200 <= rr <= 3000]
        global_hr = round(60000 / (sum(rr_valid)/len(rr_valid)), 1) if rr_valid else None

        
        # ================================
        # === RR INTERVAL MEDIO GLOBALE
        # ================================
        if rr_valid:
          rr_mean = round(sum(rr_valid) / len(rr_valid), 1)
        else:
          rr_mean = None



        results[record_name] = {
            "Global Heart Rate (bpm)": global_hr,
            "RR Interval Mean (ms)": rr_mean,
            "Leads": ecg_result
        }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)


    print("\n✅ FINITO")
    print("📂 Salvato in:", OUT_JSON)

if __name__ == "__main__":
    main()