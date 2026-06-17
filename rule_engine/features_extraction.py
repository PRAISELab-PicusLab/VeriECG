#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import wfdb
import numpy as np
import math
from pathlib import Path

# ==========================================================
# === PATH BASE DELLA REPOSITORY
# ==========================================================
BASE = Path(__file__).resolve().parents[1]

# ==========================================================
# === PATH FEATUREDB
# ==========================================================
sys.path.append(str(BASE / "rule_engine" / "FeatureDB"))

# ==========================================================
# === IMPORT FEATUREDB
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
            out.append("null")
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out.append(f"{round(float(v), ndigits)}")
        else:
            out.append(str(v))

    return "[" + ", ".join(out) + "]"

# ==========================================================
# === CALIBRAZIONE
# ==========================================================
def clinical_adjustments(feats, bounds, beat, fs):
    QT_SHIFT_MS = 12

    feats["P"]["duration"] = int(feats["P"]["duration"])
    feats["QRS"]["QRSDuration"] = int(feats["QRS"]["QRSDuration"])
    feats["T"]["duration"] = int(feats["T"]["duration"])

    if bounds["ECG_T_Offset"] is not None and bounds["ECG_R_Onset"] is not None:
        corrected_t_offset = bounds["ECG_T_Offset"] + int(QT_SHIFT_MS * fs / 1000)
        feats["QT"] = int((corrected_t_offset - bounds["ECG_R_Onset"]) * 1000 / fs)
    else:
        feats["QT"] = None

    return feats

# ==========================================================
# === FILTRO BEAT
# ==========================================================
def is_valid_beat_realistic(feats, bounds, rr_ms, fs):
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
# === SEGMENTAZIONE BATTITI
# ==========================================================
def extract_single_beats(ecg, rpeaks, fs):

    beats = []
    win_pre  = int(0.3 * fs)
    win_post = int(0.5 * fs)

    for i in range(1, len(rpeaks)-1):
        start = rpeaks[i] - win_pre
        end   = rpeaks[i] + win_post
        if start < 0 or end > len(ecg):
            continue

        beat = ecg[start:end]
        r_local = rpeaks[i] - start
        rr_ms = (rpeaks[i] - rpeaks[i-1]) * 1000 / fs

        beats.append((beat, r_local, rr_ms))
        GLOBAL_RR_MS.append(rr_ms)

    return beats

# ==========================================================
# === PROCESSO DI UN LEAD
# ==========================================================
def process_lead(ecg, fs):

    ecg = ecg * 1000.0
    ecg = smooth_avg1(ecg, radius=3)
    ecg = normalize_sig_hist(ecg)

    rpeaks = simple_qrs_detector(ecg, fs)
    if len(rpeaks) < 5:
        return None

    beats = extract_single_beats(ecg, rpeaks, fs)

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

    for beat, r_local, rr_ms in beats:
        try:
            bounds = dwt_ecg_delineator(beat, r_local, fs)

            rwpt = None
            qrs_start = bounds.get("ECG_R_Onset", None)
            if qrs_start is not None and not np.isnan(qrs_start):
                rwpt = (r_local - qrs_start) * 1000 / fs

            feats = extract_features(
                beat,
                bounds["ECG_P_Onset"],
                bounds["ECG_P_Offset"],
                bounds["ECG_R_Onset"],
                bounds["ECG_R_Offset"],
                bounds["ECG_T_Onset"],
                bounds["ECG_T_Offset"],
                fs,
                rr_ms
            )

            feats = clinical_adjustments(feats, bounds, beat, fs)

            if not is_valid_beat_realistic(feats, bounds, rr_ms, fs):
                continue

            accepted += 1
            qrs = feats["QRS"]

            r_prime_over_r = None
            r_amp = qrs.get("RAmplitude", None)
            r_prime_amp = qrs.get("R'Amplitude", None)
            if r_amp is not None and r_prime_amp is not None:
                if not np.isnan(r_amp) and not np.isnan(r_prime_amp):
                    if r_amp > 0:
                        r_prime_over_r = r_prime_amp / r_amp

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

            out["T Amplitude (mV)"].append(clean(feats["T"]["amplitude"]))
            out["T Duration (ms)"].append(clean(feats["T"]["duration"]))

            out["ST Duration (ms)"].append(clean(feats["ST"]["duration"]))
            out["ST Level (mV)"].append(clean(feats["ST"].get("amplitude")))
            out["ST Form"].append(feats["ST"]["form"])

            qt = feats.get("QT")
            out["QT Interval (ms)"].append(clean(qt))

            if qt is not None:
                out["QTc Interval (ms)"].append(int(qt / math.sqrt(rr_ms / 1000)))
            else:
                out["QTc Interval (ms)"].append(None)

        except Exception:
            continue

    print(f"Accepted beats: {accepted}/{total}")

    for k in out:
        out[k] = gem_list(out[k])

    return out

# ==========================================================
# === MAIN
# ==========================================================

ECG_DIR = BASE / "data" / "subset_prova"
OUT_JSON = BASE / "data" / "ecg_features_subset_prova.json"

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

        rr_mean = round(sum(rr_valid) / len(rr_valid), 1) if rr_valid else None

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
