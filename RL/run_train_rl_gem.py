import os
import subprocess
from pathlib import Path

def main():
    # ============================
    # CONFIGURAZIONE PERCORSI
    # ============================

    BASE = Path(__file__).resolve().parents[1]

    MODEL_PATH = BASE / "models" / "GEM_model"
    PTBXL_PATH = BASE / "data" / "ptbxl_subset.json"
    RL_DATASET = BASE / "progetto" / "RL" / "rl_dataset.json"

    IMAGE_FOLDER = BASE / "data" / "ecg_images" / "gen_images" / "ptb-xl-gen"
    ECG_FOLDER = BASE / "data" / "ecg_timeseries" / "ptbxl"

    OUTPUT_DIR = BASE / "rl_outputs"

    # ============================
    # PARAMETRI TRAINING RL
    # ============================

    LR = "1e-5"
    BATCH = "1"
    EPOCHS = "1"

    # ============================
    # COSTRUZIONE COMANDO
    # ============================

    cmd = [
        "python3",
        "train_gem_rl.py",
        "--model_path", str(MODEL_PATH),
        "--rl_dataset_path", str(RL_DATASET),
        "--ptbxl_path", str(PTBXL_PATH),
        "--image_folder", str(IMAGE_FOLDER),
        "--ecg_folder", str(ECG_FOLDER),
        "--output_dir", str(OUTPUT_DIR),
        "--lr", LR,
        "--batch_size", BATCH,
        "--num_epochs", EPOCHS
    ]

    print("\n🚀 Avvio training RL multimodale GEM...\n")
    print("Comando eseguito:")
    print(" ".join(cmd), "\n")

    # ============================
    # ESECUZIONE
    # ============================

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line.strip())

    process.wait()

    print("\n🎉 Training RL completato!")

if __name__ == "__main__":
    main()
