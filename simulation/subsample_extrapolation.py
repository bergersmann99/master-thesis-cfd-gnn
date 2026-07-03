"""
subsample_extrapolation.py
==========================
Subsampled die beiden Extrapolations-Simulationen (sturm / schwachwind)
in den Auflösungen coarse, medium und bf25 und lädt die Ergebnisse nach S3.

S3-Zielstruktur:
    graph-dataset_extrapolation/
        coarse/
            sturm_25ms_45deg.pt
            schwachwind_1_5ms_45deg.pt
        medium/
            sturm_25ms_45deg.pt
            schwachwind_1_5ms_45deg.pt
        bf25/
            sturm_25ms_45deg.pt
            schwachwind_1_5ms_45deg.pt
"""

import os
import sys
import time
import yaml
import numpy as np
import torch

import createGraphDataset as cgd


SIMULATIONS = [
    {"id": "sturm_25ms_45deg",        "U_ref": 25.0, "angle": 45.0, "status": "Converged"},
    {"id": "schwachwind_1_5ms_45deg", "U_ref":  1.5, "angle": 45.0, "status": "Converged"},
]

LEVELS = {
    "coarse": cgd.SUBSAMPLE_ZONES["coarse"],
    "medium": cgd.SUBSAMPLE_ZONES["medium"],
    "bf25":   cgd.SUBSAMPLE_ZONES["building_focus_25"],
}

S3_BUCKET  = "amzn-master-sim-bucket"
S3_PREFIX  = "graph-dataset_extrapolation"


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print(f"FEHLER: '{path}' nicht gefunden.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def upload(local_path, s3_path, max_retries=3):
    import subprocess
    local_size = os.path.getsize(local_path)
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                ["aws", "s3", "cp", local_path, s3_path,
                 "--cli-read-timeout", "120", "--cli-connect-timeout", "30"],
                check=True, capture_output=True,
            )
            result = subprocess.run(
                ["aws", "s3", "ls", s3_path],
                capture_output=True, check=True,
            )
            parts = result.stdout.decode().strip().split()
            if len(parts) >= 3 and int(parts[2]) == local_size:
                return True
            raise RuntimeError("Groesse stimmt nicht überein")
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"   WARNUNG: Upload fehlgeschlagen (Versuch {attempt}), retry in {wait}s")
                time.sleep(wait)
            else:
                print(f"   FEHLER: Upload endgültig fehlgeschlagen: {e}")
    return False


def main():
    cfg = load_config("config.yaml")
    seed = cfg["general"]["random_seed"]
    rng = np.random.default_rng(seed)

    base_path   = os.getcwd()
    vtk_base    = os.path.join(base_path, cfg["general"]["results_dir"], "vtks")
    output_base = os.path.join(base_path, cfg["general"]["results_dir"], "extrapolation_graphs")
    os.makedirs(output_base, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"   SUBSAMPLING EXTRAPOLATIONS-SIMULATIONEN")
    print(f"{'=' * 60}\n")

    for level, zones in LEVELS.items():
        level_dir = os.path.join(output_base, level)
        os.makedirs(level_dir, exist_ok=True)

        print(f"  Level: {level.upper()}")

        for sim in SIMULATIONS:
            sim_id = sim["id"]
            t0 = time.time()

            print(f"    [{sim_id}] subsampling...", end="", flush=True)

            data, stats = cgd.process_simulation(
                sim_meta=sim,
                vtk_base_dir=vtk_base,
                rng=rng,
                k_neighbors=20,
                zones=zones,
            )

            out_path = os.path.join(level_dir, f"{sim_id}.pt")
            torch.save(data, out_path)

            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            elapsed = time.time() - t0
            print(f" {stats['n_subsampled']:,} Knoten, {size_mb:.0f} MB, {elapsed:.0f}s")

            # S3 Upload
            s3_dest = f"s3://{S3_BUCKET}/{S3_PREFIX}/{level}/{sim_id}.pt"
            print(f"    [{sim_id}] upload -> {s3_dest} ...", end="", flush=True)
            ok = upload(out_path, s3_dest)
            print(" OK" if ok else " FEHLER")

        print()

    print(f"{'=' * 60}")
    print(f"   FERTIG")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
