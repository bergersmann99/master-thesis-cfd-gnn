"""
block_c_gaps.py
===============
Schliesst zwei Bericht-Luecken zu den GCN-Rerun-Modellen aus Block B:

Aufgabe 1 — rL2 (gesamt) je Stufe auf dem Testset
    Liest die vom Training erzeugte test_metrics.json aus
    output_gcn_{coarse,medium,bf25}_rerun/. Verifiziert das darin
    gespeicherte gesamt.R2 gegen die Vorgabewerte 0.9216 / 0.9574 / 0.9639
    (Toleranz ±0.0003). Bei Abweichung Abbruch.

Aufgabe 2 — feldweise R² je Test-Sim (Stufe Medium)
    Laedt das Medium-Rerun-Modell (epoch == 734), das Test-Split der
    Medium-Stufe und ruft die unveraenderte evaluate_detailed-Funktion
    aus trainGCN_efficient.py pro Einzel-Graph auf. Verifiziert das
    per-Sim gesamt-R² gegen die Vorgabewerte (Toleranz ±0.0005).
    Bei Abweichung Abbruch.

Ausgabe
-------
1. gap1_rL2_gesamt.yaml — R2_gesamt und rL2_gesamt pro Stufe.
2. gap2_per_sim_medium.csv + gap2_per_sim_medium.md —
   feldweise R² Tabelle der acht Test-Sims.

Keine neue Metrikdefinition. evaluate_detailed wird unveraendert importiert
aus /home/tbergermann/Python/GNN/trainGCN_efficient.py. Normalisierungsstats
stammen aus dem Checkpoint, Datensaetze werden mit normalize_dataset()
(aus demselben Skript) auf die fuer evaluate_detailed erwartete Form
gebracht.

Quellen: siehe QUELLEN_block_c_gaps.md.
"""

import os
import sys
import json
import csv

import numpy as np
import torch
import yaml

# evaluate_detailed und Helfer aus dem Training-Skript wiederverwenden.
sys.path.insert(0, "/home/tbergermann/Python/GNN")
from trainGCN_efficient import (
    GCNSurrogate,
    evaluate_detailed,
    load_dataset,
    normalize_dataset,
)


# ----------------------------------------------------------------------
# Pfade und Vorgabewerte
# ----------------------------------------------------------------------

GCN_DIR = "/home/tbergermann/Python/GNN"
OUT_DIR = "/home/tbergermann/Python"  # gap1/gap2-Outputs liegen hier

STAGES = {
    "coarse": {
        "ckpt": f"{GCN_DIR}/output_gcn_coarse_rerun/best_model.pt",
        "data": f"{GCN_DIR}/graph_dataset_coarse_rerun",
        "test_metrics": f"{GCN_DIR}/output_gcn_coarse_rerun/test_metrics.json",
        "expected_epoch": 1274,
        "expected_R2": 0.9216,
    },
    "medium": {
        "ckpt": f"{GCN_DIR}/output_gcn_medium_rerun/best_model.pt",
        "data": f"{GCN_DIR}/graph_dataset_medium_rerun",
        "test_metrics": f"{GCN_DIR}/output_gcn_medium_rerun/test_metrics.json",
        "expected_epoch": 734,
        "expected_R2": 0.9574,
    },
    "bf25": {
        "ckpt": f"{GCN_DIR}/output_gcn_bf25_rerun/best_model.pt",
        "data": f"{GCN_DIR}/graph_dataset_bf25_rerun",
        "test_metrics": f"{GCN_DIR}/output_gcn_bf25_rerun/test_metrics.json",
        "expected_epoch": 1694,
        "expected_R2": 0.9639,
    },
}

EXPECTED_PER_SIM = {
    "sim_001": 0.8997,
    "sim_002": 0.9558,
    "sim_008": 0.9151,
    "sim_013": 0.7688,
    "sim_014": 0.9657,
    "sim_035": 0.8987,
    "sim_038": 0.9441,
    "sim_048": 0.9545,
}

FIELDS = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
TOL_R2_TEST = 0.0003
TOL_R2_PERSIM = 0.0005


def abort(msg):
    print(f"\nABBRUCH: {msg}", file=sys.stderr)
    sys.exit(1)


def fmt6(x):
    return f"{x:.6f}"


# ----------------------------------------------------------------------
# Aufgabe 1 — rL2 (gesamt) je Stufe aus test_metrics.json
# ----------------------------------------------------------------------

def aufgabe1():
    print("=" * 72)
    print("Aufgabe 1: rL2 (gesamt) je Stufe auf dem Testset")
    print("=" * 72)
    result = {}
    for stage, cfg in STAGES.items():
        tm_path = cfg["test_metrics"]
        if not os.path.exists(tm_path):
            abort(f"Stufe '{stage}': test_metrics.json fehlt: {tm_path}")
        tm = json.load(open(tm_path))
        if "gesamt" not in tm or "R2" not in tm["gesamt"] or "rL2" not in tm["gesamt"]:
            abort(f"Stufe '{stage}': test_metrics.json unvollstaendig.")
        r2 = float(tm["gesamt"]["R2"])
        rl2 = float(tm["gesamt"]["rL2"])
        # Verifikation gegen Vorgabe
        if abs(r2 - cfg["expected_R2"]) > TOL_R2_TEST:
            abort(
                f"Stufe '{stage}': R2_gesamt={r2:.6f} weicht von "
                f"Vorgabe {cfg['expected_R2']} um > {TOL_R2_TEST} ab. "
                f"Vermutlich falscher Checkpoint."
            )
        result[stage] = {"R2_gesamt": r2, "rL2_gesamt": rl2}
        print(f"  {stage:<8} R2_gesamt={fmt6(r2)}  rL2_gesamt={fmt6(rl2)}  "
              f"(Vorgabe R2={cfg['expected_R2']}, |Δ|={abs(r2-cfg['expected_R2']):.6f} <= {TOL_R2_TEST})")

    out_path = os.path.join(OUT_DIR, "gap1_rL2_gesamt.yaml")
    with open(out_path, "w") as f:
        # Reihenfolge erhalten
        f.write("# Aufgabe 1 — rL2 (gesamt) je Stufe auf dem Testset\n")
        f.write("# Quelle: output_gcn_<stufe>_rerun/test_metrics.json\n")
        for stage in ["coarse", "medium", "bf25"]:
            r2 = result[stage]["R2_gesamt"]
            rl2 = result[stage]["rL2_gesamt"]
            f.write(f"\n{stage}:\n")
            f.write(f"  R2_gesamt: {fmt6(r2)}\n")
            f.write(f"  rL2_gesamt: {fmt6(rl2)}\n")
    print(f"\n  Geschrieben: {out_path}")
    return result


# ----------------------------------------------------------------------
# Aufgabe 2 — feldweise R² je Test-Sim (Medium)
# ----------------------------------------------------------------------

def aufgabe2():
    print("\n" + "=" * 72)
    print("Aufgabe 2: feldweise R² je Test-Sim — Stufe Medium")
    print("=" * 72)
    cfg = STAGES["medium"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint laden + verifizieren
    if not os.path.exists(cfg["ckpt"]):
        abort(f"Checkpoint fehlt: {cfg['ckpt']}")
    ck = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    if ck.get("epoch") != cfg["expected_epoch"]:
        abort(
            f"Medium-Checkpoint: epoch={ck.get('epoch')} != "
            f"erwartet {cfg['expected_epoch']}"
        )
    hp = ck["hyperparameters"]
    stats = {k: v.to(device) for k, v in ck["norm_stats"].items()}
    in_dim = int(stats["x_mean"].shape[0])
    if in_dim != 14:
        abort(f"in_dim={in_dim} (erwartet 14) — falscher Datensatz?")

    model = GCNSurrogate(
        in_dim=in_dim,
        out_dim=6,
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp.get("dropout", 0.0),
        use_gradient_checkpointing=False,  # Inferenz, daher aus
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modell: GCNSurrogate, epoch={ck['epoch']}, params={n_params:,}")
    print(f"  Checkpoint: {cfg['ckpt']}")

    # Datensatz laden (train, val, test); nur test wird ausgewertet
    train_data, val_data, test_data = load_dataset(cfg["data"])

    if len(test_data) != 8:
        abort(f"Test-Split enthaelt {len(test_data)} Graphen (erwartet 8).")

    # Sim-IDs aus den Data-Objekten, NICHT aus Listenindex
    for g in test_data:
        if not hasattr(g, "sim_id"):
            abort(f"Graph ohne sim_id-Attribut gefunden.")

    expected_set = set(EXPECTED_PER_SIM.keys())
    found_set = {g.sim_id for g in test_data}
    if expected_set != found_set:
        abort(f"Sim-IDs im Test-Split: {sorted(found_set)} != erwartet "
              f"{sorted(expected_set)}")

    # Normalisierung: evaluate_detailed denormalisiert intern,
    # erwartet aber normalisierte data.x und data.y (wie im Training).
    # cpu_stats fuer normalize_dataset (in-place, auf CPU-Tensors).
    cpu_stats = {k: v.cpu() for k, v in stats.items()}
    test_data = normalize_dataset(test_data, cpu_stats)

    print(f"\n  {'sim_id':<10} " + " ".join(f"{f:>10}" for f in FIELDS)
          + f"  {'gesamt':>10}  {'expected':>10}  {'Δ':>10}")
    print("  " + "-" * 110)

    rows = []
    deviations = {}
    # Reihenfolge der Tabelle: festgelegt durch EXPECTED_PER_SIM
    sim_to_graph = {g.sim_id: g for g in test_data}
    for sim_id in EXPECTED_PER_SIM:
        graph = sim_to_graph[sim_id]
        metrics = evaluate_detailed(model, [graph], cpu_stats, device)
        r2_per_field = {f: float(metrics[f]["R2"]) for f in FIELDS}
        r2_gesamt = float(metrics["gesamt"]["R2"])
        expected = EXPECTED_PER_SIM[sim_id]
        delta = r2_gesamt - expected
        deviations[sim_id] = (r2_gesamt, expected, delta)
        marker = " <" if abs(delta) > TOL_R2_PERSIM else "  "
        cols = " ".join(f"{r2_per_field[f]:>10.6f}" for f in FIELDS)
        print(f"  {sim_id:<10} {cols}  {r2_gesamt:>10.6f}  "
              f"{expected:>10.6f}  {delta:>+10.6f}{marker}")
        rows.append({
            "sim": sim_id,
            **{f: r2_per_field[f] for f in FIELDS},
            "gesamt": r2_gesamt,
        })

    # Verifikation: alle Sims muessen innerhalb der Toleranz liegen
    bad = [
        (s, r, e, d) for s, (r, e, d) in deviations.items()
        if abs(d) > TOL_R2_PERSIM
    ]
    if bad:
        msg = "  ".join(
            f"{s}: {r:.6f} vs {e:.6f} (Δ={d:+.6f})" for s, r, e, d in bad
        )
        abort(f"Per-Sim R² ausserhalb Toleranz {TOL_R2_PERSIM}: {msg}")

    # CSV schreiben
    csv_path = os.path.join(OUT_DIR, "gap2_per_sim_medium.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim"] + FIELDS + ["gesamt"])
        for row in rows:
            writer.writerow(
                [row["sim"]]
                + [fmt6(row[fld]) for fld in FIELDS]
                + [fmt6(row["gesamt"])]
            )
    print(f"\n  CSV geschrieben: {csv_path}")

    # Markdown schreiben
    md_path = os.path.join(OUT_DIR, "gap2_per_sim_medium.md")
    with open(md_path, "w") as f:
        f.write("# Aufgabe 2 — feldweise R² je Test-Sim (GCN Medium-Rerun)\n\n")
        f.write(f"Checkpoint: `{cfg['ckpt']}` (epoch={ck['epoch']})\n\n")
        f.write("| sim | " + " | ".join(FIELDS) + " | gesamt |\n")
        f.write("|---|" + "|".join(["---"] * (len(FIELDS) + 1)) + "|\n")
        for row in rows:
            cells = [row["sim"]]
            cells += [fmt6(row[fld]) for fld in FIELDS]
            cells += [fmt6(row["gesamt"])]
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"  Markdown geschrieben: {md_path}")

    return rows


def main():
    aufgabe1()
    aufgabe2()
    print("\nFertig.")


if __name__ == "__main__":
    main()
