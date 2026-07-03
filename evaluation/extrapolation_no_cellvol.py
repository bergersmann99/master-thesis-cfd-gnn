"""
extrapolation_no_cellvol.py
===========================
Extrapolationsauswertung der 13-Feature-Variante (no_cellvol) des
GATv2-Medium-Modells fuer die beiden Faelle Sturm (25 m/s, 45 Grad)
und Schwachwind (1.5 m/s, 45 Grad). Vergleich gegen die bestehende
14-Feature-Variante.

Vorgehen
--------
1. Validierungs-Gate auf sim_014 (separat dokumentiert):
   Spalte 4 (cell_volume) aus dem 14-F-Graphen entfernt = 13-F-Graph
   (bit-exakt; pos, edge_index, data.y ebenfalls identisch).
   Damit ist die deterministische Slicing-Ableitung der Extrap-Graphen
   bewiesen aequivalent zur off-Maschine-Subsamplung.

2. Slicing der vorhandenen 14-F-Extrap-Files
   (extrapolation_pt/medium_{sturm,schwachwind}_*.pt) auf 13 Features
   durch Entfernen der Spalte 4.

3. Inferenz mit dem no_cellvol-Checkpoint, geladen aus
   models/gatv2_medium_no_cellvol_efficient/best_model.pt. Eingaben mit
   den Trainings-Normalisierungsstatistiken aus dem Checkpoint
   (norm_stats) normalisieren, Ausgaben damit denormalisieren — keine
   Neuberechnung von Statistiken.

4. Metriken (feldweise + gesamt) berechnen: R^2 und relativer L2-
   Fehler. Feldreihenfolge: Ux, Uy, Uz, p, k, epsilon. Logik aequivalent
   zu evaluate_detailed (trainGATv2_efficient.py).

5. Vergleich mit den bekannten 14-F-GATv2-Werten aus der Arbeit:
   Sturm        R^2 = 0.848, rL2 = 0.330
   Schwachwind  R^2 = -183.63, rL2 = 6.50

Ausgabe
-------
- results/extrapolation_no_cellvol.yaml mit allen Werten.
- Konsolentabelle (Felderreihenfolge wie vorgegeben).

Siehe QUELLEN_extrapolation_no_cellvol.md.
"""

import os
import sys
import datetime

import numpy as np
import torch
import yaml

# Importiere Surrogate-Klasse aus dem unveraenderten Trainingsskript
sys.path.insert(0, "/home/tbergermann/Python/GAT")
from trainGATv2_efficient import GATv2Surrogate


# ----------------------------------------------------------------------
# Pfade (alle verifiziert vor Aufruf)
# ----------------------------------------------------------------------

CHECKPOINT = "/home/tbergermann/Python/GAT/output_gatv2_medium_no_cellvol_efficient_s3/best_model.pt"

EXTRAP_FILES_14F = {
    "sturm_25ms_45deg": "/home/tbergermann/Python/predictions/extrapolation_pt/medium_sturm_25ms_45deg.pt",
    "schwachwind_1_5ms_45deg": "/home/tbergermann/Python/predictions/extrapolation_pt/medium_schwachwind_1_5ms_45deg.pt",
}

OUT_YAML = "/home/tbergermann/results/extrapolation_no_cellvol.yaml"

FIELDS = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]

# 14-F-GATv2-Referenzwerte (aus der bestehenden Arbeit)
REF_14F = {
    "sturm_25ms_45deg":      {"R2_gesamt": 0.848,   "rL2_gesamt": 0.330},
    "schwachwind_1_5ms_45deg": {"R2_gesamt": -183.63, "rL2_gesamt": 6.50},
}


# ----------------------------------------------------------------------
# Slicing 14-F -> 13-F
# ----------------------------------------------------------------------

def slice_cellvol(g_14):
    """Entfernt Spalte 4 (cell_volume) aus den Knoten-Features."""
    g = g_14.clone()
    g.x = torch.cat([g_14.x[:, :4], g_14.x[:, 5:]], dim=1)
    return g


# ----------------------------------------------------------------------
# Modell + Stats laden
# ----------------------------------------------------------------------

def load_model_and_stats(checkpoint_path, device):
    """Laedt das 13-F-GATv2-Modell samt Normalisierungsstatistiken aus dem Checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ck = torch.load(checkpoint_path, map_location=device,
                    weights_only=False)
    arch = ck.get("architecture", "")
    if "GATv2" not in arch:
        raise RuntimeError(f"architecture='{arch}' enthaelt nicht 'GATv2'.")
    hp = ck["hyperparameters"]
    norm_stats = {k: v.to(device) for k, v in ck["norm_stats"].items()}
    in_dim = int(norm_stats["x_mean"].shape[0])
    if in_dim != 13:
        raise RuntimeError(
            f"in_dim={in_dim} im no_cellvol-Checkpoint (erwartet 13)."
        )
    model = GATv2Surrogate(
        in_dim=in_dim, out_dim=6,
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        heads=hp.get("heads", 4),
        dropout=hp.get("dropout", 0.0),
        attention_dropout=hp.get("attention_dropout", 0.0),
        use_gradient_checkpointing=False,  # Inferenz
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, norm_stats, ck, hp


# ----------------------------------------------------------------------
# Metriken (Logik aequivalent zu evaluate_detailed)
# ----------------------------------------------------------------------

def compute_field_metrics(pred, true):
    """Pred und true sind (N, 6) denormalisierte Float-Arrays.
    Liefert dict mit Feldern + 'gesamt'.
    Definitionen:
      R^2  = 1 - SS_res / SS_tot mit Mittelwert von t als Baseline.
      rL2 = ||p - t||_2 / max(||t||_2, eps).
    """
    metrics = {}
    for i, name in enumerate(FIELDS):
        p = pred[:, i]
        t = true[:, i]
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rl2 = float(
            np.sqrt(np.sum((p - t) ** 2))
            / max(np.sqrt(np.sum(t ** 2)), 1e-12)
        )
        metrics[name] = {"R2": float(r2), "rL2": float(rl2)}
    metrics["gesamt"] = {
        "R2": float(np.mean([metrics[n]["R2"] for n in FIELDS])),
        "rL2": float(np.mean([metrics[n]["rL2"] for n in FIELDS])),
    }
    return metrics


# ----------------------------------------------------------------------
# Validierungs-Gate (sim_014, fuer Bericht reproduzierbar)
# ----------------------------------------------------------------------

def gate_sim014():
    """Validierungs-Gate: prueft die bit-exakte Aequivalenz des Slicings auf sim_014."""
    g14 = None
    g13 = None
    for p in [
        "/home/tbergermann/Python/GNN/graph_dataset_medium_rerun/train.pt",
        "/home/tbergermann/Python/GNN/graph_dataset_medium_rerun/val.pt",
        "/home/tbergermann/Python/GNN/graph_dataset_medium_rerun/test.pt",
    ]:
        cand = [g for g in torch.load(p, weights_only=False)
                if g.sim_id == "sim_014"]
        if cand:
            g14 = cand[0]
            break
    for p in [
        "/home/tbergermann/Python/datasets/medium_no_cellvol/train.pt",
        "/home/tbergermann/Python/datasets/medium_no_cellvol/val.pt",
        "/home/tbergermann/Python/datasets/medium_no_cellvol/test.pt",
    ]:
        cand = [g for g in torch.load(p, weights_only=False)
                if g.sim_id == "sim_014"]
        if cand:
            g13 = cand[0]
            break
    if g14 is None or g13 is None:
        raise RuntimeError("Gate: sim_014 nicht in beiden Datensaetzen gefunden.")
    x_sliced = torch.cat([g14.x[:, :4], g14.x[:, 5:]], dim=1)
    checks = {
        "x[:, !=4]": torch.equal(x_sliced, g13.x),
        "pos":        torch.equal(g14.pos, g13.pos),
        "edge_index": torch.equal(g14.edge_index, g13.edge_index),
        "y":          torch.equal(g14.y, g13.y),
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Gate FEHLGESCHLAGEN: {failed}")
    return {k: bool(v) for k, v in checks.items()}


# ----------------------------------------------------------------------
# Hauptlauf
# ----------------------------------------------------------------------

def main():
    """Fuehrt Gate, Inferenz und Metrikberechnung fuer beide Extrapolationsfaelle aus."""
    print("=" * 72)
    print("Validierungs-Gate (sim_014)")
    print("=" * 72)
    gate_results = gate_sim014()
    for k, v in gate_results.items():
        print(f"  {'OK' if v else 'FAIL':<4}  {k}")
    print("Gate BESTANDEN — Slicing ist bit-exakt aequivalent.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLade no_cellvol-Modell aus S3-Spiegel ...")
    model, norm_stats, ck, hp = load_model_and_stats(CHECKPOINT, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Checkpoint:  {CHECKPOINT}")
    print(f"  Epoche:      {ck.get('epoch')}")
    print(f"  val_loss:    {ck.get('val_loss')}")
    print(f"  Parameter:   {n_params:,}")
    print(f"  in_dim:      {norm_stats['x_mean'].shape[0]} (no_cellvol)")
    print(f"  HP:          hidden_dim={hp['hidden_dim']}, "
          f"layers={hp['num_layers']}, heads={hp.get('heads')}")

    x_mean = norm_stats["x_mean"]
    x_std = norm_stats["x_std"]
    y_mean = norm_stats["y_mean"]
    y_std = norm_stats["y_std"]

    all_results = {
        "_meta": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "checkpoint": CHECKPOINT,
            "checkpoint_epoch": int(ck.get("epoch", -1)),
            "checkpoint_val_loss": float(ck.get("val_loss", 0.0)),
            "n_params": int(n_params),
            "gate_sim014": gate_results,
            "method": "13-F-Extrap durch Slicing der 14-F-Files "
                     "(Spalte 4 entfernt). Gate auf sim_014 bestanden.",
            "fields_order": FIELDS,
        },
        "cases": {},
        "reference_14F_GATv2": REF_14F,
    }

    for case, src in EXTRAP_FILES_14F.items():
        print(f"\n{'=' * 72}")
        print(f"Fall: {case}")
        print('=' * 72)
        g14 = torch.load(src, weights_only=False)[0]
        print(f"  14-F-Quelle: {src}")
        print(f"  sim_id={g14.sim_id}, U_ref={g14.U_ref}, "
              f"angle={g14.angle}, n_nodes={g14.x.size(0)}, "
              f"n_edges={g14.edge_index.size(1)}")
        if g14.x.size(1) != 14:
            raise RuntimeError(
                f"  Unerwartet: x.shape[1]={g14.x.size(1)} (erwartet 14)"
            )
        # Slicing
        g13 = slice_cellvol(g14)
        assert g13.x.size(1) == 13

        # Normalisieren
        x = g13.x.to(device)
        ei = g13.edge_index.to(device)
        x_norm = (x - x_mean) / x_std
        # Inferenz
        with torch.no_grad():
            pred_norm = model(x_norm, ei)
            pred = pred_norm * y_std + y_mean
        pred_np = pred.cpu().numpy()
        true_np = g13.y.numpy()

        metrics = compute_field_metrics(pred_np, true_np)

        # Konsolentabelle
        print(f"  {'Feld':<10} {'R2':>12} {'rL2':>12}")
        print("  " + "-" * 36)
        for f in FIELDS:
            print(f"  {f:<10} {metrics[f]['R2']:>12.6f} "
                  f"{metrics[f]['rL2']:>12.6f}")
        print("  " + "-" * 36)
        print(f"  {'gesamt':<10} {metrics['gesamt']['R2']:>12.6f} "
              f"{metrics['gesamt']['rL2']:>12.6f}")

        # Vergleich mit 14-F-Referenz
        ref = REF_14F[case]
        d_r2 = metrics["gesamt"]["R2"] - ref["R2_gesamt"]
        d_rl2 = metrics["gesamt"]["rL2"] - ref["rL2_gesamt"]
        print("\n  Vergleich gegen 14-F GATv2 (Referenz):")
        print(f"    R²_gesamt    no_cellvol={metrics['gesamt']['R2']:.4f}  "
              f"14-F={ref['R2_gesamt']:.4f}  Δ={d_r2:+.4f}")
        print(f"    rL2_gesamt   no_cellvol={metrics['gesamt']['rL2']:.4f}  "
              f"14-F={ref['rL2_gesamt']:.4f}  Δ={d_rl2:+.4f}")

        all_results["cases"][case] = {
            "source_file_14F": src,
            "n_nodes": int(g14.x.size(0)),
            "n_edges": int(g14.edge_index.size(1)),
            "U_ref": float(g14.U_ref),
            "angle": float(g14.angle),
            "metrics": metrics,
            "vs_14F": {
                "R2_14F": ref["R2_gesamt"],
                "rL2_14F": ref["rL2_gesamt"],
                "delta_R2": float(d_r2),
                "delta_rL2": float(d_rl2),
            },
        }

    # YAML schreiben
    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        f.write("# Extrapolation no_cellvol (13-F-GATv2 Medium)\n")
        f.write("# Slicing-Ableitung der 14-F-Extrap-Graphen (Spalte "
                "cell_volume entfernt).\n")
        f.write("# Validierungs-Gate auf sim_014 bestanden "
                "(bit-exakt aequivalent).\n\n")
        yaml.safe_dump(all_results, f, sort_keys=False,
                       default_flow_style=False)
    print(f"\nGeschrieben: {OUT_YAML}")


if __name__ == "__main__":
    main()
