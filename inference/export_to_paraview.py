"""
Export der GNN-Vorhersagen als VTK-Dateien für ParaView.

Exportiert pro Testgraph und Modell:
  - Ground Truth (alle 6 Felder)
  - Vorhersage (alle 6 Felder)
  - Absoluter Fehler pro Feld
  - Relativer Fehler pro Feld

Ausgabe: .vtu-Dateien (VTK Unstructured Grid) — direkt in ParaView öffenbar.

Verwendung:
  python export_to_paraview.py \
    --gcn-checkpoint  /path/to/gcn/best_model.pt \
    --gat-checkpoint  /path/to/gat/best_model.pt \
    --data-dir        /path/to/datasets/medium \
    --output-dir      /path/to/paraview_export \
    --subsampling     medium
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pyvista as pv

# ── Modell-Importe ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "GNN"))
sys.path.insert(0, str(Path(__file__).parent / "GAT"))

FIELD_NAMES = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]


def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Macht die z-Score-Normalisierung rückgängig."""
    return tensor * std + mean


def build_point_cloud(pos: np.ndarray,
                      fields: dict[str, np.ndarray]) -> pv.PolyData:
    """
    Erstellt ein PyVista PolyData-Objekt aus Knotenpositionen und Feldwerten.
    Jeder Knoten ist ein Punkt; Felder werden als Punktdaten gespeichert.
    """
    cloud = pv.PolyData(pos)
    for name, values in fields.items():
        cloud.point_data[name] = values
    return cloud


def run_inference(model, data, device, y_mean, y_std):
    """Führt Inferenz durch und gibt denormalisierte Vorhersage zurück."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        pred_norm = model(data.x, data.edge_index)

    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float32, device=device)

    pred = denormalize(pred_norm, y_mean_t, y_std_t).cpu().numpy()
    true = denormalize(data.y,   y_mean_t, y_std_t).cpu().numpy()
    pos  = data.pos.cpu().numpy()

    return pos, pred, true


def export_graph(pos, pred, true, out_dir: Path, prefix: str):
    """
    Exportiert Ground Truth, Vorhersage und Fehler als .vtu-Dateien.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Ground Truth ──────────────────────────────────────────────────────────
    fields_true = {name: true[:, i] for i, name in enumerate(FIELD_NAMES)}
    cloud_true = build_point_cloud(pos, fields_true)
    path_true = out_dir / f"{prefix}_ground_truth.vtu"
    cloud_true.save(str(path_true))
    print(f"  Gespeichert: {path_true.name}")

    # ── Vorhersage ────────────────────────────────────────────────────────────
    fields_pred = {name: pred[:, i] for i, name in enumerate(FIELD_NAMES)}
    cloud_pred = build_point_cloud(pos, fields_pred)
    path_pred = out_dir / f"{prefix}_prediction.vtu"
    cloud_pred.save(str(path_pred))
    print(f"  Gespeichert: {path_pred.name}")

    # ── Fehler ────────────────────────────────────────────────────────────────
    fields_err = {}
    for i, name in enumerate(FIELD_NAMES):
        abs_err = np.abs(pred[:, i] - true[:, i])
        # Relativer Fehler: |pred - true| / (|true| + eps)
        rel_err = abs_err / (np.abs(true[:, i]) + 1e-8)
        fields_err[f"{name}_abs_error"] = abs_err
        fields_err[f"{name}_rel_error"] = rel_err

    cloud_err = build_point_cloud(pos, fields_err)
    path_err = out_dir / f"{prefix}_error.vtu"
    cloud_err.save(str(path_err))
    print(f"  Gespeichert: {path_err.name}")


def load_gcn_model(checkpoint_path: Path, device):
    """Lädt das GCN-Modell samt Normalisierungsstatistiken aus dem Checkpoint."""
    from trainGCN import GCNSurrogate
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparameters"]
    model = GCNSurrogate(
        in_dim=14,
        hidden_dim=hp["hidden_dim"],
        out_dim=6,
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["norm_stats"]


def load_gat_model(checkpoint_path: Path, device):
    """Lädt das GATv2-Modell samt Normalisierungsstatistiken aus dem Checkpoint."""
    from trainGATv2 import GATv2Surrogate
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparameters"]
    model = GATv2Surrogate(
        in_dim=14,
        hidden_dim=hp["hidden_dim"],
        out_dim=6,
        num_layers=hp["num_layers"],
        heads=hp.get("heads", 4),
        dropout=hp.get("dropout", 0.0),
        attention_dropout=hp.get("attention_dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["norm_stats"]


def main():
    """Parst CLI-Argumente und exportiert Testgraphen beider Modelle als VTU-Dateien."""
    parser = argparse.ArgumentParser(description="Export GNN predictions to ParaView VTU files")
    parser.add_argument("--gcn-checkpoint", type=str, required=False,
                        help="Pfad zum besten GCN Checkpoint (.pt)")
    parser.add_argument("--gat-checkpoint", type=str, required=False,
                        help="Pfad zum besten GATv2 Checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Verzeichnis mit test.pt")
    parser.add_argument("--output-dir", type=str, default="./paraview_export",
                        help="Ausgabeverzeichnis für VTU-Dateien")
    parser.add_argument("--subsampling", type=str, default="medium",
                        help="Subsampling-Bezeichnung (grob/medium/fein)")
    parser.add_argument("--max-graphs", type=int, default=3,
                        help="Maximale Anzahl Testgraphen (Standard: 3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Gerät: {device}")

    # ── Testdaten laden ───────────────────────────────────────────────────────
    test_path = Path(args.data_dir) / "test.pt"
    test_data = torch.load(test_path, weights_only=False)
    print(f"  Testgraphen geladen: {len(test_data)}")

    out_base = Path(args.output_dir) / args.subsampling

    # ── GCN Export ────────────────────────────────────────────────────────────
    if args.gcn_checkpoint:
        print(f"\n[GCN] Lade Checkpoint: {args.gcn_checkpoint}")
        gcn_model, gcn_stats = load_gcn_model(Path(args.gcn_checkpoint), device)
        y_mean = np.array(gcn_stats["y_mean"])
        y_std  = np.array(gcn_stats["y_std"])

        for i, data in enumerate(test_data[:args.max_graphs]):
            sim_id = getattr(data, "sim_id", f"graph_{i:03d}")
            print(f"\n  Graph {i+1}/{min(len(test_data), args.max_graphs)}: {sim_id}")
            pos, pred, true = run_inference(gcn_model, data, device, y_mean, y_std)
            export_graph(pos, pred, true,
                         out_dir=out_base / "GCN",
                         prefix=f"{sim_id}_GCN_{args.subsampling}")

    # ── GATv2 Export ──────────────────────────────────────────────────────────
    if args.gat_checkpoint:
        print(f"\n[GATv2] Lade Checkpoint: {args.gat_checkpoint}")
        gat_model, gat_stats = load_gat_model(Path(args.gat_checkpoint), device)
        y_mean = np.array(gat_stats["y_mean"])
        y_std  = np.array(gat_stats["y_std"])

        for i, data in enumerate(test_data[:args.max_graphs]):
            sim_id = getattr(data, "sim_id", f"graph_{i:03d}")
            print(f"\n  Graph {i+1}/{min(len(test_data), args.max_graphs)}: {sim_id}")
            pos, pred, true = run_inference(gat_model, data, device, y_mean, y_std)
            export_graph(pos, pred, true,
                         out_dir=out_base / "GATv2",
                         prefix=f"{sim_id}_GATv2_{args.subsampling}")

    print(f"\n✓ Export abgeschlossen. Dateien in: {out_base}")
    print("  In ParaView öffnen: File → Open → *.vtu")
    print("  Tipp: 'Point Gaussian' Darstellung für Punktwolken-Visualisierung")


if __name__ == "__main__":
    main()
