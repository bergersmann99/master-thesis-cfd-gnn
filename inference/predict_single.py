"""
Inferenz-Skript für GNN-Surrogate Modelle (GCN und GATv2).

Führt Vorhersagen für beliebige Graphen durch und speichert:
  - VTU-Datei mit Vorhersage (für ParaView)
  - NumPy-Arrays mit Positionen und Vorhersagen (für Interpolation)
  - Optional: Ground Truth und Fehler (falls .y im Graphen vorhanden)

Verwendung — neuer Graph ohne Ground Truth:
  python predict.py \
    --model gcn \
    --checkpoint /path/to/gcn/best_model.pt \
    --graph /path/to/new_graph.pt \
    --output-dir ./predictions \
    --prefix sim_new

Verwendung — Testgraph mit Ground Truth:
  python predict.py \
    --model gat \
    --checkpoint /path/to/gat/best_model.pt \
    --graph /path/to/datasets/medium/test.pt \
    --graph-index 0 \
    --output-dir ./predictions \
    --prefix sim_001
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent / "GNN"))
sys.path.insert(0, str(Path(__file__).parent / "GAT"))

FIELD_NAMES = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]


def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return tensor * std + mean


def load_gcn_model(checkpoint_path: Path, device):
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


def run_inference(model, data, device, stats: dict):
    """Führt Inferenz durch und gibt denormalisierte Werte zurück."""
    model.eval()
    y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device)
    y_std  = torch.tensor(stats["y_std"],  dtype=torch.float32, device=device)

    with torch.no_grad():
        data = data.to(device)
        pred_norm = model(data.x, data.edge_index)

    pred = denormalize(pred_norm, y_mean, y_std).cpu().numpy()
    pos  = data.pos.cpu().numpy()

    true = None
    if data.y is not None:
        true = denormalize(data.y, y_mean, y_std).cpu().numpy()

    return pos, pred, true


def save_vtu(pos: np.ndarray, fields: dict, path: Path):
    cloud = pv.PolyData(pos)
    for name, values in fields.items():
        cloud.point_data[name] = values
    cloud.save(str(path))
    print(f"  Gespeichert: {path.name}")


def main():
    parser = argparse.ArgumentParser(description="GNN Inferenz für neue Graphen")
    parser.add_argument("--model",       required=True, choices=["gcn", "gat"],
                        help="Modelltyp: gcn oder gat")
    parser.add_argument("--checkpoint",  required=True, type=str,
                        help="Pfad zum Checkpoint (.pt)")
    parser.add_argument("--graph",       required=True, type=str,
                        help="Pfad zur Graphdatei (.pt — PyG Data oder Liste)")
    parser.add_argument("--graph-index", type=int, default=0,
                        help="Index falls .pt eine Liste enthält (Standard: 0)")
    parser.add_argument("--output-dir",  type=str, default="./predictions",
                        help="Ausgabeverzeichnis (Standard: ./predictions)")
    parser.add_argument("--prefix",      type=str, default="prediction",
                        help="Dateiname-Präfix (Standard: prediction)")
    parser.add_argument("--no-ground-truth", action="store_true",
                        help="Ground Truth unterdrücken (auch wenn .y vorhanden)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGerät: {device}")

    # ── Modell laden ──────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if args.model == "gcn":
        model, stats = load_gcn_model(ckpt_path, device)
        model_name = "GCN"
    else:
        model, stats = load_gat_model(ckpt_path, device)
        model_name = "GATv2"
    print(f"Modell geladen: {model_name}  ({ckpt_path.name})")

    # ── Graph laden ───────────────────────────────────────────────────────────
    graph_data = torch.load(args.graph, weights_only=False)
    if isinstance(graph_data, list):
        data = graph_data[args.graph_index]
        print(f"Graph [{args.graph_index}] aus Liste ({len(graph_data)} Graphen)")
    else:
        data = graph_data

    sim_id = getattr(data, "sim_id", args.prefix)
    print(f"  sim_id: {sim_id}")
    print(f"  Knoten: {data.num_nodes:,}  |  Kanten: {data.num_edges:,}")

    # ── Inferenz ──────────────────────────────────────────────────────────────
    print("Inferenz läuft...")
    pos, pred, true = run_inference(model, data, device, stats)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{args.prefix}_{model_name}"

    # Vorhersage → VTU
    fields_pred = {name: pred[:, i] for i, name in enumerate(FIELD_NAMES)}
    save_vtu(pos, fields_pred, out_dir / f"{pfx}_prediction.vtu")

    # NumPy-Arrays → für interpolate_to_full_mesh.py
    np.save(out_dir / f"{pfx}_pos.npy",  pos)
    np.save(out_dir / f"{pfx}_pred.npy", pred)
    print(f"  NumPy gespeichert: {pfx}_pos.npy, {pfx}_pred.npy")

    # Ground Truth + Fehler (optional)
    if true is not None and not args.no_ground_truth:
        fields_true = {name: true[:, i] for i, name in enumerate(FIELD_NAMES)}
        save_vtu(pos, fields_true, out_dir / f"{pfx}_ground_truth.vtu")
        np.save(out_dir / f"{pfx}_true.npy", true)

        fields_err = {}
        for i, name in enumerate(FIELD_NAMES):
            abs_err = np.abs(pred[:, i] - true[:, i])
            rel_err = abs_err / (np.abs(true[:, i]) + 1e-8)
            fields_err[f"{name}_abs_error"] = abs_err
            fields_err[f"{name}_rel_error"] = rel_err
        save_vtu(pos, fields_err, out_dir / f"{pfx}_error.vtu")

    print(f"\n✓ Fertig. Ausgabe: {out_dir}/")
    print("  Nächster Schritt: interpolate_to_full_mesh.py für vollständiges CFD-Netz")


if __name__ == "__main__":
    main()
