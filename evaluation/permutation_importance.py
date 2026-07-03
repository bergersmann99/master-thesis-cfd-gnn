"""
permutation_importance.py
=========================
Permutation Importance fuer GNN-Surrogat-Modelle.

Misst den Einfluss jedes Input-Features auf die Modellguete,
indem jeweils ein Feature ueber alle Knoten permutiert wird
und die Verschlechterung des R² gemessen wird.

Verwendung:
    python permutation_importance.py \
        --checkpoint /pfad/zu/best_model.pt \
        --graph-source /pfad/zu/test.pt \
        --n-repeats 5 \
        --output-dir ./permutation_importance
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================================================================
# Feature-Namen (aus metadata.yaml)
# ==================================================================

INPUT_FEATURES = [
    "x", "y", "z",
    "wall_distance", "cell_volume",
    "type_interior", "type_inlet", "type_outlet",
    "type_ground", "type_building", "type_top", "type_sides",
    "U_ref", "angle",
]

OUTPUT_FEATURES = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]


# ==================================================================
# Modell-Definitionen (identisch zu predict.py)
# ==================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GCNProcessor(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(
                hidden_dim, hidden_dim,
                add_self_loops=True, normalize=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x)
            x = x + x_residual
        return x


class GATv2Processor(nn.Module):
    def __init__(self, hidden_dim, num_layers, heads=4,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        dim_per_head = hidden_dim // heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, dim_per_head, heads=heads,
                concat=True, dropout=attention_dropout,
                add_self_loops=True, share_weights=False))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x)
            x = x + x_residual
        return x


class GCNSurrogate(nn.Module):
    def __init__(self, in_dim=14, out_dim=6, hidden_dim=128,
                 num_layers=10, dropout=0.0):
        super().__init__()
        self.encoder = MLP(in_dim, hidden_dim, hidden_dim)
        self.processor = GCNProcessor(hidden_dim, num_layers)
        self.decoder = MLP(hidden_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = self.processor(h, edge_index)
        return self.decoder(h)


class GATv2Surrogate(nn.Module):
    def __init__(self, in_dim=14, out_dim=6, hidden_dim=128,
                 num_layers=10, heads=4, dropout=0.0,
                 attention_dropout=0.0):
        super().__init__()
        self.encoder = MLP(in_dim, hidden_dim, hidden_dim)
        self.processor = GATv2Processor(
            hidden_dim, num_layers, heads)
        self.decoder = MLP(hidden_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = self.processor(h, edge_index)
        return self.decoder(h)


# ==================================================================
# Hilfsfunktionen
# ==================================================================

def load_model(checkpoint_path, device):
    """Laedt Modell aus Checkpoint."""
    cp = torch.load(checkpoint_path, map_location=device,
                    weights_only=False)
    hp = cp["hyperparameters"]
    ns = {k: v.to(device) for k, v in cp["norm_stats"].items()}
    arch = cp.get("architecture", "")

    if "GATv2" in arch:
        model = GATv2Surrogate(
            in_dim=14, out_dim=6,
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            heads=hp.get("heads", 4))
        arch_name = "GATv2"
    else:
        model = GCNSurrogate(
            in_dim=14, out_dim=6,
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"])
        arch_name = "GCN"

    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Modell:      {arch_name}")
    print(f"  Hidden Dim:  {hp['hidden_dim']}")
    print(f"  Layers:      {hp['num_layers']}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter:   {n_params:,}")

    return model, ns, arch_name


def compute_r2(pred, true):
    """R² pro Feld und gesamt."""
    r2_per_field = {}
    for i, name in enumerate(OUTPUT_FEATURES):
        p, t = pred[:, i], true[:, i]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2_per_field[name] = 1.0 - ss_res / max(ss_tot, 1e-12)
    r2_per_field["gesamt"] = np.mean(
        [r2_per_field[n] for n in OUTPUT_FEATURES])
    return r2_per_field


@torch.no_grad()
def evaluate(model, graphs, norm_stats, device):
    """Berechnet R² auf allen Graphen (denormalisiert)."""
    all_pred, all_true = [], []
    y_mean = norm_stats["y_mean"]
    y_std = norm_stats["y_std"]

    for data in graphs:
        x = data.x.to(device)
        ei = data.edge_index.to(device)

        pred_norm = model(x, ei)
        pred = (pred_norm * y_std + y_mean).cpu().numpy()

        true = (data.y_orig.numpy()
                if hasattr(data, "y_orig")
                else data.y.numpy())

        all_pred.append(pred)
        all_true.append(true)

    return compute_r2(np.concatenate(all_pred),
                      np.concatenate(all_true))


# ==================================================================
# Permutation Importance
# ==================================================================

def run_permutation_importance(model, graphs, norm_stats, device,
                               n_repeats=5):
    """
    Berechnet Permutation Importance fuer alle 14 Input-Features.

    Fuer jedes Feature:
        1. Permutiere die normalisierten Werte ueber alle Knoten
        2. Berechne R² mit permutierten Daten
        3. Importance = R²_baseline - R²_permuted

    Parameter
    ---------
    model : nn.Module
    graphs : list[Data]
        Normalisierte Test-Graphen.
    norm_stats : dict
    device : torch.device
    n_repeats : int
        Anzahl Wiederholungen pro Feature.

    Rueckgabe
    ---------
    dict : {feature_name: {mean, std, scores}}
    """
    n_features = 14

    # Baseline
    print("\n  Baseline-Evaluation...")
    baseline = evaluate(model, graphs, norm_stats, device)
    print(f"  Baseline R² (gesamt): {baseline['gesamt']:.4f}")

    # Globale Features: pro Graph konstant, muessen zwischen
    # Graphen permutiert werden (nicht innerhalb)
    GLOBAL_FEATURES = {"U_ref", "angle"}

    results = {}

    for feat_idx in range(n_features):
        feat_name = INPUT_FEATURES[feat_idx]
        scores = []
        is_global = feat_name in GLOBAL_FEATURES

        for r in range(n_repeats):
            permuted_graphs = []

            if is_global:
                # Graph-Level Permutation: Werte zwischen
                # Graphen tauschen
                n_graphs = len(graphs)
                perm = torch.randperm(n_graphs)
                for i, data in enumerate(graphs):
                    data_p = data.clone()
                    src = graphs[perm[i].item()]
                    # Alle Knoten bekommen den Wert vom
                    # permutierten Graph
                    data_p.x[:, feat_idx] = src.x[0, feat_idx]
                    permuted_graphs.append(data_p)
            else:
                # Knoten-Level Permutation: Werte innerhalb
                # jedes Graphen shuffeln
                for data in graphs:
                    data_p = data.clone()
                    n_nodes = data_p.x.size(0)
                    perm = torch.randperm(n_nodes)
                    data_p.x[:, feat_idx] = data_p.x[perm, feat_idx]
                    permuted_graphs.append(data_p)

            r2_perm = evaluate(
                model, permuted_graphs, norm_stats, device)
            importance = baseline["gesamt"] - r2_perm["gesamt"]
            scores.append(importance)

            # Per-field importance fuer letzten Repeat speichern
            if r == n_repeats - 1:
                per_field = {
                    name: baseline[name] - r2_perm[name]
                    for name in OUTPUT_FEATURES
                }

        mean_imp = np.mean(scores)
        std_imp = np.std(scores)

        results[feat_name] = {
            "mean": float(mean_imp),
            "std": float(std_imp),
            "scores": [float(s) for s in scores],
            "per_field": {k: float(v) for k, v in per_field.items()},
        }

        print(f"  [{feat_idx + 1:2d}/14] {feat_name:<16s}  "
              f"ΔR² = {mean_imp:+.4f} ± {std_imp:.4f}")

    return baseline, results


# ==================================================================
# Visualisierung
# ==================================================================

def plot_importance(baseline, results, arch_name, output_dir):
    """Erstellt Permutation-Importance-Plot."""

    # Sortieren nach Importance (absteigend)
    sorted_features = sorted(
        results.keys(),
        key=lambda f: results[f]["mean"],
        reverse=True)

    means = [results[f]["mean"] for f in sorted_features]
    stds = [results[f]["std"] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_features))

    bars = ax.barh(y_pos, means, xerr=stds, height=0.6,
                   color="#2196F3", edgecolor="white",
                   capsize=3, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (ΔR²)", fontsize=12)
    ax.set_title(
        f"Permutation Importance — {arch_name} Medium h128\n"
        f"Baseline R² = {baseline['gesamt']:.4f}",
        fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    path = os.path.join(output_dir, "permutation_importance.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\n  Plot: {path}")

    # Per-Field Heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    field_matrix = np.zeros((len(sorted_features),
                             len(OUTPUT_FEATURES)))
    for i, feat in enumerate(sorted_features):
        for j, field in enumerate(OUTPUT_FEATURES):
            field_matrix[i, j] = results[feat]["per_field"][field]

    im = ax2.imshow(field_matrix, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(OUTPUT_FEATURES)))
    ax2.set_xticklabels(OUTPUT_FEATURES, fontsize=11)
    ax2.set_yticks(range(len(sorted_features)))
    ax2.set_yticklabels(sorted_features, fontsize=11)

    # Werte in Zellen schreiben
    for i in range(len(sorted_features)):
        for j in range(len(OUTPUT_FEATURES)):
            val = field_matrix[i, j]
            color = "white" if val > field_matrix.max() * 0.6 \
                else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=8, color=color)

    plt.colorbar(im, ax=ax2, label="ΔR²", shrink=0.8)
    ax2.set_title(
        f"Permutation Importance pro Feld — {arch_name}\n"
        f"(ΔR² bei Permutation des jeweiligen Input-Features)",
        fontsize=12, fontweight="bold")

    plt.tight_layout()
    path2 = os.path.join(output_dir,
                         "permutation_importance_heatmap.png")
    fig2.savefig(path2, dpi=200)
    plt.close(fig2)
    print(f"  Heatmap: {path2}")


# ==================================================================
# Hauptprogramm
# ==================================================================

def main():
    """CLI-Einstieg: laedt Modell und Graphen, berechnet Importance, speichert JSON und Plots."""
    parser = argparse.ArgumentParser(
        description="Permutation Importance fuer GNN-Surrogat")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--graph-source", type=str, required=True)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default="./permutation_importance")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    print(f"\n{'=' * 60}")
    print("  PERMUTATION IMPORTANCE")
    print(f"{'=' * 60}")
    print(f"  Geraet:     {device}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Repeats:    {args.n_repeats}")
    print()

    # Modell laden
    model, norm_stats, arch_name = load_model(
        args.checkpoint, device)

    # Test-Graphen laden und normalisieren
    print(f"\n  Lade Graphen: {args.graph_source}")
    graphs = torch.load(args.graph_source, map_location="cpu",
                        weights_only=False)
    print(f"  Anzahl: {len(graphs)}")

    x_mean = norm_stats["x_mean"].cpu()
    x_std = norm_stats["x_std"].cpu()

    for data in graphs:
        # Original-y sichern (unnormalisiert)
        data.y_orig = data.y.clone()
        # x normalisieren (wie im Training)
        data.x = (data.x - x_mean) / x_std

    # Permutation Importance berechnen
    baseline, results = run_permutation_importance(
        model, graphs, norm_stats, device,
        n_repeats=args.n_repeats)

    # Ergebnisse speichern
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": arch_name,
        "checkpoint": args.checkpoint,
        "n_test_graphs": len(graphs),
        "n_repeats": args.n_repeats,
        "baseline_r2": {k: float(v) for k, v in baseline.items()},
        "importance": results,
    }

    json_path = os.path.join(args.output_dir,
                             "permutation_importance.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Ergebnisse: {json_path}")

    # Plots
    plot_importance(baseline, results, arch_name, args.output_dir)

    print(f"\n  Fertig. Ergebnisse in {args.output_dir}/")


if __name__ == "__main__":
    main()
