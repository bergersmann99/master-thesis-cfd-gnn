"""
predict.py
==========
Inferenz-Skript fuer GNN-Surrogat-Modelle (GCN und GATv2).

Zwei Modi:
    1. predict:  Neue Windparameter (U_ref, angle) auf einem
                 existierenden Graph vorhersagen.
    2. eval:     Alle Test-Graphen vorhersagen, mit Ground Truth
                 vergleichen, Metriken berechnen, Fehler exportieren.

Das Modell (GCN oder GATv2, Standard oder Efficient) wird
automatisch aus dem Checkpoint erkannt.

Verwendung:
    # Vorhersage mit Config-Datei:
    python predict.py --config predict_config.yaml

    # Vorhersage mit CLI-Argumenten:
    python predict.py --mode predict \\
        --checkpoint ./output_gatv2_medium_h128/best_model.pt \\
        --graph-source ./graph_dataset/test.pt \\
        --graph-index 0 \\
        --U-ref 12.5 --angle 45.0 \\
        --output-dir ./vorhersage

    # Evaluation auf Testdaten:
    python predict.py --mode eval \\
        --checkpoint ./output_gatv2_medium_h128/best_model.pt \\
        --graph-source ./graph_dataset/test.pt \\
        --output-dir ./evaluation

Ausgabe:
    - vorhersage.vtu          VTU-Datei fuer ParaView
    - positions.npy           Knotenpositionen [N, 3]
    - prediction.npy          Vorhergesagte Felder [N, 6]
    - prediction_report.json  Inferenzzeit, Modellinfo, Parameter
    - (Eval-Modus zusaetzlich:)
      ground_truth.vtu        CFD-Loesung
      error.vtu               Absoluter und relativer Fehler
      eval_metrics.json       R², rL2 pro Feld

Siehe QUELLEN_predict.md fuer Quellenangaben.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from functools import partial


# ======================================================================
# Modell-Definitionen (identisch zu den Trainingsskripten)
# ======================================================================

class MLP(nn.Module):
    """MLP: 3x Linear + 2x ReLU (Pfaff et al., 2020)."""

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
    """GCN Processor: M x GCNConv + Residual + LayerNorm."""

    def __init__(self, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(
                in_channels=hidden_dim, out_channels=hidden_dim,
                add_self_loops=True, normalize=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0.0 and self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = norm(x)
            x = x + x_residual
        return x


class GATv2Processor(nn.Module):
    """GATv2 Processor: M x GATv2Conv + Residual + LayerNorm."""

    def __init__(self, hidden_dim, num_layers, heads=4, dropout=0.0,
                 attention_dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        assert hidden_dim % heads == 0
        dim_per_head = hidden_dim // heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim, out_channels=dim_per_head,
                heads=heads, concat=True, dropout=attention_dropout,
                add_self_loops=True, share_weights=False))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0.0 and self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = norm(x)
            x = x + x_residual
        return x


class GCNSurrogate(nn.Module):
    """Encode-Process-Decode mit GCNConv."""

    def __init__(self, in_dim=14, out_dim=6, hidden_dim=128,
                 num_layers=10, dropout=0.0):
        super().__init__()
        self.encoder = MLP(in_dim, hidden_dim, hidden_dim)
        self.processor = GCNProcessor(hidden_dim, num_layers, dropout)
        self.decoder = MLP(hidden_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = self.processor(h, edge_index)
        return self.decoder(h)


class GATv2Surrogate(nn.Module):
    """Encode-Process-Decode mit GATv2Conv."""

    def __init__(self, in_dim=14, out_dim=6, hidden_dim=128,
                 num_layers=10, heads=4, dropout=0.0,
                 attention_dropout=0.0):
        super().__init__()
        self.encoder = MLP(in_dim, hidden_dim, hidden_dim)
        self.processor = GATv2Processor(
            hidden_dim, num_layers, heads, dropout, attention_dropout)
        self.decoder = MLP(hidden_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = self.processor(h, edge_index)
        return self.decoder(h)


# ======================================================================
# Modell aus Checkpoint laden
# ======================================================================

def load_model_from_checkpoint(checkpoint_path, device):
    """
    Laedt ein Modell aus einem Checkpoint und erkennt automatisch
    die Architektur (GCN oder GATv2) anhand des gespeicherten
    architecture-Strings.

    Parameter
    ---------
    checkpoint_path : str
        Pfad zum best_model.pt Checkpoint.
    device : torch.device
        Rechengeraet (CPU/GPU).

    Rueckgabe
    ---------
    tuple : (model, norm_stats, hyperparameters, architecture_name)
    """
    if not os.path.exists(checkpoint_path):
        print(f"FEHLER: Checkpoint nicht gefunden: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    arch = checkpoint.get("architecture", "")
    hp = checkpoint["hyperparameters"]
    norm_stats = checkpoint["norm_stats"]

    # Normalisierungsstatistiken auf Device verschieben
    norm_stats = {k: v.to(device) for k, v in norm_stats.items()}

    hidden_dim = hp["hidden_dim"]
    num_layers = hp["num_layers"]
    dropout = hp.get("dropout", 0.0)

    # Input-Dimension aus norm_stats ableiten (unterstützt 13 oder 14 Features)
    in_dim = int(norm_stats["x_mean"].shape[0])

    if "GATv2" in arch:
        heads = hp.get("heads", 4)
        attn_dropout = hp.get("attention_dropout", 0.0)
        model = GATv2Surrogate(
            in_dim=in_dim, out_dim=6, hidden_dim=hidden_dim,
            num_layers=num_layers, heads=heads, dropout=dropout,
            attention_dropout=attn_dropout)
        arch_name = "GATv2"
    else:
        model = GCNSurrogate(
            in_dim=in_dim, out_dim=6, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout)
        arch_name = "GCN"

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modell geladen: {arch_name}")
    print(f"  Architektur:    {arch}")
    print(f"  Hidden Dim:     {hidden_dim}")
    print(f"  Layers:         {num_layers}")
    if "GATv2" in arch:
        print(f"  Heads:          {heads}")
    print(f"  Parameter:      {n_params:,}")
    print(f"  Checkpoint:     Epoche {checkpoint.get('epoch', '?')}")
    print(f"  Val-Loss:       {checkpoint.get('val_loss', '?')}")
    print("")

    return model, norm_stats, hp, arch_name


# ======================================================================
# VTU-Export
# ======================================================================

def export_vtu(positions, fields, field_names, filepath):
    """
    Exportiert Punktdaten als VTU-Datei fuer ParaView.

    Verwendet das VTK XML-Format (UnstructuredGrid) mit
    Vertex-Zellen. Jeder Knoten ist eine eigene Zelle vom
    Typ VTK_VERTEX (Typ-ID 1).

    Parameter
    ---------
    positions : np.ndarray
        (N, 3) Knotenpositionen.
    fields : dict
        {feldname: np.ndarray} mit Skalar- oder Vektorfeldern.
    field_names : list[str]
        Reihenfolge der Felder fuer die Ausgabe.
    filepath : str
        Ausgabepfad (.vtu).
    """
    n_points = len(positions)

    with open(filepath, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{n_points}" '
                f'NumberOfCells="{n_points}">\n')

        # Punktdaten (Felder)
        f.write('      <PointData>\n')
        for name in field_names:
            data = fields[name]
            if data.ndim == 1:
                n_comp = 1
            else:
                n_comp = data.shape[1]

            f.write(f'        <DataArray type="Float32" '
                    f'Name="{name}" '
                    f'NumberOfComponents="{n_comp}" '
                    f'format="ascii">\n')

            if n_comp == 1:
                for val in data:
                    f.write(f'          {val:.6e}\n')
            else:
                for row in data:
                    vals = " ".join(f"{v:.6e}" for v in row)
                    f.write(f'          {vals}\n')

            f.write('        </DataArray>\n')
        f.write('      </PointData>\n')

        # Punkte (Koordinaten)
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" '
                'NumberOfComponents="3" format="ascii">\n')
        for pt in positions:
            f.write(f'          {pt[0]:.6e} {pt[1]:.6e} '
                    f'{pt[2]:.6e}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Zellen (jeder Punkt ist eine Vertex-Zelle)
        f.write('      <Cells>\n')

        # Connectivity
        f.write('        <DataArray type="Int32" '
                'Name="connectivity" format="ascii">\n')
        for i in range(n_points):
            f.write(f'          {i}\n')
        f.write('        </DataArray>\n')

        # Offsets
        f.write('        <DataArray type="Int32" '
                'Name="offsets" format="ascii">\n')
        for i in range(1, n_points + 1):
            f.write(f'          {i}\n')
        f.write('        </DataArray>\n')

        # Types (1 = VTK_VERTEX)
        f.write('        <DataArray type="UInt8" '
                'Name="types" format="ascii">\n')
        for _ in range(n_points):
            f.write('          1\n')
        f.write('        </DataArray>\n')

        f.write('      </Cells>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')


# ======================================================================
# Normalisierung
# ======================================================================

def normalize_features(x, stats):
    """Normalisiert Input-Features mit z-Score."""
    return (x - stats["x_mean"]) / stats["x_std"]


def denormalize_output(y_norm, stats):
    """Denormalisiert Output-Features."""
    return y_norm * stats["y_std"] + stats["y_mean"]


# ======================================================================
# Vorhersage (einzelner Graph)
# ======================================================================

@torch.no_grad()
def predict_single(model, data, norm_stats, device):
    """
    Fuehrt eine Vorhersage auf einem einzelnen Graph durch.

    Parameter
    ---------
    model : nn.Module
        Trainiertes Modell.
    data : torch_geometric.data.Data
        Graph-Objekt (unnormalisiert).
    norm_stats : dict
        Normalisierungsstatistiken.
    device : torch.device
        Rechengeraet.

    Rueckgabe
    ---------
    dict : {
        'prediction': np.ndarray (N, 6) — denormalisiert,
        'positions': np.ndarray (N, 3),
        'inference_time_s': float,
    }
    """
    model.eval()

    # Normalisieren
    x_norm = normalize_features(data.x.to(device),
                                norm_stats)
    edge_index = data.edge_index.to(device)

    # GPU Warmup (erster Aufruf ist langsamer)
    if device.type == "cuda":
        _ = model(x_norm, edge_index[:, :100])
        torch.cuda.synchronize()

    # Inferenz mit Zeitmessung
    start = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()

    pred_norm = model(x_norm, edge_index)

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start

    # Denormalisieren
    pred = denormalize_output(pred_norm, norm_stats)

    # Positionen aus dem Graph
    if hasattr(data, "pos") and data.pos is not None:
        positions = data.pos.numpy()
    else:
        # Positionen aus den ersten 3 Input-Features
        positions = data.x[:, :3].numpy()

    return {
        "prediction": pred.cpu().numpy(),
        "positions": positions,
        "inference_time_s": inference_time,
    }


# ======================================================================
# Modus: Predict (neue Parameter)
# ======================================================================

def run_predict(model, norm_stats, hp, arch_name, cfg, device):
    """
    Vorhersage mit neuen Windparametern.

    Laedt einen existierenden Graph, ersetzt U_ref und angle,
    und fuehrt die Vorhersage durch.

    Parameter
    ---------
    model : nn.Module
    norm_stats : dict
    hp : dict
    arch_name : str
    cfg : dict
        Konfiguration (aus YAML oder CLI).
    device : torch.device
    """
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Graph laden
    graph_path = cfg["graph_source"]
    graph_index = cfg.get("graph_index", 0)
    new_U_ref = cfg["U_ref"]
    new_angle = cfg["angle"]

    print(f"  Lade Graph: {graph_path} [Index {graph_index}]")
    graphs = torch.load(graph_path, weights_only=False)
    if not isinstance(graphs, list):  # S3-Fallback-Graphen sind einzelne Data-Objekte
        graphs = [graphs]

    if graph_index >= len(graphs):
        print(f"FEHLER: graph_index {graph_index} ausserhalb "
              f"(max: {len(graphs) - 1})")
        sys.exit(1)

    data = graphs[graph_index]
    n_nodes = data.x.size(0)
    n_edges = data.edge_index.size(1)

    # Original-Parameter anzeigen
    orig_U_ref = data.x[0, 12].item()  # Feature-Index 12 = U_ref
    orig_angle = data.x[0, 13].item()  # Feature-Index 13 = angle

    print(f"  Original:  U_ref={orig_U_ref:.2f} m/s, "
          f"angle={orig_angle:.2f} deg")
    print(f"  Neu:       U_ref={new_U_ref:.2f} m/s, "
          f"angle={new_angle:.2f} deg")
    print(f"  Knoten:    {n_nodes:,}")
    print(f"  Kanten:    {n_edges:,}")
    print("")

    # U_ref und angle ersetzen
    data_modified = data.clone()
    data_modified.x[:, 12] = new_U_ref
    data_modified.x[:, 13] = new_angle

    # Vorhersage
    print(f"  Starte Inferenz...")
    result = predict_single(model, data_modified, norm_stats, device)

    pred = result["prediction"]
    positions = result["positions"]
    t_inf = result["inference_time_s"]

    print(f"  Inferenzzeit: {t_inf:.3f} s")
    print("")

    # Felder aufteilen
    field_names = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
    fields = {}
    for i, name in enumerate(field_names):
        fields[name] = pred[:, i]

    # Geschwindigkeitsvektor fuer ParaView
    fields["U"] = pred[:, :3]

    # Geschwindigkeitsbetrag
    fields["U_mag"] = np.linalg.norm(pred[:, :3], axis=1)

    export_fields = ["U", "U_mag", "p", "k", "epsilon"]

    # VTU exportieren
    if cfg.get("export_vtk", True):
        vtu_path = os.path.join(output_dir, "vorhersage.vtu")
        print(f"  Exportiere VTU: {vtu_path}")
        export_vtu(positions, fields, export_fields, vtu_path)

    # NumPy exportieren
    if cfg.get("export_numpy", True):
        pos_path = os.path.join(output_dir, "positions.npy")
        pred_path = os.path.join(output_dir, "prediction.npy")
        np.save(pos_path, positions)
        np.save(pred_path, pred)
        print(f"  Exportiere NumPy: {pos_path}, {pred_path}")

    # Bericht
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "predict",
        "architecture": arch_name,
        "checkpoint": cfg["checkpoint"],
        "hyperparameters": hp,
        "graph_source": graph_path,
        "graph_index": graph_index,
        "original_U_ref": orig_U_ref,
        "original_angle": orig_angle,
        "new_U_ref": new_U_ref,
        "new_angle": new_angle,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "inference_time_s": t_inf,
    }

    report_path = os.path.join(output_dir, "prediction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Bericht: {report_path}")

    print(f"\n  Fertig. Ergebnisse in {output_dir}/")


# ======================================================================
# Modus: Eval (Testdaten)
# ======================================================================

def compute_metrics(pred, true):
    """
    Berechnet Metriken pro Feld (denormalisiert).

    Parameter
    ---------
    pred : np.ndarray (N, 6)
    true : np.ndarray (N, 6)

    Rueckgabe
    ---------
    dict : Metriken pro Feld und gesamt.
    """
    field_names = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
    metrics = {}

    for i, name in enumerate(field_names):
        p = pred[:, i]
        t = true[:, i]

        mse = np.mean((t - p) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(t - p))

        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        rl2 = (np.sqrt(np.sum((p - t) ** 2))
               / max(np.sqrt(np.sum(t ** 2)), 1e-12))

        metrics[name] = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "rL2": float(rl2),
        }

    metrics["gesamt"] = {
        "MSE": np.mean([metrics[n]["MSE"] for n in field_names]),
        "RMSE": np.mean([metrics[n]["RMSE"] for n in field_names]),
        "MAE": np.mean([metrics[n]["MAE"] for n in field_names]),
        "R2": np.mean([metrics[n]["R2"] for n in field_names]),
        "rL2": np.mean([metrics[n]["rL2"] for n in field_names]),
    }

    return metrics


def run_eval(model, norm_stats, hp, arch_name, cfg, device):
    """
    Evaluation auf allen Test-Graphen.

    Parameter
    ---------
    model : nn.Module
    norm_stats : dict
    hp : dict
    arch_name : str
    cfg : dict
    device : torch.device
    """
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Test-Graphen laden
    graph_path = cfg["graph_source"]
    print(f"  Lade Test-Graphen: {graph_path}")
    test_graphs = torch.load(graph_path, weights_only=False)
    n_graphs = len(test_graphs)
    print(f"  Anzahl: {n_graphs}\n")

    field_names = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
    all_pred = []
    all_true = []
    total_inference_time = 0.0
    per_graph_results = []

    for i, data in enumerate(test_graphs):
        n_nodes = data.x.size(0)

        # Sim-ID falls vorhanden
        sim_id = getattr(data, "sim_id", f"graph_{i:03d}")

        sys.stdout.write(
            f"\r  [{i + 1}/{n_graphs}] {sim_id}: Inferenz...")
        sys.stdout.flush()

        # Vorhersage
        result = predict_single(model, data, norm_stats, device)
        pred = result["prediction"]
        positions = result["positions"]
        t_inf = result["inference_time_s"]
        total_inference_time += t_inf

        # Ground Truth denormalisieren
        y_mean = norm_stats["y_mean"].cpu()
        y_std = norm_stats["y_std"].cpu()
        # Die y-Daten sind noch unnormalisiert (direkt aus test.pt)
        true = data.y.numpy()

        all_pred.append(pred)
        all_true.append(true)

        # Pro-Graph Metriken
        graph_metrics = compute_metrics(pred, true)

        per_graph_results.append({
            "sim_id": sim_id,
            "n_nodes": n_nodes,
            "inference_time_s": t_inf,
            "R2": graph_metrics["gesamt"]["R2"],
            "rL2": graph_metrics["gesamt"]["rL2"],
        })

        # Pro-Graph VTU-Export
        if cfg.get("export_vtk", True):
            graph_dir = os.path.join(output_dir, sim_id)
            os.makedirs(graph_dir, exist_ok=True)

            # Vorhersage-Felder
            pred_fields = {}
            for j, name in enumerate(field_names):
                pred_fields[name] = pred[:, j]
            pred_fields["U"] = pred[:, :3]
            pred_fields["U_mag"] = np.linalg.norm(
                pred[:, :3], axis=1)

            export_vtu(positions, pred_fields,
                       ["U", "U_mag", "p", "k", "epsilon"],
                       os.path.join(graph_dir, "vorhersage.vtu"))

            # Ground Truth
            true_fields = {}
            for j, name in enumerate(field_names):
                true_fields[name] = true[:, j]
            true_fields["U"] = true[:, :3]
            true_fields["U_mag"] = np.linalg.norm(
                true[:, :3], axis=1)

            export_vtu(positions, true_fields,
                       ["U", "U_mag", "p", "k", "epsilon"],
                       os.path.join(graph_dir, "ground_truth.vtu"))

            # Fehler
            error_fields = {}
            for j, name in enumerate(field_names):
                abs_err = np.abs(pred[:, j] - true[:, j])
                rel_err = abs_err / np.maximum(
                    np.abs(true[:, j]), 1e-8)
                error_fields[f"{name}_abs_error"] = abs_err
                error_fields[f"{name}_rel_error"] = rel_err

            error_names = []
            for name in field_names:
                error_names.append(f"{name}_abs_error")
                error_names.append(f"{name}_rel_error")

            export_vtu(positions, error_fields, error_names,
                       os.path.join(graph_dir, "error.vtu"))

        print(f"\r  [{i + 1}/{n_graphs}] {sim_id}: "
              f"R²={graph_metrics['gesamt']['R2']:.4f}, "
              f"rL2={graph_metrics['gesamt']['rL2']:.4f}, "
              f"{t_inf:.3f}s{' ' * 20}")

    # Gesamt-Metriken
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    total_metrics = compute_metrics(all_pred, all_true)

    # Ergebnisse anzeigen
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION — {arch_name}")
    print(f"{'=' * 60}\n")

    print(f"  {'Feld':<10s} {'MSE':>12s} {'RMSE':>12s} "
          f"{'MAE':>12s} {'R²':>10s} {'rL2':>10s}")
    print(f"  {'-' * 66}")

    for name in field_names:
        m = total_metrics[name]
        print(f"  {name:<10s} {m['MSE']:>12.4f} "
              f"{m['RMSE']:>12.4f} {m['MAE']:>12.4f} "
              f"{m['R2']:>10.4f} {m['rL2']:>10.4f}")

    m = total_metrics["gesamt"]
    print(f"  {'-' * 66}")
    print(f"  {'GESAMT':<10s} {m['MSE']:>12.4f} "
          f"{m['RMSE']:>12.4f} {m['MAE']:>12.4f} "
          f"{m['R2']:>10.4f} {m['rL2']:>10.4f}")

    avg_time = total_inference_time / n_graphs
    print(f"\n  Gesamt-Inferenzzeit:     {total_inference_time:.2f} s")
    print(f"  Durchschnitt pro Graph:  {avg_time:.3f} s")
    print(f"  Speedup vs. CFD (~27min): ~{27 * 60 / avg_time:.0f}x")

    # Metriken speichern
    eval_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "eval",
        "architecture": arch_name,
        "checkpoint": cfg["checkpoint"],
        "hyperparameters": hp,
        "n_test_graphs": n_graphs,
        "total_inference_time_s": total_inference_time,
        "avg_inference_time_s": avg_time,
        "metrics": total_metrics,
        "per_graph": per_graph_results,
    }

    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Metriken: {metrics_path}")

    print(f"\n  Fertig. Ergebnisse in {output_dir}/")


# ======================================================================
# Hauptprogramm
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNN-Surrogat Inferenz und Evaluation")

    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML-Konfigurationsdatei")

    # CLI-Argumente (alternativ zu Config-Datei)
    parser.add_argument(
        "--mode", type=str, choices=["predict", "eval"],
        default="predict",
        help="Modus: predict (neue Parameter) oder "
             "eval (Testdaten)")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Pfad zum Modell-Checkpoint (best_model.pt)")
    parser.add_argument(
        "--graph-source", type=str, default=None,
        help="Pfad zu test.pt (oder anderem Graph-Split)")
    parser.add_argument(
        "--graph-index", type=int, default=0,
        help="Index des Graphs in test.pt "
             "(nur fuer predict-Modus)")
    parser.add_argument(
        "--U-ref", type=float, default=None,
        help="Neue Windgeschwindigkeit [m/s] "
             "(nur fuer predict-Modus)")
    parser.add_argument(
        "--angle", type=float, default=None,
        help="Neuer Windwinkel [Grad] "
             "(nur fuer predict-Modus)")
    parser.add_argument(
        "--output-dir", type=str, default="./prediction",
        help="Ausgabeverzeichnis")
    parser.add_argument(
        "--export-vtk", action="store_true", default=True,
        help="VTU-Dateien exportieren")
    parser.add_argument(
        "--export-numpy", action="store_true", default=True,
        help="NumPy-Arrays exportieren")
    parser.add_argument(
        "--no-vtk", action="store_true",
        help="VTU-Export deaktivieren")
    parser.add_argument(
        "--no-numpy", action="store_true",
        help="NumPy-Export deaktivieren")

    args = parser.parse_args()

    # Konfiguration zusammenbauen
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        if not args.checkpoint:
            print("FEHLER: --checkpoint oder --config angeben.")
            sys.exit(1)
        if not args.graph_source:
            print("FEHLER: --graph-source oder --config angeben.")
            sys.exit(1)

        cfg = {
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            "graph_source": args.graph_source,
            "graph_index": args.graph_index,
            "output_dir": args.output_dir,
            "export_vtk": not args.no_vtk,
            "export_numpy": not args.no_numpy,
        }

        if args.U_ref is not None:
            cfg["U_ref"] = args.U_ref
        if args.angle is not None:
            cfg["angle"] = args.angle

    # Device
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    # Header
    print(f"\n{'=' * 60}")
    print(f"  GNN SURROGAT — INFERENZ")
    print(f"{'=' * 60}")
    print(f"  Geraet:  {device}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"  Modus:   {cfg.get('mode', 'predict')}")
    print(f"")

    # Modell laden
    model, norm_stats, hp, arch_name = load_model_from_checkpoint(
        cfg["checkpoint"], device)

    # Modus ausfuehren
    mode = cfg.get("mode", "predict")

    if mode == "eval":
        run_eval(model, norm_stats, hp, arch_name, cfg, device)
    elif mode == "predict":
        if "U_ref" not in cfg or "angle" not in cfg:
            print("FEHLER: --U-ref und --angle angeben "
                  "(oder in Config-Datei).")
            sys.exit(1)
        run_predict(model, norm_stats, hp, arch_name, cfg, device)
    else:
        print(f"FEHLER: Unbekannter Modus '{mode}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
